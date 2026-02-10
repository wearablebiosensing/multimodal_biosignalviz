import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
import scipy.stats as stats
import neurokit2 as nk
import wfdb
import bioread
import tempfile
import zipfile
import os
import shutil
import time
import traceback
import uuid
import gc
import warnings
import json

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Agent Import
try:
    import firebase_module
except ImportError:
    # Fallback for local testing
    class MockFirebase:
        def init_firebase(self): return None
        def create_analysis_session(self, *args): return "local-session"
        def log_visualization_metrics(self, *args): pass
        def log_computation_metrics(self, *args): pass
        def log_plot_performance(self, *args): pass
        def fetch_benchmark_results(self, *args): return pd.DataFrame()
        class PerformanceMonitor:
            def __enter__(self): 
                self.start = time.perf_counter()
                return self
            def __exit__(self, *args): self.duration = time.perf_counter() - self.start
    firebase_module = MockFirebase()

# -----------------------------------------------------------------------------
# Helper Functions (Data Processing)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    detected_annotations = []
    try:
        if file.name.lower().endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
            return df, None, []
        elif file.name.lower().endswith('.acq'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.acq') as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            try:
                data = bioread.read_file(tmp_path)
                df = pd.DataFrame()
                df['time'] = data.time_index
                for ch in data.channels:
                    col_name = ch.name if ch.name else f"Channel {ch.frequency}"
                    df[col_name] = ch.data
                return df, int(data.channels[0].samples_per_second), []
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
        elif file.name.lower().endswith('.txt'):
            df = pd.read_csv(file, sep=None, engine='python')
            return df, None, []
        elif file.name.lower().endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "temp.zip")
                with open(zip_path, "wb") as f:
                    f.write(file.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                header_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for filename in files:
                        if filename.endswith(".hea"):
                            header_files.append(os.path.join(root, filename))
                if not header_files: return None, None, []
                record_path = header_files[0].replace('.hea', '')
                signals, fields = wfdb.rdsamp(record_path)
                df = pd.DataFrame(signals, columns=fields['sig_name'])
                fs = fields['fs']
                df['time'] = np.arange(len(df)) / fs
                try:
                    if os.path.exists(record_path + '.atr'):
                        ann_obj = wfdb.rdann(record_path, 'atr')
                        for sample_idx, symbol in zip(ann_obj.sample, ann_obj.symbol):
                            t_sec = sample_idx / fs
                            label_map = {'N': 'Normal Beat', 'V': 'PVC', 'A': 'APC', 'L': 'LBBB', 'R': 'RBBB', '+': 'Rhythm Change', '~': 'Artifact'}
                            readable_label = label_map.get(symbol, f"Beat_{symbol}")
                            detected_annotations.append({
                                'id': str(uuid.uuid4()), 'start_time': t_sec, 'end_time': t_sec,
                                'sample_idx': sample_idx, 'label': readable_label, 'type': 'Instantaneous',
                                'notes': f"Native WFDB (Sym: {symbol})"
                            })
                except: pass
                return df, fs, detected_annotations
        return None, None, []
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, []

def process_ecg(data_series, fs, method="pantompkins1985", invert=False):
    try:
        series = pd.to_numeric(data_series, errors='coerce').interpolate().ffill().bfill().fillna(0)
        clean_data = series.values.astype(np.float64)
        if invert: clean_data = -clean_data
        ecg_cleaned = nk.ecg_clean(clean_data, sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
        peaks = info['ECG_R_Peaks']
        heart_rate = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len(clean_data)) if len(peaks) >= 2 else np.zeros(len(clean_data))
        return peaks, heart_rate, {}, ecg_cleaned
    except: return [], [], {}, np.zeros(10)

# --- ANNOTATION HELPERS ---
def save_annotation(session_id, start_t, end_t, label, notes, ann_type, sample_idx=None):
    db = firebase_module.init_firebase()
    if db and session_id:
        try:
            db.collection('analysis_logs').document(session_id).collection('annotations').add({
                'start_time': start_t, 'end_time': end_t, 'sample_idx': sample_idx,
                'label': label, 'type': ann_type, 'notes': notes,
                'created_at': firebase_module.firestore.SERVER_TIMESTAMP
            })
            return True
        except: return False
    return False

def get_annotations(session_id):
    db = firebase_module.init_firebase()
    anns = []
    if db and session_id:
        try:
            docs = db.collection('analysis_logs').document(session_id).collection('annotations').order_by('start_time').stream()
            for d in docs:
                dd = d.to_dict(); dd['id'] = d.id; anns.append(dd)
        except: pass
    return anns

def delete_annotation(session_id, ann_id):
    db = firebase_module.init_firebase()
    if db and session_id:
        try:
            db.collection('analysis_logs').document(session_id).collection('annotations').document(ann_id).delete()
        except: pass

def merge_annotations(df, annotations, x_col, use_index, fs=1):
    merged = df.copy()
    merged['event_label'] = None
    merged['event_type'] = None
    merged['event_notes'] = None
    
    for ann in annotations:
        ann_type = ann.get('type', 'Interval')
        s_idx = ann.get('sample_idx')
        if s_idx is None:
            s_idx = int(ann['start_time'] * fs)
        e_idx = int(ann['end_time'] * fs) if ann_type == 'Interval' else s_idx

        mask = None
        if use_index:
            if ann_type == 'Instantaneous':
                if 0 <= s_idx < len(merged): mask = merged.index == s_idx
            else:
                mask = (merged.index >= s_idx) & (merged.index <= e_idx)
        else:
            start_val = ann['start_time']
            end_val = ann['end_time']
            if ann_type == 'Instantaneous':
                try:
                    if pd.api.types.is_numeric_dtype(merged[x_col]):
                        idx = (merged[x_col] - start_val).abs().idxmin()
                        mask = merged.index == idx
                    else: mask = merged[x_col] == start_val
                except: pass
            else:
                mask = (merged[x_col] >= start_val) & (merged[x_col] <= end_val)
        
        if mask is not None and (mask.any() if hasattr(mask, 'any') else mask is not None):
            def append_str(current, new):
                if pd.isna(current) or current == "" or current is None: return new
                if new in str(current): return current
                return f"{current}; {new}"
            merged.loc[mask, 'event_label'] = merged.loc[mask, 'event_label'].apply(lambda x: append_str(x, ann['label']))
            merged.loc[mask, 'event_type'] = merged.loc[mask, 'event_type'].apply(lambda x: append_str(x, ann['type']))
            if ann.get('notes'):
                merged.loc[mask, 'event_notes'] = merged.loc[mask, 'event_notes'].apply(lambda x: append_str(x, ann['notes']))
    return merged

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
st.set_page_config(page_title="BioViz Studio", page_icon="📈", layout="wide")
st.title("BioViz Studio: High-Performance Signal Annotation")

with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio("Select Mode", ["Analysis Dashboard", "CSV Concatenator (Prep)", "Evaluation Experiment"])
    st.markdown("---")

if app_mode == "CSV Concatenator (Prep)":
    st.header("📂 Preprocessing: Concatenate CSV Files")
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Concatenate Files", type="primary"):
            st.warning("Concatenation logic placeholder.")

elif app_mode == "Analysis Dashboard":
    if 'custom_labels' not in st.session_state: st.session_state.custom_labels = []

    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader("Upload BioSignal File", type=['csv', 'zip', 'acq', 'txt'])

    if uploaded_file is not None:
        df, detected_fs, native_anns = load_data(uploaded_file)
        fs_val = detected_fs if detected_fs else 1
        
        if df is not None:
            if 'current_file_id' not in st.session_state or st.session_state.current_file_id != uploaded_file.name:
                session_id = firebase_module.create_analysis_session(uploaded_file.name)
                st.session_state.current_file_id = uploaded_file.name
                st.session_state.firebase_doc_id = session_id
                st.session_state.start_row = 0
                st.session_state.end_row = min(len(df), 5000)
                st.session_state.native_annotations = native_anns
                st.toast(f"✅ Loaded {len(native_anns)} native annotations!", icon="🧬")
            
            current_doc_id = st.session_state.get('firebase_doc_id')

            st.subheader("1. General Visualization & Annotation")
            use_index = st.checkbox("Use Row Index (Samples) as X-Axis", value=True)
            x_axis = "Row Index" if use_index else st.selectbox("Select X-Axis", list(df.columns))

            selected_columns = st.multiselect("Select Signals", options=df.columns, default=[c for c in df.columns if any(x in c.lower() for x in ['ecg', 'mlii', 'v1'])][:2])

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                slice_range = st.slider("Select Range", 0, len(df), (st.session_state.start_row, st.session_state.end_row), step=100)
                st.session_state.start_row, st.session_state.end_row = slice_range
                btn_prev, btn_next = st.columns(2)
                win = st.session_state.end_row - st.session_state.start_row
                if btn_prev.button("⬅️ Previous"):
                    st.session_state.start_row = max(0, st.session_state.start_row - win)
                    st.session_state.end_row = st.session_state.start_row + win; st.rerun()
                if btn_next.button("Next ➡️"):
                    st.session_state.start_row = min(len(df) - win, st.session_state.start_row + win)
                    st.session_state.end_row = st.session_state.start_row + win; st.rerun()
                start_row, end_row = st.session_state.start_row, st.session_state.end_row

            with col_ctrl2: downsample_rate = st.slider("Downsample", 1, 100, 1)
            with col_ctrl3: view_mode = st.radio("View Mode", ["Overlay", "Stacked"], horizontal=True)

            # --- Annotation Management ---
            db_anns = get_annotations(current_doc_id)
            all_current_anns = db_anns + st.session_state.native_annotations + st.session_state.get('stress_test_annotations', [])
            
            with st.sidebar:
                st.header("3. Annotation Filters")
                unique_labels = sorted(list(set([a['label'] for a in all_current_anns])))
                selected_labels = st.multiselect("Show in Legend/Plot", options=unique_labels, default=unique_labels)

            with st.expander("📝 Annotation Toolkit", expanded=False):
                st.markdown("#### 🛠️ Stress Test Injection")
                stress_file = st.file_uploader("Upload annotation_stress_test.json", type=['json'])
                if stress_file:
                    try:
                        s_data = json.load(stress_file); injected = []
                        for item in s_data:
                            at = "Instantaneous" if item.get("type") == "point" else "Interval"
                            injected.append({"label": item.get("label", "Stress_Test"), "type": at, "start_time": item.get("time", item.get("start")), "end_time": item.get("time", item.get("end")), "notes": "Stress Test Injection"})
                        st.session_state.stress_test_annotations = injected; st.success("Loaded Stress Test!")
                    except Exception as e: st.error(f"Error: {e}")

                ac1, ac2, ac3, ac4, ac5 = st.columns([1.5, 1, 1, 1.5, 1])
                with ac2: ann_type = st.selectbox("Type", ["Interval", "Instantaneous"])
                with ac1: ann_label = st.selectbox("Event Label", list(dict.fromkeys(st.session_state.custom_labels + ["Noise", "Artifact", "Arrhythmia", "R-wave"])))
                
                unit_label = "(Samples)" if use_index else "(sec)"
                def_start = start_row if use_index else (start_row / fs_val)
                def_end = end_row if use_index else (end_row / fs_val)
                
                with ac3: ann_start_input = st.number_input(f"Start {unit_label}", value=float(def_start))
                with ac4: ann_end_input = st.number_input(f"End {unit_label}", value=float(def_end), disabled=(ann_type=="Instantaneous"))
                with ac5: st.write(""); add_btn = st.button("➕ Add Event")
                
                notes = st.text_area("Notes")
                if add_btn and ann_label:
                    if use_index:
                        ann_start_sec, ann_end_sec = ann_start_input / fs_val, ann_end_input / fs_val
                        s_idx = int(ann_start_input)
                    else:
                        ann_start_sec, ann_end_sec = ann_start_input, ann_end_input
                        s_idx = int(ann_start_input * fs_val)
                    if save_annotation(current_doc_id, ann_start_sec, ann_end_sec if ann_type=="Interval" else ann_start_sec, ann_label, notes, ann_type, sample_idx=s_idx):
                        st.success(f"Added {ann_label}"); time.sleep(0.5); st.rerun()

                if db_anns:
                    df_db_anns = pd.DataFrame(db_anns)
                    st.dataframe(df_db_anns[['label', 'type', 'start_time', 'end_time']], use_container_width=True, height=150)
                    del_id = st.selectbox("Delete Entry", [a['id'] for a in db_anns], format_func=lambda x: f"ID: {x[-6:]}")
                    if st.button("Confirm Delete"): delete_annotation(current_doc_id, del_id); st.rerun()
                
                st.markdown("---")
                st.write("**Export Data**")
                col_export_1, col_export_2 = st.columns(2)
                with col_export_1:
                    if db_anns:
                        csv_anns = pd.DataFrame(db_anns).to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Annotations (CSV)", csv_anns, f"annotations_{current_doc_id}.csv", "text/csv", use_container_width=True)
                with col_export_2:
                    if st.checkbox("Prepare Merged Dataset (ML)"):
                        with st.spinner("Merging..."):
                            merged_df = merge_annotations(df, db_anns, x_axis, use_index, fs=fs_val)
                            csv_merged = merged_df.to_csv(index=use_index).encode('utf-8')
                            st.download_button("📦 Download Merged Dataset (CSV)", csv_merged, f"merged_{current_doc_id}.csv", "text/csv", type="primary", use_container_width=True)

            # --- PLOTTING LOGIC ---
            if selected_columns:
                with st.spinner("Rendering Plot..."):
                    with firebase_module.PerformanceMonitor() as pm:
                        df_slice = df.iloc[start_row:end_row:downsample_rate]
                        x_data = df_slice.index if use_index else df_slice[x_axis]
                        fig = make_subplots(rows=len(selected_columns), cols=1, shared_xaxes=True, vertical_spacing=0.05) if view_mode == "Stacked" else go.Figure()
                        for i, col in enumerate(selected_columns):
                            trace = go.Scattergl(x=x_data, y=df_slice[col], mode='lines', name=col)
                            if view_mode == "Stacked": fig.add_trace(trace, row=i+1, col=1)
                            else: fig.add_trace(trace)

                        plot_anns = [a for a in all_current_anns if a['label'] in selected_labels]
                        px_colors = px.colors.qualitative.Alphabet 
                        color_map = {lbl: px_colors[idx % len(px_colors)] for idx, lbl in enumerate(selected_labels)}
                        for lbl in selected_labels:
                            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color_map[lbl], size=12, symbol='square'), name=f"Event: {lbl}", showlegend=True, legendgroup="Annotations"))

                        for ann in plot_anns:
                            s_idx = ann.get('sample_idx', int(ann['start_time'] * fs_val))
                            e_idx = int(ann['end_time'] * fs_val) if ann.get('type') == 'Interval' else s_idx
                            if max(s_idx, start_row) <= min(e_idx, end_row):
                                x_pos = s_idx if use_index else (s_idx / fs_val)
                                color = color_map.get(ann['label'], "gray")
                                if ann.get('type') == 'Instantaneous':
                                    fig.add_vline(x=x_pos, line_width=1.5, line_dash="dash", line_color=color, opacity=0.9)
                                    try: y_max = df_slice[selected_columns[0]].max()
                                    except: y_max = 0
                                    fig.add_trace(go.Scattergl(x=[x_pos], y=[y_max], mode='markers', marker=dict(color=color, size=12, symbol='diamond-tall'), hoverinfo='text', text=f"{ann['label']}: {ann.get('notes', '')}", showlegend=False))
                                else:
                                    x_end = e_idx if use_index else (e_idx / fs_val)
                                    fig.add_vrect(x0=x_pos, x1=x_end, fillcolor=color, opacity=0.3, layer="below", line_width=0, row="all" if view_mode == "Stacked" else None)

                        dynamic_x_title = "Samples (N)" if use_index else (x_axis if not use_index else "Time")
                        medical_keywords = ['ecg', 'mlii', 'v1', 'eda', 'ppg']
                        is_medical = any(any(k in col.lower() for k in medical_keywords) for col in selected_columns)
                        dynamic_y_title = "Amplitude (mV)" if is_medical else "Magnitude"
                        fig.update_layout(height=400*len(selected_columns) if view_mode == "Stacked" else 600, template="plotly_white", legend=dict(groupclick="toggleitem", font=dict(size=16), itemsizing='constant'), xaxis_title=dict(text=dynamic_x_title, font=dict(size=18)), yaxis_title=dict(text=dynamic_y_title, font=dict(size=18)) if view_mode == "Overlay" else None, font=dict(size=14), margin=dict(t=50))
                        fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
                        fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))
                        if view_mode == "Stacked":
                            for i in range(len(selected_columns)): fig.update_yaxes(title_text=dynamic_y_title, row=i+1, col=1, title_font=dict(size=18))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"⚡ Plot Gen: {pm.duration*1000:.2f} ms")

            st.markdown("---")
            st.subheader("2. Advanced ECG Analysis")
            with st.expander("ECG Settings"):
                tgt = st.selectbox("ECG Col", [c for c in df.columns if any(x in c.lower() for x in ['ecg', 'mlii', 'v1'])] or df.columns)
                if st.button("Run Analysis"):
                    peaks, hr, _, clean = process_ecg(df[tgt].iloc[start_row:end_row], fs_val)
                    f1 = go.Figure(); f1.add_trace(go.Scattergl(y=clean, name='Signal', line_color='gray'))
                    f1.add_trace(go.Scattergl(x=peaks, y=clean[peaks], mode='markers', name='R-Peaks', marker_color='red'))
                    st.plotly_chart(f1, use_container_width=True)

# -----------------------------------------------------------------------------
# RESTORED: Evaluation Benchmarking
# -----------------------------------------------------------------------------
elif app_mode == "Evaluation Experiment":
    st.header("🧪 Experiment: Visualization Latency Benchmark")
    source_type = st.radio("Data Source", ["Local Path", "File Upload"], horizontal=True)
    dataset_path = None
    
    if source_type == "Local Path":
        dataset_path = st.text_input("Local Path", placeholder="/path/to/dataset").strip()
    else: 
        uploaded_files = st.file_uploader("Upload Data Files (CSV or WFDB Folder Content)", accept_multiple_files=True)
        if uploaded_files:
            if 'eval_temp_dir' not in st.session_state: st.session_state.eval_temp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                save_path = os.path.join(st.session_state.eval_temp_dir, uf.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f: f.write(uf.getbuffer())
            dataset_path = st.session_state.eval_temp_dir

    if dataset_path and os.path.exists(dataset_path):
        if 'eval_files' not in st.session_state: st.session_state.eval_files = []
        if st.button("Scan Directory"):
            found = []
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.endswith('.hea'):
                        rec_name = f.replace('.hea', '')
                        rec_path = os.path.join(root, rec_name)
                        try:
                            h = wfdb.rdheader(rec_path)
                            found.append({
                                'file': rec_name, 'path': rec_path, 'type': 'WFDB',
                                'duration_min': (h.sig_len/h.fs)/60 if h.fs else 0, 
                                'samples': h.sig_len, 'n_sig': h.n_sig
                            })
                        except: pass
                    elif f.endswith('.csv'):
                        try:
                            f_path = os.path.join(root, f)
                            df_tmp = pd.read_csv(f_path, nrows=2)
                            found.append({
                                'file': f, 'path': f_path, 'type': 'CSV',
                                'duration_min': 0, 'samples': 0, 'n_sig': len(df_tmp.columns)
                            })
                        except: pass
            st.session_state.eval_files = found
            st.success(f"Found {len(found)} records (CSV and WFDB supported).")

        if st.session_state.eval_files:
            st.subheader("⚙️ Benchmark Settings")
            with st.expander("📂 File Filtering", expanded=True):
                filter_term = st.text_input("Filter files by name substring", "")
                filtered_files = [f for f in st.session_state.eval_files if filter_term.lower() in f['file'].lower()]
                st.write(f"**Status:** {len(filtered_files)} files selected.")

            c1, c2, c3 = st.columns(3)
            with c1: n_trials = st.number_input("Trials per File", 1, 20, 5)
            with c2: n_ch = st.number_input("Channels to Render", 1, 20, 1)
            with c3: max_p = st.number_input("Max Points (0=All)", 0, 1000000, 0)

            if st.button("🚀 Start Benchmark"):
                if not filtered_files:
                    st.error("No files match filter.")
                else:
                    sid = firebase_module.create_analysis_session("BENCHMARK", "evaluation_experiment")
                    pb = st.progress(0); status = st.empty()
                    total_ops = len(filtered_files) * n_trials; curr_op = 0
                    
                    for f_info in filtered_files:
                        try:
                            status.write(f"Preparing data for {f_info['file']}...")
                            t_load_start = time.perf_counter()
                            if f_info['type'] == 'WFDB':
                                record, _ = wfdb.rdsamp(f_info['path'])
                                data_block = record
                            else:
                                df_bench = pd.read_csv(f_info['path'], engine='c', low_memory=False)
                                data_block = df_bench.select_dtypes(include=[np.number]).values
                                del df_bench; gc.collect()
                            t_load = time.perf_counter() - t_load_start
                            
                            for t in range(n_trials):
                                status.write(f"Benchmarking {f_info['file']} (Trial {t+1}/{n_trials})")
                                t1 = time.perf_counter()
                                fig = go.Figure()
                                end_idx = max_p if (max_p > 0 and max_p < len(data_block)) else len(data_block)
                                actual_ch = min(n_ch, data_block.shape[1])
                                for ch_idx in range(actual_ch):
                                    fig.add_trace(go.Scatter(y=data_block[:end_idx, ch_idx], mode='lines'))
                                t_plot = time.perf_counter() - t1
                                total_p = end_idx * actual_ch
                                
                                firebase_module.log_computation_metrics(sid, f_info['file'], f"bench_{f_info['type']}", t_load + t_plot, 0, (total_p / (t_load + t_plot)) / 1000)
                                firebase_module.log_plot_performance(sid, f_info['file'], t_plot * 1000, total_p, actual_ch)
                                curr_op += 1; pb.progress(curr_op / total_ops)
                            
                            del data_block; gc.collect()
                        except Exception as e:
                            st.error(f"Error on {f_info['file']}: {str(e)}")
                            curr_op += n_trials; pb.progress(min(1.0, curr_op / total_ops))
                    
                    st.success("Benchmark Finished"); st.balloons()
                    with st.spinner("Preparing Results CSV..."):
                        df_results = firebase_module.fetch_benchmark_results(sid)
                        if not df_results.empty:
                            st.download_button("📥 Download Benchmark Results (CSV)", df_results.to_csv(index=False).encode('utf-8'), f"benchmark_{sid}.csv", "text/csv", type="primary")