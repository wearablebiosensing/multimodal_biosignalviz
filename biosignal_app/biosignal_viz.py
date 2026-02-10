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

def merge_annotations(df, annotations, x_col, use_index):
    merged = df.copy()
    merged['event_label'] = None
    for ann in annotations:
        start, end = ann['start_time'], ann['end_time']
        mask = (merged.index >= start) & (merged.index <= end) if use_index else (merged[x_col] >= start) & (merged[x_col] <= end)
        if mask.any(): merged.loc[mask, 'event_label'] = ann['label']
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

if app_mode == "Analysis Dashboard":
    if 'custom_labels' not in st.session_state: st.session_state.custom_labels = []

    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader("Upload BioSignal File", type=['csv', 'zip', 'acq', 'txt'])

    if uploaded_file is not None:
        df, detected_fs, native_anns = load_data(uploaded_file)
        
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
                            injected.append({"label": item.get("label", "Stress_Test"), "type": at, "start_time": item.get("time", item.get("start")), "end_time": item.get("time", item.get("end")), "notes": "Stress Test"})
                        st.session_state.stress_test_annotations = injected; st.success("Loaded Stress Test!")
                    except Exception as e: st.error(f"Error: {e}")

                ac1, ac2, ac3, ac4, ac5 = st.columns([1.5, 1, 1, 1.5, 1])
                with ac2: ann_type = st.selectbox("Type", ["Interval", "Instantaneous"])
                with ac1: ann_label = st.selectbox("Event Label", list(dict.fromkeys(st.session_state.custom_labels + ["Noise", "Artifact", "Arrhythmia", "R-wave"])))
                cur_x_start = start_row / (detected_fs or 1)
                with ac3: ann_start = st.number_input("Start (sec)", value=float(cur_x_start))
                with ac4: ann_end = st.number_input("End (sec)", value=float(cur_x_start + 1), disabled=(ann_type=="Instantaneous"))
                with ac5: st.write(""); add_btn = st.button("➕ Add Event")
                
                notes = st.text_area("Notes")
                if add_btn and ann_label:
                    s_idx = int(ann_start * (detected_fs or 1))
                    if save_annotation(current_doc_id, ann_start, ann_end if ann_type=="Interval" else ann_start, ann_label, notes, ann_type, sample_idx=s_idx):
                        st.success(f"Added {ann_label}"); time.sleep(0.5); st.rerun()

                if db_anns:
                    st.dataframe(pd.DataFrame(db_anns)[['label', 'type', 'start_time', 'end_time']], use_container_width=True, height=150)
                    del_id = st.selectbox("Delete Entry", [a['id'] for a in db_anns], format_func=lambda x: f"ID: {x[-6:]}")
                    if st.button("Confirm Delete"): delete_annotation(current_doc_id, del_id); st.rerun()

            # --- PLOTTING LOGIC WITH LEGEND & DASHED LINES ---
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

                        # Annotation Color Map & Legend Proxies
                        plot_anns = [a for a in all_current_anns if a['label'] in selected_labels]
                        px_colors = px.colors.qualitative.Prism
                        color_map = {lbl: px_colors[i % len(px_colors)] for i, lbl in enumerate(selected_labels)}
                        
                        # Add Legend Proxy Traces (Invisible points just for the legend)
                        for lbl in selected_labels:
                            fig.add_trace(go.Scatter(
                                x=[None], y=[None], mode='markers',
                                marker=dict(color=color_map[lbl], size=10, symbol='square'),
                                name=f"Event: {lbl}", showlegend=True, legendgroup="Annotations"
                            ))

                        for ann in plot_anns:
                            s_idx = ann.get('sample_idx', int(ann['start_time'] * (detected_fs or 1)))
                            if start_row <= s_idx <= end_row:
                                x_pos = s_idx if use_index else (s_idx / (detected_fs or 1))
                                color = color_map.get(ann['label'], "gray")
                                
                                if ann.get('type') == 'Instantaneous':
                                    # Restore Vertical Dashed Line
                                    fig.add_vline(x=x_pos, line_width=1.5, line_dash="dash", line_color=color, opacity=0.8)
                                    # Add Marker for interaction/hover
                                    try: y_max = df_slice[selected_columns[0]].max()
                                    except: y_max = 0
                                    fig.add_trace(go.Scattergl(
                                        x=[x_pos], y=[y_max], mode='markers', 
                                        marker=dict(color=color, size=10, symbol='diamond-tall'),
                                        hoverinfo='text', text=f"{ann['label']}: {ann.get('notes', '')}",
                                        showlegend=False
                                    ))
                                else:
                                    # Restore Interval Box
                                    e_idx = int(ann['end_time'] * (detected_fs or 1))
                                    x_end = e_idx if use_index else (e_idx / (detected_fs or 1))
                                    fig.add_vrect(x0=x_pos, x1=x_end, fillcolor=color, opacity=0.2, layer="below", line_width=0)

                        fig.update_layout(height=400*len(selected_columns) if view_mode == "Stacked" else 600, template="plotly_white", legend=dict(groupclick="toggleitem"))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"⚡ Plot Gen: {pm.duration*1000:.2f} ms")

            st.markdown("---")
            st.subheader("2. Advanced ECG Analysis")
            with st.expander("ECG Settings"):
                tgt = st.selectbox("ECG Col", [c for c in df.columns if any(x in c.lower() for x in ['ecg', 'mlii', 'v1'])] or df.columns)
                if st.button("Run Analysis"):
                    peaks, hr, _, clean = process_ecg(df[tgt].iloc[start_row:end_row], detected_fs or 500)
                    f1 = go.Figure(); f1.add_trace(go.Scattergl(y=clean, name='Signal', line_color='gray'))
                    f1.add_trace(go.Scattergl(x=peaks, y=clean[peaks], mode='markers', name='R-Peaks', marker_color='red'))
                    st.plotly_chart(f1, use_container_width=True)

elif app_mode == "Evaluation Experiment":
    st.header("🧪 Experiment: Visualization Latency Benchmark")
    uploaded_files = st.file_uploader("Upload Data Files", accept_multiple_files=True)
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for uf in uploaded_files:
            p = os.path.join(temp_dir, uf.name); os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f: f.write(uf.getbuffer())
        
        found = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.endswith('.hea'):
                    rec = os.path.join(root, f.replace('.hea', ''))
                    try:
                        h = wfdb.rdheader(rec)
                        found.append({'file': f, 'path': rec, 'type': 'WFDB', 'samples': h.sig_len})
                    except: pass
        
        if st.button("🚀 Run Benchmark"):
            sid = firebase_module.create_analysis_session("BENCHMARK")
            pb = st.progress(0)
            for i, f in enumerate(found):
                t1 = time.perf_counter()
                data, _ = wfdb.rdsamp(f['path'])
                fig = go.Figure(); fig.add_trace(go.Scatter(y=data[:10000, 0]))
                dur = (time.perf_counter() - t1) * 1000
                firebase_module.log_plot_performance(sid, f['file'], dur, 10000, 1)
                pb.progress((i+1)/len(found))
            st.success("Benchmark Complete!")