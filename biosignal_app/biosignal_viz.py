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

# Agent Import
# import agent
import firebase_module

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Biosignal Visualizer", page_icon="📈", layout="wide")
st.title("Biosignal CSV Visualizer & Preprocessor")

# -----------------------------------------------------------------------------
# Helper Functions (Data Processing)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    """
    Loads data from CSV, Zipped WFDB, Biopac (.acq), or Text (.txt) files.
    """
    try:
        # 1. Handle CSV
        if file.name.lower().endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
            return df, None
        
        # 2. Handle Biopac .ACQ
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
                return df, int(data.channels[0].samples_per_second)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        # 3. Handle Biopac .TXT (or generic text)
        elif file.name.lower().endswith('.txt'):
            try:
                df = pd.read_csv(file, sep=None, engine='python')
                if len(df.columns) < 2 or df.dtypes.iloc[1].kind not in 'iuf':
                    file.seek(0)
                    content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
                    header_row = 0
                    for i, line in enumerate(content[:50]):
                        if any(x in line.lower() for x in ['min', 'sec', 'time', 'ch', 'volts']):
                            header_row = i
                            break
                    file.seek(0)
                    df = pd.read_csv(file, sep=None, engine='python', skiprows=header_row)
                return df, None
            except Exception as e:
                st.error(f"Failed to parse text file: {e}")
                return None, None

        # 4. Handle Zipped WFDB
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
                
                if not header_files:
                    st.error("No .hea header file found in the zip archive.")
                    return None, None
                
                record_path = header_files[0].replace('.hea', '')
                
                try:
                    signals, fields = wfdb.rdsamp(record_path)
                    df = pd.DataFrame(signals, columns=fields['sig_name'])
                    fs = fields['fs']
                    df['time'] = np.arange(len(df)) / fs
                    return df, fs
                except Exception as e:
                    st.error(f"Failed to read WFDB record: {e}")
                    return None, None

        else:
            st.error("Unsupported file format.")
            return None, None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

def normalize_signal(data):
    data = np.nan_to_num(np.array(data, dtype=float))
    if np.std(data) == 0: return data - np.mean(data)
    return (data - np.mean(data)) / np.std(data)

def calculate_psd_ratio(segment, fs):
    segment = np.array(segment, dtype=float)
    if len(segment) < fs: return 0.0
    try:
        freqs, psd = signal.welch(segment, fs=fs, nperseg=min(len(segment), 256))
        idx_5_15 = np.logical_and(freqs >= 5, freqs <= 15)
        power_5_15 = np.sum(psd[idx_5_15])
        idx_0_45 = np.logical_and(freqs >= 0, freqs <= 45)
        power_0_45 = np.sum(psd[idx_0_45])
        if power_0_45 == 0: return 0
        return power_5_15 / power_0_45
    except: return 0.0

def process_ecg(data_series, fs, method="pantompkins1985", invert=False):
    try:
        series = pd.to_numeric(data_series, errors='coerce')
        series = series.interpolate(method='linear').ffill().bfill().fillna(0)
        clean_data = series.values.astype(np.float64)
        if invert: clean_data = -clean_data
        
        if len(clean_data) == 0: return [], [], {}, clean_data

        try:
            ecg_cleaned = nk.ecg_clean(clean_data, sampling_rate=fs, method="neurokit")
        except: ecg_cleaned = clean_data

        try:
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
            peaks = info['ECG_R_Peaks']
        except Exception:
            if method != "neurokit":
                _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="neurokit")
                peaks = info['ECG_R_Peaks']
            else:
                 return [], [], {}, ecg_cleaned 
        
        if len(peaks) < 2: 
            heart_rate = np.zeros(len(clean_data))
        else: 
            heart_rate = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len(clean_data))
        
        sqis = {
            'skewness': stats.skew(ecg_cleaned) if len(ecg_cleaned) > 0 else 0,
            'kurtosis': stats.kurtosis(ecg_cleaned) if len(ecg_cleaned) > 0 else 0,
            'psd_ratio': calculate_psd_ratio(ecg_cleaned, fs)
        }
        return peaks, heart_rate, sqis, ecg_cleaned

    except Exception as e:
        st.warning(f"ECG Error: {e}")
        return [], [], {}, np.zeros(10)

# --- ANNOTATION HELPERS ---
def save_annotation(session_id, start_t, end_t, label, notes, ann_type):
    db = firebase_module.init_firebase()
    if db and session_id:
        try:
            db.collection('analysis_logs').document(session_id).collection('annotations').add({
                'start_time': start_t,
                'end_time': end_t,
                'label': label,
                'type': ann_type,
                'notes': notes,
                'created_at': firebase_module.firestore.SERVER_TIMESTAMP
            })
            return True
        except Exception as e:
            st.error(f"Save failed: {e}")
            return False
    return False

def get_annotations(session_id):
    db = firebase_module.init_firebase()
    anns = []
    if db and session_id:
        try:
            docs = db.collection('analysis_logs').document(session_id).collection('annotations').order_by('start_time').stream()
            for d in docs:
                dd = d.to_dict()
                dd['id'] = d.id
                anns.append(dd)
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
    merged['event_type'] = None
    merged['event_notes'] = None
    
    for ann in annotations:
        start = ann['start_time']
        end = ann['end_time']
        ann_type = ann.get('type', 'Interval')
        
        mask = None
        if use_index:
            if ann_type == 'Instantaneous':
                idx = int(round(start))
                if 0 <= idx < len(merged): mask = merged.index == idx
            else:
                mask = (merged.index >= start) & (merged.index <= end)
        else:
            if ann_type == 'Instantaneous':
                try:
                    if pd.api.types.is_numeric_dtype(merged[x_col]):
                        idx = (merged[x_col] - start).abs().idxmin()
                        mask = merged.index == idx
                    else:
                        mask = merged[x_col] == start
                except: pass 
            else:
                mask = (merged[x_col] >= start) & (merged[x_col] <= end)
        
        if mask is not None and mask.any():
            def append_str(current, new):
                if pd.isna(current) or current == "": return new
                if new in str(current): return current
                return f"{current}; {new}"

            merged.loc[mask, 'event_label'] = merged.loc[mask, 'event_label'].apply(lambda x: append_str(x, ann['label']))
            merged.loc[mask, 'event_type'] = merged.loc[mask, 'event_type'].apply(lambda x: append_str(x, ann['type']))
            if ann.get('notes'):
                merged.loc[mask, 'event_notes'] = merged.loc[mask, 'event_notes'].apply(lambda x: append_str(x, ann['notes']))
                
    return merged

# -----------------------------------------------------------------------------
# App Navigation & Logic
# -----------------------------------------------------------------------------
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
    if 'annotations' not in st.session_state: st.session_state.annotations = []
    if 'custom_labels' not in st.session_state: st.session_state.custom_labels = []

    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader("Drag and drop file here", type=['csv', 'zip', 'acq', 'txt'])
        st.header("2. View Settings")

    if uploaded_file is not None:
        df, detected_fs = load_data(uploaded_file)
        
        if df is not None:
            if 'current_file_id' not in st.session_state or st.session_state.current_file_id != uploaded_file.file_id:
                session_id = firebase_module.create_analysis_session(uploaded_file.name)
                st.session_state.current_file_id = uploaded_file.file_id
                st.session_state.firebase_doc_id = session_id
                st.session_state.annotations = []
                # Initialize slice range
                st.session_state.start_row = 0
                st.session_state.end_row = min(len(df), 5000)
                if session_id: 
                    firebase_module.log_visualization_metrics(session_id, df, uploaded_file)
                    st.toast(f"Tracking ID: {session_id}", icon="🔥")
            
            current_doc_id = st.session_state.get('firebase_doc_id')

            with st.expander("📊 Dataset Statistics & Preview", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", f"{len(df):,}")
                col2.metric("Total Columns", len(df.columns))
                col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                if detected_fs: st.success(f"✅ Detected FS: **{detected_fs} Hz**")
                st.dataframe(df.head())

            st.subheader("1. General Visualization & Annotation")
            all_cols = list(df.columns)
            default_x_index = 0
            if 'samp_no' in all_cols: default_x_index = all_cols.index('samp_no')
            elif 'timestamp' in all_cols: default_x_index = all_cols.index('timestamp')
            elif 'time' in all_cols: default_x_index = all_cols.index('time')

            use_index = st.checkbox("Use Row Index as X-Axis", value=True)
            if not use_index: x_axis = st.selectbox("Select X-Axis Column", all_cols, index=default_x_index)
            else: x_axis = df.index.name if df.index.name else "Row Index"

            common_names = ['ecg', 'eda', 'ppg_ir', 'body_temp', 'i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
            default_selections = [c for c in common_names if any(x in c.lower() for x in df.columns)]
            if not default_selections:
                 default_selections = [c for c in df.columns if any(x in c.lower() for x in common_names)]
            
            selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns, default=default_selections[:4])

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1: 
                # Ensure session state exists
                if 'start_row' not in st.session_state: st.session_state.start_row = 0
                if 'end_row' not in st.session_state: st.session_state.end_row = min(len(df), 5000)

                # Slider for manual adjustment
                slice_range = st.slider(
                    "Select Row Range", 
                    0, len(df), 
                    (st.session_state.start_row, st.session_state.end_row), 
                    step=100
                )
                st.session_state.start_row, st.session_state.end_row = slice_range
                
                # Navigation Buttons
                btn_prev, btn_next = st.columns(2)
                window_size = st.session_state.end_row - st.session_state.start_row
                
                if btn_prev.button("⬅️ Previous", use_container_width=True):
                    new_start = max(0, st.session_state.start_row - window_size)
                    st.session_state.start_row = new_start
                    st.session_state.end_row = new_start + window_size
                    st.rerun()
                
                if btn_next.button("Next ➡️", use_container_width=True):
                    new_start = min(len(df) - window_size, st.session_state.start_row + window_size)
                    # Ensure start doesn't exceed bounds if window is larger than remaining data
                    if new_start < 0: new_start = 0
                    st.session_state.start_row = new_start
                    st.session_state.end_row = min(len(df), new_start + window_size)
                    st.rerun()

                start_row, end_row = st.session_state.start_row, st.session_state.end_row

            with col_ctrl2: downsample_rate = st.slider("Downsample", 1, 100, 1)
            with col_ctrl3: view_mode = st.radio("View Mode", ["Overlay", "Stacked"], horizontal=True)

            with st.expander("📝 Annotation Toolkit", expanded=False):
                label_file = st.file_uploader("Upload Custom Labels", type=['txt'])
                if label_file:
                    content = label_file.getvalue().decode("utf-8")
                    st.session_state.custom_labels = [l.strip() for l in content.split(',') if l.strip()]

                ac1, ac2, ac3, ac4, ac5 = st.columns([1.5, 1, 1, 1.5, 1])
                with ac2: ann_type = st.selectbox("Type", ["Interval", "Instantaneous"])
                with ac1:
                    defaults = ["Noise", "Motion Artifact", "Baseline Wander", "Signal Loss", "Arrhythmia", "Stress Event", "P-wave", "R-wave", "T-wave"] if ann_type == "Interval" else ["P-wave", "Q-wave", "R-wave", "S-wave", "T-wave", "Other"]
                    full_options = list(dict.fromkeys(st.session_state.custom_labels + defaults))
                    ann_label = st.selectbox("Event Label", full_options)
                
                cur_x_start = start_row if use_index else (df[x_axis].iloc[start_row] if not df.empty else 0)
                cur_x_end = end_row if use_index else (df[x_axis].iloc[end_row-1] if not df.empty and end_row > 0 else 0)
                
                with ac3: ann_start = st.number_input("Start", value=float(cur_x_start))
                with ac4: ann_end = st.number_input("End", value=float(cur_x_end), disabled=(ann_type=="Instantaneous"))
                with ac5: 
                    st.write("")
                    add_btn = st.button("➕ Add Event")

                ann_notes = st.text_area("Clinical Notes", height=68)
                if add_btn and ann_label:
                    if save_annotation(current_doc_id, ann_start, ann_end if ann_type == "Interval" else ann_start, ann_label, ann_notes, ann_type):
                        st.success(f"Added {ann_label}")
                        time.sleep(0.5)
                        st.rerun()

                existing_anns = get_annotations(current_doc_id)
                if existing_anns:
                    df_anns = pd.DataFrame(existing_anns)
                    st.dataframe(df_anns[['label', 'type', 'start_time', 'end_time']].rename(columns={'start_time': 'Start', 'end_time': 'End'}), use_container_width=True, height=150)
                    
                    del_col1, del_col2 = st.columns([3, 1])
                    with del_col1: del_id = st.selectbox("Select to Delete", df_anns['id'].tolist(), format_func=lambda x: f"ID: ...{x[-6:]}")
                    with del_col2:
                        if st.button("Delete"):
                            delete_annotation(current_doc_id, del_id)
                            st.rerun()

            if selected_columns:
                with st.spinner("Rendering..."):
                    valid_traces = 0
                    with firebase_module.PerformanceMonitor() as pm_plot:
                        df_slice = df.iloc[start_row:end_row]
                        if downsample_rate > 1: df_slice = df_slice.iloc[::downsample_rate, :]
                        x_data = df_slice.index if use_index else df_slice[x_axis]

                        if view_mode == "Stacked":
                            fig = make_subplots(rows=len(selected_columns), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=selected_columns)
                            for i, col in enumerate(selected_columns):
                                try:
                                    y_data = pd.to_numeric(df_slice[col], errors='coerce')
                                    fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='lines', name=col), row=i+1, col=1)
                                    valid_traces += 1
                                except: pass
                        else:
                            fig = go.Figure()
                            for col in selected_columns:
                                try:
                                    y_data = pd.to_numeric(df_slice[col], errors='coerce')
                                    fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='lines', name=col, opacity=0.8))
                                    valid_traces += 1
                                except: pass

                        # --- OVERLAY ANNOTATIONS & LEGEND FIX ---
                        if x_axis != "Row Index" or use_index:
                            anns = get_annotations(current_doc_id)
                            colors = px.colors.qualitative.Plotly
                            label_color_map = {}
                            
                            # Add "Dummy" traces for the legend
                            unique_labels = sorted(list(set(a['label'] for a in anns)))
                            for idx, lbl in enumerate(unique_labels):
                                color = colors[idx % len(colors)]
                                if "Artifact" in lbl or "Noise" in lbl: color = "rgba(128, 128, 128, 0.5)"
                                label_color_map[lbl] = color
                                
                                # Dummy trace for legend entry
                                fig.add_trace(go.Scatter(
                                    x=[None], y=[None], mode='markers',
                                    marker=dict(size=10, color=color, symbol='square'),
                                    legendgroup="annotations", legendgrouptitle_text="Annotations",
                                    name=lbl, showlegend=True
                                ))

                            for ann in anns:
                                color = label_color_map.get(ann['label'], "gray")
                                if ann.get('type') == 'Interval':
                                    fig.add_vrect(
                                        x0=ann['start_time'], x1=ann['end_time'],
                                        fillcolor=color, opacity=0.2,
                                        layer="below", line_width=0,
                                        annotation_text=ann['label'],
                                        annotation_position="top left"
                                    )
                                else:
                                    fig.add_vline(
                                        x=ann['start_time'],
                                        line_width=2, line_dash="dash", line_color=color,
                                        annotation_text=ann['label'],
                                        annotation_position="top left"
                                    )
                            
                        height = 300 * len(selected_columns) if view_mode == "Stacked" else 500
                        fig.update_layout(height=height, margin=dict(l=0,r=0,t=40,b=0), template="plotly_white", hovermode="x unified")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"⚡ Plot Gen: {pm_plot.duration*1000:.2f} ms | Points: {len(df_slice)*valid_traces:,}")

            st.markdown("---")
            st.subheader("2. Advanced ECG Analysis")
            with st.expander("⚙️ Settings", expanded=True):
                ecg_opts = [c for c in df.columns if any(x in c.lower() for x in ['ecg', 'i', 'ii', 'mlii', 'v1'])]
                tgt_col = st.selectbox("ECG Column", ecg_opts if ecg_opts else df.columns)
                fs_ecg = st.number_input("Fs (Hz)", value=int(detected_fs) if detected_fs else 500)
                algo = st.selectbox("Algorithm", ["pantompkins1985", "neurokit", "elgendi2010"], index=1)
                invert = st.checkbox("Invert Signal")
                
                # DATA RANGE SELECTOR
                analysis_range = st.radio(
                    "Select Analysis Scope", 
                    ["Current View (Selected Chunk)", "Full Dataset"], 
                    horizontal=True,
                    help="Choose whether to process only the points visible in the slider/chunk selection or the entire file."
                )

            if st.button("Run ECG Analysis"):
                # Determine data to process based on user choice
                if analysis_range == "Current View (Selected Chunk)":
                    seg_data = df[tgt_col].iloc[start_row:end_row]
                    scope_desc = f"chunk {start_row}-{end_row}"
                else:
                    seg_data = df[tgt_col]
                    scope_desc = "full dataset"
                
                seg = pd.to_numeric(seg_data, errors='coerce').dropna()
                
                if len(seg) >= fs_ecg:
                    with st.spinner(f"Processing {scope_desc}..."):
                        with firebase_module.PerformanceMonitor() as pm:
                            peaks, hr, sqis, clean = process_ecg(seg, fs_ecg, method=algo, invert=invert)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Execution Time", f"{pm.duration*1000:.1f} ms")
                        m2.metric("Detected Peaks", len(peaks))

                        t1, t2 = st.tabs(["Detection", "Heart Rate"])
                        with t1:
                            f1 = go.Figure()
                            f1.add_trace(go.Scattergl(y=clean, name='Clean Signal', line=dict(color='gray')))
                            vp = peaks[peaks < len(clean)]
                            f1.add_trace(go.Scattergl(x=vp, y=clean[vp], mode='markers', name='R-Peaks', marker=dict(color='red', size=8, symbol='x')))
                            st.plotly_chart(f1, use_container_width=True)
                        with t2:
                            f2 = go.Figure()
                            f2.add_trace(go.Scattergl(y=hr, name='Heart Rate (BPM)', line=dict(color='red')))
                            st.plotly_chart(f2, use_container_width=True)
                else:
                    st.error(f"Signal segment too short ({len(seg)} samples) for analysis at {fs_ecg}Hz.")

    else:
        st.info("Waiting for file upload...")

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
                with open(os.path.join(st.session_state.eval_temp_dir, uf.name), "wb") as f: f.write(uf.getbuffer())
            dataset_path = st.session_state.eval_temp_dir

    if dataset_path and os.path.exists(dataset_path):
        if 'eval_files' not in st.session_state: st.session_state.eval_files = []
        if st.button("Scan Directory"):
            found = []
            files = os.listdir(dataset_path)
            # Support for both WFDB (.hea) and MindGame (.csv)
            for f in files:
                if f.endswith('.hea'):
                    rec = f.replace('.hea', '')
                    try:
                        h = wfdb.rdheader(os.path.join(dataset_path, rec))
                        found.append({
                            'file': rec, 
                            'path': os.path.join(dataset_path, rec), 
                            'type': 'WFDB',
                            'duration_min': (h.sig_len/h.fs)/60, 
                            'samples': h.sig_len,
                            'n_sig': h.n_sig
                        })
                    except: pass
                elif f.endswith('.csv'):
                    try:
                        f_path = os.path.join(dataset_path, f)
                        # Metadata read
                        df_tmp = pd.read_csv(f_path, nrows=2)
                        found.append({
                            'file': f,
                            'path': f_path,
                            'type': 'CSV',
                            'duration_min': 0, 
                            'samples': 0, 
                            'n_sig': len(df_tmp.columns)
                        })
                    except: pass
            
            st.session_state.eval_files = found
            if not found:
                st.warning("No compatible records found. Ensure you uploaded .csv files or .hea/.dat pairs.")
            else:
                st.success(f"Found {len(found)} records (CSV and WFDB supported).")

        if st.session_state.eval_files:
            st.subheader("⚙️ Benchmark Settings")
            c1, c2, c3 = st.columns(3)
            with c1: n_trials = st.number_input("Trials per File", 1, 20, 5)
            with c2: n_ch = st.number_input("Channels to Render", 1, 20, 1)
            with c3: max_p = st.number_input("Max Points (0=All)", 0, 1000000, 0)

            if st.button("🚀 Start Benchmark"):
                sid = firebase_module.create_analysis_session("BENCHMARK", "evaluation_experiment")
                pb = st.progress(0)
                status = st.empty()
                files = st.session_state.eval_files
                total_ops = len(files) * n_trials
                curr_op = 0
                
                for i, f_info in enumerate(files):
                    # --- OPTIMIZATION: LOAD DATA ONCE PER FILE ---
                    try:
                        status.write(f"Preparing data for {f_info['file']}...")
                        t_load_start = time.perf_counter()
                        if f_info['type'] == 'WFDB':
                            record, _ = wfdb.rdsamp(f_info['path'])
                            data_block = record
                        else:
                            # Use engine='c' and low_memory for better stability on large files
                            df_bench = pd.read_csv(f_info['path'], engine='c', low_memory=False)
                            data_block = df_bench.select_dtypes(include=[np.number]).values
                            del df_bench # Explicitly free memory
                            gc.collect() # Trigger garbage collection for large files
                        t_load = time.perf_counter() - t_load_start
                    except Exception as e:
                        st.error(f"Failed to load {f_info['file']}: {str(e)}")
                        curr_op += n_trials # Skip trials
                        pb.progress(curr_op / total_ops)
                        continue

                    # Benchmark Trials (Visualization Latency)
                    for t in range(n_trials):
                        try:
                            status.write(f"Benchmarking {f_info['file']} (Trial {t+1}/{n_trials})")
                            
                            # RENDERING BENCHMARK
                            t1 = time.perf_counter()
                            fig = go.Figure()
                            end_idx = max_p if (max_p > 0 and max_p < len(data_block)) else len(data_block)
                            actual_ch = min(n_ch, data_block.shape[1])
                            
                            for ch_idx in range(actual_ch):
                                y = data_block[:end_idx, ch_idx]
                                fig.add_trace(go.Scatter(y=y, mode='lines'))
                                
                            t_plot = time.perf_counter() - t1
                            total_points = end_idx * actual_ch
                            
                            # LOGGING
                            firebase_module.log_computation_metrics(
                                sid, f_info['file'], f"bench_{f_info['type']}", 
                                t_load + t_plot, 0, (total_points / (t_load + t_plot)) / 1000
                            )
                            firebase_module.log_plot_performance(
                                sid, f_info['file'], t_plot * 1000, total_points, actual_ch
                            )
                        except Exception as e:
                            st.error(f"Error on {f_info['file']} trial {t}: {str(e)}")
                        
                        curr_op += 1
                        pb.progress(curr_op / total_ops)
                    
                    # Cleanup after all trials for this file
                    del data_block
                    gc.collect()
                
                status.write("Benchmark Complete.")
                st.success("Benchmark Finished")
                st.balloons()
                
                # --- DOWNLOAD LOGIC RESTORED ---
                with st.spinner("Preparing Results CSV..."):
                    df_results = firebase_module.fetch_benchmark_results(sid)
                    if df_results is not None and not df_results.empty:
                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Benchmark Results (CSV)",
                            data=csv,
                            file_name=f"benchmark_results_{sid}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                    else:
                        st.warning("No results found to download. Verify Firestore connection.")