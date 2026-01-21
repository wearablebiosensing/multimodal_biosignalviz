import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
import traceback # Added for detailed error tracking

# Agent Import
import agent
import firebase_module

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Biosignal Visualizer", page_icon="📈", layout="wide")
st.title("🫀 Biosignal CSV Visualizer & Preprocessor")

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
            return df
        
        # 2. Handle Biopac .ACQ
        elif file.name.lower().endswith('.acq'):
            # Bioread requires a physical file path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.acq') as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            
            try:
                data = bioread.read_file(tmp_path)
                df = pd.DataFrame()
                
                # Biopac files have a time index
                df['time'] = data.time_index
                
                # Extract channels
                for ch in data.channels:
                    # Clean channel name (remove special chars if needed)
                    col_name = ch.name if ch.name else f"Channel {ch.frequency}"
                    df[col_name] = ch.data
                
                return df
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        # 3. Handle Biopac .TXT (or generic text)
        elif file.name.lower().endswith('.txt'):
            # Biopac text exports are often tab-delimited or header-heavy
            # Try reading with common separators
            try:
                # Attempt to detect Biopac header (usually first X lines)
                # For now, try standard pandas robust read
                df = pd.read_csv(file, sep=None, engine='python')
                
                # Check if we read metadata as columns (common in Biopac exports)
                # If column 0 is "min" or "sec" or numeric, it's likely good.
                # If many columns are object type, we might need to skip rows.
                if len(df.columns) < 2 or df.dtypes.iloc[1].kind not in 'iuf':
                    # Retry skipping header lines (Biopac often has ~10-20 header lines)
                    file.seek(0)
                    # Heuristic: Find header row starting with "min", "sec", or "time"
                    content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
                    header_row = 0
                    for i, line in enumerate(content[:50]):
                        if any(x in line.lower() for x in ['min', 'sec', 'time', 'ch', 'volts']):
                            header_row = i
                            break
                    
                    file.seek(0)
                    df = pd.read_csv(file, sep=None, engine='python', skiprows=header_row)

                return df
            except Exception as e:
                st.error(f"Failed to parse text file: {e}")
                return None

        # 4. Handle Zipped WFDB (common for PhysioNet downloads)
        elif file.name.lower().endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save and extract zip
                zip_path = os.path.join(temp_dir, "temp.zip")
                with open(zip_path, "wb") as f:
                    f.write(file.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find header files (.hea)
                header_files = [f for f in os.listdir(temp_dir) if f.endswith('.hea')]
                
                if not header_files:
                    st.error("No .hea header file found in the zip archive.")
                    return None
                
                # Assume single record for simplicity, or take the first one
                record_name = header_files[0].replace('.hea', '')
                record_path = os.path.join(temp_dir, record_name)
                
                try:
                    # Read WFDB record
                    signals, fields = wfdb.rdsamp(record_path)
                    
                    # Convert to DataFrame
                    # fields['sig_name'] contains channel names
                    df = pd.DataFrame(signals, columns=fields['sig_name'])
                    
                    # Add time column based on sampling frequency
                    fs = fields['fs']
                    df['time'] = np.arange(len(df)) / fs
                    
                    return df
                    
                except Exception as e:
                    st.error(f"Failed to read WFDB record: {e}")
                    return None

        else:
            st.error("Unsupported file format. Please upload CSV, ACQ, TXT, or Zipped WFDB.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def normalize_signal(data):
    data = np.array(data, dtype=float)
    if np.std(data) == 0: return data - np.mean(data)
    return (data - np.mean(data)) / np.std(data)

def calculate_psd_ratio(segment, fs):
    segment = np.array(segment, dtype=float)
    # Handle short segments to avoid Welch errors
    if len(segment) < fs: 
        return 0.0
        
    try:
        freqs, psd = signal.welch(segment, fs=fs, nperseg=min(len(segment), 256))
        idx_5_15 = np.logical_and(freqs >= 5, freqs <= 15)
        power_5_15 = np.sum(psd[idx_5_15])
        idx_0_45 = np.logical_and(freqs >= 0, freqs <= 45)
        power_0_45 = np.sum(psd[idx_0_45])
        if power_0_45 == 0: return 0
        return power_5_15 / power_0_45
    except Exception:
        return 0.0

def process_ecg(data_series, fs, method="pantompkins1985"):
    """
    Processes ECG signal using NeuroKit2 with selectable method.
    Includes robust error handling and fallback for noisy data (NSTDB).
    """
    # 1. Input Validation & Sanitization
    try:
        # Coerce to numeric, handle NaNs with interpolation for continuity, then fill remaining
        series = pd.to_numeric(data_series, errors='coerce')
        series = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').fillna(0)
        # FORCE FLOAT64 to prevent int overflows in signal libraries
        clean_data = series.values.astype(np.float64)
        
        # Check for flatline or empty
        if len(clean_data) == 0 or np.std(clean_data) < 1e-9:
            st.warning("Signal is flat or empty. Cannot process.")
            return [], [], {}, clean_data

        # Check for Infinites (common in stress test data artifacts)
        clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.error(f"Data Preparation Error: {e}")
        return [], [], {}, np.zeros(10)

    # 2. Cleaning
    try:
        # Use 'neurokit' cleaning (0.5Hz Highpass + Powerline filter)
        ecg_cleaned = nk.ecg_clean(clean_data, sampling_rate=fs, method="neurokit")
        # Check if filter exploded (returns NaNs on extremely noisy data)
        if np.isnan(ecg_cleaned).any() or np.isinf(ecg_cleaned).any():
             ecg_cleaned = np.nan_to_num(ecg_cleaned)
    except Exception as e:
        st.warning(f"Signal cleaning failed ({str(e)}). Using raw signal for detection.")
        ecg_cleaned = clean_data

    # 3. Find Peaks using SELECTED algorithm with MULTI-STAGE FALLBACK
    peaks = []
    try:
        # Attempt 1: User Selection
        # Fix: nk.ecg_peaks returns (signals_df, info_dict). 
        # The indices are in info_dict['ECG_R_Peaks']
        _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
        peaks = info['ECG_R_Peaks']
    except Exception as e_primary:
        # Attempt 2: Fallback to NeuroKit (Gradient) - Robust to noise
        try:
            if method != "neurokit":
                # st.info(f"Method '{method}' failed. Retrying with 'neurokit' adaptive algorithm.")
                _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="neurokit")
                peaks = info['ECG_R_Peaks']
            else:
                 raise e_primary 
        except Exception as e_secondary:
            # Attempt 3: Hail Mary - Simple SciPy Threshold
            try:
                # Basic heuristic: Peaks > 75th percentile, min distance 0.3s
                thresh = np.percentile(ecg_cleaned, 75)
                min_dist = int(0.3 * fs)
                peaks, _ = signal.find_peaks(ecg_cleaned, height=thresh, distance=min_dist)
                st.warning(f"Advanced algorithms failed (Error: {e_secondary}). Using basic threshold detection.")
            except Exception as e_final:
                # Capture full traceback
                err_trace = traceback.format_exc()
                st.error(f"All detection methods failed.")
                with st.expander("See Error Trace"):
                    st.code(err_trace)
                return [], [], {}, ecg_cleaned
    
    # 4. Calculate Metrics
    try:
        # Handle case where too few peaks are found
        # nk.signal_rate needs at least 2 peaks
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
        # Catch-all for calculation failures
        st.error(f"Metric calculation failed: {e}")
        with st.expander("Metric Error Trace"):
            st.code(traceback.format_exc())
        return peaks, np.zeros(len(clean_data)), {}, ecg_cleaned

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def decompose_eda(data, fs, scl_cutoff=0.05):
    n = len(data)
    fft_vals = np.fft.fft(data)
    freqs = np.fft.fftfreq(n, d=1/fs)
    mask = np.abs(freqs) < scl_cutoff
    scl_fft = fft_vals * mask
    scl = np.real(np.fft.ifft(scl_fft))
    scr = data - scl
    return scl, scr

def process_eda_signal(data_series, fs):
    raw_values = np.array(data_series.values, dtype=float)
    raw_mean = np.mean(raw_values)
    raw_std = np.std(raw_values)
    if raw_std == 0: raw_std = 1
    
    z_scored = (raw_values - raw_mean) / raw_std
    sd_window_sec = 60
    sd_threshold = 0.01
    slope_tolerance = 1e-4
    
    window_samples = int(sd_window_sec * fs)
    rolling_std = pd.Series(z_scored).rolling(window=window_samples, center=True).std().fillna(method='bfill').fillna(method='ffill').values
    grads = np.gradient(z_scored) * fs
    is_artifact = (rolling_std < sd_threshold) | (np.abs(grads) > slope_tolerance)
    
    clean_z = z_scored.copy()
    clean_z[is_artifact] = np.nan
    clean_z = pd.Series(clean_z).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    cleaned_signal = (clean_z * raw_std) + raw_mean
    
    filtered_1_5 = butter_lowpass_filter(cleaned_signal, 1.5, fs, order=8)
    filtered_10 = butter_lowpass_filter(cleaned_signal, 10, fs, order=4)
    scl, scr = decompose_eda(filtered_1_5, fs, scl_cutoff=0.05)
    thresh = np.std(scr) * 0.07
    peaks, _ = signal.find_peaks(scr, height=thresh, distance=int(0.5*fs))
    
    return {
        'time': np.arange(len(raw_values))/fs,
        'raw': raw_values,
        'cleaned': cleaned_signal,
        'filtered_10hz': filtered_10,
        'filtered_1_5hz': filtered_1_5,
        'scl': scl,
        'scr': scr,
        'scr_peaks': peaks
    }

# -----------------------------------------------------------------------------
# App Navigation & Logic
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio("Select Mode", ["Analysis Dashboard", "CSV Concatenator (Prep)", "Evaluation Experiment"])
    st.markdown("---")
    
    # Initialize Agent
    agent_active = agent.init_agent()
    if agent_active:
        st.success("🟢 Dr. Signal (AI) Active")
    else:
        st.warning("🔴 Dr. Signal Offline")

if app_mode == "CSV Concatenator (Prep)":
    st.header("📂 Preprocessing: Concatenate CSV Files")
    uploaded_files = st.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Concatenate Files", type="primary"):
            st.warning("Concatenation logic placeholder for this demo.")

elif app_mode == "Analysis Dashboard":
    with st.sidebar:
        st.header("1. Data Input")
        # Allow CSV, ZIP, ACQ, TXT uploads
        uploaded_file = st.file_uploader(
            "Drag and drop file here", 
            type=['csv', 'zip', 'acq', 'txt'],
            help="Supports: CSV, Biopac (.acq, .txt), and PhysioNet WFDB (.zip)"
        )
        st.header("2. View Settings")
        if uploaded_file is not None:
            st.info("💡 Tip: For large files (1M+ rows), use 'Slice Data' to zoom in.")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # --- SESSION MANAGEMENT ---
            if 'current_file_id' not in st.session_state or st.session_state.current_file_id != uploaded_file.file_id:
                session_id = firebase_module.create_analysis_session(uploaded_file.name)
                st.session_state.current_file_id = uploaded_file.file_id
                st.session_state.firebase_doc_id = session_id
                if session_id: 
                    firebase_module.log_visualization_metrics(session_id, df, uploaded_file)
                    st.toast(f"Tracking ID: {session_id}", icon="🔥")
            
            current_doc_id = st.session_state.get('firebase_doc_id')

            with st.expander("📊 Dataset Statistics & Preview", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", f"{len(df):,}")
                col2.metric("Total Columns", len(df.columns))
                col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.dataframe(df.head())

            st.subheader("1. General Visualization")
            all_cols = list(df.columns)
            default_x_index = 0
            if 'samp_no' in all_cols: default_x_index = all_cols.index('samp_no')
            elif 'timestamp' in all_cols: default_x_index = all_cols.index('timestamp')
            elif 'time' in all_cols: default_x_index = all_cols.index('time') # Added for WFDB/Biopac

            use_index = st.checkbox("Use Row Index as X-Axis", value=True)
            if not use_index: x_axis = st.selectbox("Select X-Axis Column", all_cols, index=default_x_index)
            else: x_axis = df.index.name if df.index.name else "Row Index"

            # Pre-select common columns including standard lead names
            common_names = ['ecg', 'eda', 'ppg_ir', 'body_temp', 'i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
            default_selections = [c for c in common_names if any(x in c.lower() for x in df.columns)]
            # If nothing matched (e.g. strict naming), fallback to just columns
            if not default_selections:
                 default_selections = [c for c in df.columns if any(x in c.lower() for x in common_names)]
            
            selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns, default=default_selections[:4]) # Limit default to 4

            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1: 
                start_row, end_row = st.slider(
                    "Select Row Range (Slice)", 
                    min_value=0, 
                    max_value=len(df), 
                    value=(0, min(len(df), 5000)), 
                    step=100,
                    key="slice_slider" 
                )
            with col_ctrl2: 
                downsample_rate = st.slider("Downsample Rate", min_value=1, max_value=1000, value=1)

            if selected_columns:
                with st.spinner("Generating Plot..."):
                    with firebase_module.PerformanceMonitor() as pm_plot:
                        df_slice = df.iloc[start_row:end_row]
                        if downsample_rate > 1: df_slice = df_slice.iloc[::downsample_rate, :]
                        fig = go.Figure()
                        
                        valid_traces = 0
                        for col in selected_columns:
                            try:
                                y_data = pd.to_numeric(df_slice[col], errors='coerce')
                                x_data = df_slice.index if use_index else df_slice[x_axis]
                                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=col, opacity=0.8))
                                valid_traces += 1
                            except: pass
                        
                        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    plot_time_ms = pm_plot.duration * 1000
                    total_render_points = len(df_slice) * valid_traces
                    
                    st.caption(f"⚡ Plot Gen: {plot_time_ms:.2f} ms | Points: {total_render_points:,} | Signals: {valid_traces}")
                    
                    if current_doc_id:
                        firebase_module.log_plot_performance(
                            doc_id=current_doc_id,
                            file_name=uploaded_file.name,
                            exec_time_ms=plot_time_ms,
                            total_points=total_render_points,
                            active_traces=valid_traces
                        )

            st.markdown("---")
            st.subheader("2. Advanced ECG Analysis")
            
            # --- ADAPTIVE CONFIGURATION SECTION ---
            with st.expander("⚙️ Analysis Settings", expanded=True):
                ecg_col_options = [c for c in df.columns if any(x in c.lower() for x in ['ecg', 'i', 'ii', 'v1', 'v2'])]
                if not ecg_col_options: ecg_col_options = df.columns
                
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: target_ecg_col = st.selectbox("Select ECG Column", ecg_col_options)
                with c2: fs_ecg = st.number_input("ECG Sampling Rate (Hz)", value=500, min_value=1) # Default for PTB-XL is 500
                with c3:
                    # Explicit Algorithm Selection
                    algo_options = {
                        "Standard (Pan-Tompkins 1985)": "pantompkins1985",
                        "Adaptive Gradient (NeuroKit)": "neurokit",
                        "Wavelet (Elgendi 2010)": "elgendi2010",
                        "Slope (Hamilton 2002)": "hamilton2002"
                    }
                    algo_choice = st.selectbox("Peak Detection Algorithm", list(algo_options.keys()), index=1)
                    selected_method = algo_options[algo_choice]
            
            # --- ACTION BUTTONS ---
            st.write("")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_manual = st.button("Run Analysis (Manual)", type="secondary", use_container_width=True)
            with btn_col2:
                run_agentic = st.button("✨ Run Dr. Signal Analysis (Agentic)", type="primary", use_container_width=True)

            if run_manual or run_agentic:
                raw_data = df.iloc[start_row:end_row][target_ecg_col]
                analysis_data = pd.to_numeric(raw_data, errors='coerce').dropna()
                
                if len(analysis_data) < fs_ecg: 
                    st.error("Not enough valid data in current window.")
                else:
                    with st.spinner(f"Processing ECG using {selected_method}..."):
                        
                        with firebase_module.PerformanceMonitor() as pm:
                            # Pass user-selected method to processing function
                            peaks, heart_rate, sqis, ecg_cleaned = process_ecg(analysis_data, fs_ecg, method=selected_method)
                        
                        throughput = len(analysis_data) / pm.duration if pm.duration > 0 else 0
                        
                        firebase_module.log_computation_metrics(
                            doc_id=current_doc_id, 
                            file_name=uploaded_file.name,
                            analysis_type="ECG", 
                            duration_sec=pm.duration, 
                            memory_mb=pm.memory_used_mb, 
                            throughput_ksps=throughput/1000
                        )

                        st.markdown("#### ⚙️ System Telemetry")
                        perf_c1, perf_c2, perf_c3 = st.columns(3)
                        perf_c1.metric("Execution", f"{pm.duration*1000:.2f} ms")
                        perf_c2.metric("Throughput", f"{throughput/1000:.1f} kS/s")
                        perf_c3.metric("Peak RAM", f"{pm.memory_used_mb:.2f} MB")

                        if len(peaks) == 0: 
                            st.warning(f"No peaks detected using **{selected_method}**. \n\n**Troubleshooting:**\n1. Check if Sampling Rate ({fs_ecg} Hz) matches your device.\n2. Try switching algorithms (e.g., 'Adaptive Gradient').")
                        else:
                            if run_agentic:
                                if agent_active:
                                    with st.status("🤖 Dr. Signal is analyzing results...", expanded=True):
                                        st.write("Compiling metrics...")
                                        avg_sqis = {
                                            'skew_avg': np.mean(sqis['skewness']),
                                            'kurt_avg': np.mean(sqis['kurtosis']),
                                            'psd_avg': np.mean(sqis['psd_ratio'])
                                        }
                                        duration = len(analysis_data) / fs_ecg
                                        st.write("Consulting LLM...")
                                        report = agent.generate_ecg_report(avg_sqis, len(peaks), duration, fs_ecg)
                                        st.markdown("### 📋 Dr. Signal's Report")
                                        st.markdown(report)
                                else:
                                    st.error("Agent is offline. Please check API Key.")

                            tab1, tab2, tab3 = st.tabs(["Peak Detection", "Continuous Heart Rate", "SQI Metrics"])
                            with tab1:
                                fig_peaks = go.Figure()
                                fig_peaks.add_trace(go.Scatter(y=ecg_cleaned, name='Cleaned ECG', line=dict(color='gray')))
                                valid_peaks = peaks[peaks < len(ecg_cleaned)]
                                fig_peaks.add_trace(go.Scatter(x=valid_peaks, y=ecg_cleaned[valid_peaks], mode='markers', name='R-Peaks', marker=dict(color='red', size=8, symbol='x')))
                                st.plotly_chart(fig_peaks, use_container_width=True)
                            
                            with tab2:
                                fig_hr = go.Figure()
                                fig_hr.add_trace(go.Scatter(y=heart_rate, mode='lines', name='Heart Rate (BPM)', line=dict(color='red', width=2)))
                                fig_hr.update_layout(title="Continuous Heart Rate (BPM)", xaxis_title="Sample", yaxis_title="BPM")
                                st.plotly_chart(fig_hr, use_container_width=True)
                            
                            with tab3:
                                c_sq1, c_sq2, c_sq3 = st.columns(3)
                                c_sq1.metric("Skewness", f"{np.mean(sqis['skewness']):.2f}")
                                c_sq2.metric("Kurtosis", f"{np.mean(sqis['kurtosis']):.2f}")
                                c_sq3.metric("PSD Ratio", f"{np.mean(sqis['psd_ratio']):.2f}")

            st.markdown("---")
            st.subheader("3. Advanced EDA Analysis")
            with st.expander("Show EDA Analysis Tools", expanded=True):
                st.write("EDA Tools available in full version.") 
    else:
        st.info("Waiting for CSV file upload...")

elif app_mode == "Evaluation Experiment":
    st.header("🧪 Experiment: Visualization Latency Benchmark")
    
    with st.expander("ℹ️ Experiment Instructions", expanded=True):
        st.markdown("""
        This tool benchmarks the time it takes to:
        1. **Load Data**: Read signals from local files (WFDB format).
        2. **Render Plot**: Create the Plotly figure object.
        
        **Requirements**:
        - Local folder containing `.dat` and `.hea` files (e.g., MIT-BIH Arrhythmia).
        - Firebase connection for logging.
        """)

    # 1. Inputs - Select Source
    source_type = st.radio("Data Source", ["Local Path", "File Upload"], horizontal=True)
    
    dataset_path = None
    
    if source_type == "Local Path":
        dataset_path = st.text_input("Local Dataset Path (Absolute Path)", placeholder="/Users/username/data/mit-bih").strip()
    
    else: # File Upload
        uploaded_files = st.file_uploader("Upload .hea and .dat files (Folder Contents)", accept_multiple_files=True)
        if uploaded_files:
            # Save to temporary directory
            if 'eval_temp_dir' not in st.session_state:
                st.session_state.eval_temp_dir = tempfile.mkdtemp()
            
            # Save files
            for uf in uploaded_files:
                with open(os.path.join(st.session_state.eval_temp_dir, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            
            dataset_path = st.session_state.eval_temp_dir
            st.info(f"Files staged in temporary directory: {dataset_path}")

    if dataset_path and os.path.exists(dataset_path):
        # 2. Scanning & Filtering
        if 'eval_files' not in st.session_state: st.session_state.eval_files = []
        
        if st.button("scan Directory"):
            with st.spinner("Scanning header files..."):
                found_files = []
                try:
                    for f in os.listdir(dataset_path):
                        if f.endswith('.hea'):
                            record_name = f.replace('.hea', '')
                            # Light read to get info
                            try:
                                full_record_path = os.path.join(dataset_path, record_name)
                                header = wfdb.rdheader(full_record_path)
                                
                                duration_min = (header.sig_len / header.fs) / 60
                                found_files.append({
                                    'file': record_name,
                                    'path': full_record_path,
                                    'duration_min': duration_min,
                                    'fs': header.fs,
                                    'samples': header.sig_len,
                                    'n_sig': header.n_sig # Capture number of signals
                                })
                            except Exception as e:
                                pass # Skip bad files
                    st.session_state.eval_files = found_files
                    st.success(f"Found {len(found_files)} records.")
                except Exception as e:
                    st.error(f"Error scanning directory: {e}")

        # Filter UI
        if st.session_state.eval_files:
            all_durations = [f['duration_min'] for f in st.session_state.eval_files]
            if all_durations:
                min_glob, max_glob = min(all_durations), max(all_durations)
                
                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    if min_glob < max_glob:
                        filter_min, filter_max = st.slider("Filter by Duration (Minutes)", 
                                                         min_value=float(min_glob), 
                                                         max_value=float(max_glob), 
                                                         value=(float(min_glob), float(max_glob)))
                    else:
                        st.info(f"All files have uniform duration: {min_glob:.2f} min")
                        filter_min, filter_max = float(min_glob), float(max_glob)
                
                filtered_files = [f for f in st.session_state.eval_files if filter_min <= f['duration_min'] <= filter_max]
                
                st.write(f"**Selected Files: {len(filtered_files)}** (out of {len(st.session_state.eval_files)})")
                with st.expander("View File List"):
                    st.dataframe(pd.DataFrame(filtered_files))

                st.markdown("---")
                st.subheader("⚙️ Benchmark Settings")
                
                # Dynamic Check for Max Channels
                # Use .get('n_sig', 1) to handle legacy session states
                avail_channels = [f.get('n_sig', 1) for f in filtered_files]
                max_ch_limit = min(avail_channels) if avail_channels else 1
                
                c_set1, c_set2, c_set3, c_set4 = st.columns(4)
                with c_set1: n_trials = st.number_input("Trials per File", min_value=1, value=5)
                with c_set2: target_channel = st.text_input("Start Channel Index (0=First)", value="0")
                with c_set3: num_channels_view = st.number_input("Channels to Render", min_value=1, max_value=max_ch_limit, value=1)
                with c_set4: max_points = st.number_input("Max Points (0=All)", min_value=0, value=0, help="Limit number of data points per file to test rendering scaling.")

                start_eval = st.button("🚀 Start Benchmark", type="primary")
                
                if start_eval and filtered_files:
                    # Create Session Log
                    session_id = firebase_module.create_analysis_session(f"EVAL_BATCH_{dataset_path.split('/')[-1]}", session_type="evaluation_experiment")
                    st.toast(f"Session ID: {session_id}", icon="🧪")
                    
                    prog_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_ops = len(filtered_files) * n_trials
                    curr_op = 0
                    
                    results_log = []

                    for idx, f_info in enumerate(filtered_files):
                        record_path = f_info['path']
                        rec_name = f_info['file']
                        
                        for t in range(n_trials):
                            try:
                                status_text.write(f"Testing {rec_name} (Trial {t+1}/{n_trials})...")
                                
                                # 1. Measure Data Load
                                t0 = time.perf_counter()
                                record, fields = wfdb.rdsamp(record_path)
                                t_load = time.perf_counter() - t0
                                
                                # 2. Measure Plot Generation (Backend Only)
                                t1 = time.perf_counter()
                                fig = go.Figure()
                                
                                # Use target char or default 0
                                start_idx = int(target_channel) if target_channel.isdigit() else 0
                                
                                # Render N channels
                                for ch_offset in range(num_channels_view):
                                    # Handle wrap around if user selected index + count > total channels
                                    current_ch_idx = (start_idx + ch_offset) % record.shape[1]
                                    
                                    # Slice data if limit is set
                                    if max_points > 0 and max_points < len(record):
                                        y_data = record[:max_points, current_ch_idx]
                                    else:
                                        y_data = record[:, current_ch_idx]

                                    fig.add_trace(go.Scatter(y=y_data, mode='lines', name=f"Ch_{current_ch_idx}"))
                                
                                t_plot = time.perf_counter() - t1
                                
                                # Log to Firebase
                                firebase_module.log_computation_metrics(
                                    doc_id=session_id,
                                    file_name=rec_name,
                                    analysis_type="eval_benchmark",
                                    duration_sec=t_load + t_plot,
                                    memory_mb=0, # Not tracking RAM deeply here for speed
                                    throughput_ksps=((len(y_data) * num_channels_view) / (t_load + t_plot))/1000
                                )
                                
                                # Also log specific breakdown
                                firebase_module.log_plot_performance(
                                    doc_id=session_id,
                                    file_name=rec_name,
                                    exec_time_ms=t_plot * 1000,
                                    total_points=len(y_data) * num_channels_view,
                                    active_traces=num_channels_view
                                )
                                
                                curr_op += 1
                                prog_bar.progress(curr_op / total_ops)
                                
                            except Exception as e:
                                st.error(f"Failed on {rec_name}: {e}")
                    
                    st.success("✅ Evaluation Complete!")
                    st.balloons()
                    
                    # --- RESULTS DOWNLOAD ---
                    with st.spinner("Preparing Results CSV..."):
                        df_results = firebase_module.fetch_benchmark_results(session_id)
                        if df_results is not None and not df_results.empty:
                            # Convert to CSV
                            csv = df_results.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                label="📥 Download Benchmark Results (CSV)",
                                data=csv,
                                file_name=f"benchmark_results_{session_id}.csv",
                                mime="text/csv",
                                type="primary"
                            )
                        else:
                            st.warning("No results found to download.")
    
    elif dataset_path:
        st.error(f"Path does not exist on server: '{dataset_path}'")