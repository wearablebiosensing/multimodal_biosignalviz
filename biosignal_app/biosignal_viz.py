import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal as signal
import scipy.stats as stats
import neurokit2 as nk
import wfdb
import tempfile
import zipfile
import os
import shutil

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
    Loads data from CSV or Zipped WFDB (PTB-XL style) files.
    """
    try:
        # 1. Handle CSV
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
            return df
        
        # 2. Handle Zipped WFDB (common for PhysioNet downloads)
        elif file.name.endswith('.zip'):
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
            st.error("Unsupported file format. Please upload CSV or Zipped WFDB.")
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
    freqs, psd = signal.welch(segment, fs=fs, nperseg=len(segment))
    idx_5_15 = np.logical_and(freqs >= 5, freqs <= 15)
    power_5_15 = np.sum(psd[idx_5_15])
    idx_0_45 = np.logical_and(freqs >= 0, freqs <= 45)
    power_0_45 = np.sum(psd[idx_0_45])
    if power_0_45 == 0: return 0
    return power_5_15 / power_0_45

def process_ecg(data_series, fs, method="pantompkins1985"):
    """
    Processes ECG signal using NeuroKit2 with selectable method.
    """
    # Ensure data is numeric and handle NaNs
    clean_data = pd.to_numeric(data_series, errors='coerce').fillna(0).values
    
    # NeuroKit2 processing pipeline
    try:
        # 1. Clean the signal
        # Use 'neurokit' cleaning which is generally robust
        ecg_cleaned = nk.ecg_clean(clean_data, sampling_rate=fs, method="neurokit")
        
        # 2. Find Peaks using SELECTED algorithm
        # This makes the detection adaptive based on user choice
        peaks_dict, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method=method)
        peaks = peaks_dict['ECG_R_Peaks']
        
        # 3. Calculate Heart Rate
        heart_rate = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len(clean_data))
        
        # 4. SQI Calculation
        sqis = {
            'skewness': stats.skew(ecg_cleaned),
            'kurtosis': stats.kurtosis(ecg_cleaned),
            'psd_ratio': calculate_psd_ratio(ecg_cleaned, fs)
        }
        
        return peaks, heart_rate, sqis, ecg_cleaned

    except Exception as e:
        # If specific method fails, try fallback or just return empty
        st.warning(f"ECG Processing Warning ({method}): {e}")
        return [], [], {}, clean_data

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
    app_mode = st.radio("Select Mode", ["Analysis Dashboard", "CSV Concatenator (Prep)"])
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
        # Allow both CSV and ZIP uploads
        uploaded_file = st.file_uploader("Drag and drop CSV or Zipped WFDB file here", type=['csv', 'zip'])
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
            elif 'time' in all_cols: default_x_index = all_cols.index('time') # Added for WFDB

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