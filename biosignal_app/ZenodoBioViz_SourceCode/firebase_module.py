# Firebase Imports

import time
import os
import tracemalloc
import re
import uuid
import pandas as pd
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
# -----------------------------------------------------------------------------
# Performance Monitoring Utilities
# -----------------------------------------------------------------------------
class PerformanceMonitor:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.start_memory = 0
        self.peak_memory = 0
        self.duration = 0
        self.memory_used_mb = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        _, self.peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.duration = self.end_time - self.start_time
        self.memory_used_mb = (self.peak_memory - self.start_memory) / (1024 * 1024)

# -----------------------------------------------------------------------------
# Firebase Configuration & Logging
# -----------------------------------------------------------------------------
def init_firebase():
    """Initializes Firestore client with singleton pattern prevention."""
    try:
        if not firebase_admin._apps:
            if os.path.exists('firebase_key.json'):
                cred = credentials.Certificate('firebase_key.json')
                firebase_admin.initialize_app(cred)
                return firestore.client()
            else:
                return None
        return firestore.client()
    except Exception as e:
        print(f"Firebase Init Error: {e}")
        return None

def sanitize_document_id(file_name):
    """
    Creates a readable, safe document ID.
    Format: CleanName_Timestamp_ShortHash
    Example: patient_data_csv_202310271030_a1b2
    """
    # Remove non-alphanumeric chars (keep underscores/hyphens)
    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', file_name)
    # Remove extension if present (e.g., .csv)
    clean_name = clean_name.rsplit('_csv', 1)[0] if '_csv' in clean_name.lower() else clean_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:4]
    
    return f"{clean_name}_{timestamp}_{short_uuid}"

def create_analysis_session(file_name, session_type="standard_analysis"):
    """Creates a root session document with a readable ID."""
    db = init_firebase()
    if db is None: return None
    try:
        # Generate readable ID
        custom_doc_id = sanitize_document_id(file_name)
        
        doc_ref = db.collection('analysis_logs').document(custom_doc_id)
        doc_ref.set({
            'file_name': file_name,
            'session_start': firestore.SERVER_TIMESTAMP,
            'timestamp_iso': datetime.now().isoformat(), # CSV Friendly
            'status': 'active',
            'session_type': session_type,
            'user_agent_mode': 'streamlit_standard'
        })
        return custom_doc_id
    except Exception as e:
        print(f"Error creating session: {e}")
        return None

def log_visualization_metrics(doc_id, df, file_obj):
    """Logs static file statistics with flattened fields for CSV export."""
    db = init_firebase()
    if db is None or not doc_id: return
    try:
        stats_data = {
            'session_id': doc_id, # Link for flattened CSV
            'file_name': file_obj.name,
            'row_count': len(df),
            'file_size_bytes': file_obj.size,
            'file_type': file_obj.type if file_obj.type else "csv",
            'total_columns': len(df.columns),
            'memory_usage_bytes': int(df.memory_usage(deep=True).sum()),
            'timestamp_iso': datetime.now().isoformat(), # CSV Friendly
            'logged_at': firestore.SERVER_TIMESTAMP
        }
        db.collection('analysis_logs').document(doc_id).collection('file_stats').add(stats_data)
    except Exception as e:
        print(f"Error logging vis metrics: {e}")

def log_plot_performance(doc_id, file_name, exec_time_ms, total_points, active_traces):
    """
    Logs dynamic plot performance with denormalized fields (Context + Data).
    """
    db = init_firebase()
    if db is None or not doc_id: return
    try:
        perf_data = {
            'session_id': doc_id,  # DENORMALIZED: Allows grouping without joining in CSV
            'file_name': file_name, # DENORMALIZED: Human readable context in every row
            'plot_gen_time_ms': float(f"{exec_time_ms:.2f}"),
            'total_points_rendered': int(total_points),
            'active_trace_count': int(active_traces),
            'timestamp_iso': datetime.now().isoformat(), # CSV Friendly string
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        # Using 'plot_performance' subcollection for high-frequency updates
        db.collection('analysis_logs').document(doc_id).collection('plot_performance').add(perf_data)
    except Exception as e:
        print(f"Error logging plot perf: {e}")

def log_computation_metrics(doc_id, file_name, analysis_type, duration_sec, memory_mb, throughput_ksps=0):
    """Logs heavy computation analysis metrics with flattened structure."""
    db = init_firebase()
    if db is None or not doc_id: return
    try:
        perf_data = {
            'session_id': doc_id, # Link for flattened CSV
            'file_name': file_name,
            'analysis_type': analysis_type,
            'execution_time_ms': duration_sec * 1000,
            'peak_memory_mb': memory_mb,
            'throughput_ksps': throughput_ksps,
            'timestamp_iso': datetime.now().isoformat(), # CSV Friendly
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        db.collection('analysis_logs').document(doc_id).collection('computation_metrics').add(perf_data)
    except Exception as e:
        print(f"Error logging comp metrics: {e}")

def fetch_benchmark_results(session_id):
    """
    Fetches computation and plot metrics for a session and returns a flattened DataFrame.
    """
    db = init_firebase()
    if db is None or not session_id: return None
    
    try:
        # 1. Fetch Computation Metrics & Sort by Timestamp (to align trials)
        comp_ref = db.collection('analysis_logs').document(session_id).collection('computation_metrics')
        comp_docs = comp_ref.stream()
        comp_data = sorted([d.to_dict() for d in comp_docs], key=lambda x: x.get('timestamp', 0))
        
        # 2. Fetch Plot Performance & Sort by Timestamp
        plot_ref = db.collection('analysis_logs').document(session_id).collection('plot_performance')
        plot_docs = plot_ref.stream()
        plot_data = sorted([d.to_dict() for d in plot_docs], key=lambda x: x.get('timestamp', 0))
        
        if not comp_data:
            return None

        df_comp = pd.DataFrame(comp_data)
        
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            
            # Select only relevant columns to add (avoid duplicate metadata like file_name/session_id)
            cols_to_add = [c for c in ['plot_gen_time_ms', 'total_points_rendered', 'active_trace_count'] if c in df_plot.columns]
            
            if cols_to_add:
                df_plot_clean = df_plot[cols_to_add]
                # Merge horizontally based on sorted order (assuming 1:1 consistent logging)
                df_comp = pd.concat([df_comp.reset_index(drop=True), df_plot_clean.reset_index(drop=True)], axis=1)

        return df_comp
        
    except Exception as e:
        print(f"Error fetching results: {e}")
        return None
