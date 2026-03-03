import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import os
import datetime

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
KEY_PATH = 'firebase_key.json'
OUTPUT_DIR = 'exported_data'

# -----------------------------------------------------------------------------
# Firebase Initialization
# -----------------------------------------------------------------------------
def init_firebase():
    if not os.path.exists(KEY_PATH):
        print(f"Error: {KEY_PATH} not found. Please place it in the same directory.")
        return None
    
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Firebase connection failed: {e}")
        return None

# -----------------------------------------------------------------------------
# Export Logic
# -----------------------------------------------------------------------------
def export_firestore_data():
    db = init_firebase()
    if not db: return

    print("--- Starting Data Export from Firestore ---")
    
    # Storage lists
    all_sessions = []
    all_file_stats = []
    all_plot_perf = []
    all_comp_perf = []

    # 1. Get all Root Documents (Sessions)
    sessions_ref = db.collection('analysis_logs')
    docs = sessions_ref.stream()

    for doc in docs:
        session_id = doc.id
        session_data = doc.to_dict()
        session_data['session_id'] = session_id
        all_sessions.append(session_data)
        
        print(f"Processing Session: {session_id}")

        # 2. Get 'visualization_metrics' (File Stats) Sub-collection
        vis_ref = sessions_ref.document(session_id).collection('visualization_metrics')
        for v_doc in vis_ref.stream():
            v_data = v_doc.to_dict()
            v_data['session_id'] = session_id # Link back to parent
            v_data['parent_file'] = session_data.get('file_name', 'Unknown')
            all_file_stats.append(v_data)

        # 3. Get 'rendering_perf' (Plotting) Sub-collection
        plot_ref = sessions_ref.document(session_id).collection('rendering_perf')
        for p_doc in plot_ref.stream():
            p_data = p_doc.to_dict()
            p_data['session_id'] = session_id
            all_plot_perf.append(p_data)

        # 4. Get 'computation_perf' (Analysis) Sub-collection
        comp_ref = sessions_ref.document(session_id).collection('computation_perf')
        for c_doc in comp_ref.stream():
            c_data = c_doc.to_dict()
            c_data['session_id'] = session_id
            all_comp_perf.append(c_data)

    # -------------------------------------------------------------------------
    # Save to CSV
    # -------------------------------------------------------------------------
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Helper to save
    def save_csv(data_list, filename):
        if data_list:
            df = pd.DataFrame(data_list)
            # Clean timestamps for CSV (remove timezone objects if problematic)
            for col in df.columns:
                if 'timestamp' in col or 'date' in col or 'at' in col:
                    df[col] = df[col].astype(str)
            
            path = f"{OUTPUT_DIR}/{filename}_{timestamp}.csv"
            df.to_csv(path, index=False)
            print(f"✅ Saved {len(df)} rows to {path}")
        else:
            print(f"⚠️ No data found for {filename}")

    save_csv(all_sessions, "sessions_metadata")
    save_csv(all_file_stats, "file_statistics")
    save_csv(all_plot_perf, "plot_performance")
    save_csv(all_comp_perf, "computation_metrics")

    print("\n--- Export Complete! Check 'exported_data' folder ---")

if __name__ == "__main__":
    export_firestore_data()