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
