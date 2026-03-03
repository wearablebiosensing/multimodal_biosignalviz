from ml_svm import svm_leave_one_participant_out_weighted
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd


# ==========================================================
# 1. SELECT FEATURE DATAFRAME
# ==========================================================
def select_feature_dataframe():

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Select PSD Feature Table",
        "Please select:\n\n"
        "psd_features_two_second.csv\n\n"
        "The file MUST contain:\n"
        "- participantId\n"
        "- BehaviorCode (binary 0/1)\n"
        "- activityID\n"
        "- WindowNumber\n"
        "- PSD feature columns"
    )

    file_path = filedialog.askopenfilename(
        title="Select Feature CSV",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        raise Exception("No file selected.")

    df = pd.read_csv(file_path)

    print("\nLoaded file:", file_path)
    print("Shape:", df.shape)

    required_cols = ["participantId", "BehaviorCode", "WindowNumber"]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    return df


# ==========================================================
# 2. LOAD DATA
# ==========================================================
df_raw = select_feature_dataframe()

if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

df_raw["type"] = df_raw["BehaviorCode"]
df_raw = df_raw[df_raw["type"].isin([0, 1])]

print("\nFinal Shape After Binary Filter:", df_raw.shape)
print("Class Distribution:")
print(df_raw["type"].value_counts())


# ==========================================================
# 3. RUN LOPO SVM
# ==========================================================
results_df, summary = svm_leave_one_participant_out_weighted(
    df_raw,
    undersample_method="Rus",   # "Clus", "Rus", or None
    save_dir="SVM_LOPO_RESULTS"
)