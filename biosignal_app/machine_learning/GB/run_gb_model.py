from ml_gb import gb_leave_one_participant_out
import tkinter as tk
from tkinter import filedialog
import pandas as pd


def select_feature_dataframe():

    root=tk.Tk()
    root.withdraw()

    file_path=filedialog.askopenfilename(filetypes=[("CSV","*.csv")])

    if not file_path:
        raise Exception("No file selected.")

    return pd.read_csv(file_path)


df_raw=select_feature_dataframe()

if "Unnamed: 0" in df_raw.columns:
    df_raw=df_raw.drop(columns=["Unnamed: 0"])

df_raw["type"]=df_raw["BehaviorCode"]
df_raw=df_raw[df_raw["type"].isin([0,1])]

gb_leave_one_participant_out(
    df_raw,
    undersample_method="Rus",
    save_dir="GB_LOPO_RESULTS"
)