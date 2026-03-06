import pandas as pd
import numpy as np
import glob
from scipy.signal import savgol_filter
import os

STAGE_GENES = ["kr", "krint", "mlpt", "mlptint", "svb", "svbint", "gt", "eve-en", "cad"]

STAGE_GENE_ALIASES = {
    "svb(int)": "svbint",
    "kr(int)": "krint",
    "mlpt(int)": "mlptint",
}

def normalize(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    return (x - x.min()) / (x.max() - x.min())

def smooth_signal(y, frac=0.1, polyorder=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y
    window_length = int(max(polyorder + 2, (n * frac) // 2 * 2 + 1))
    window_length = min(window_length, n if n % 2 == 1 else n - 1)

    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def read_stage_folder(folder_path):
    stage_data = {}
    for fpath in glob.glob(os.path.join(folder_path, "EMB_*.csv")):
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        embryo_data = {}
        x = normalize(df["Distance_(microns)"])
        # Check all possible gene names (both direct and aliased)
        for gene in STAGE_GENES:
            if gene in df.columns:
                y = normalize(df[gene])
                embryo_data[gene] = {"x": x, "y": y}

        # Also check for aliased names (kr(int) -> krint, etc.)
        for ui_name, csv_col in STAGE_GENE_ALIASES.items():
            if csv_col in df.columns:
                y = normalize(df[csv_col])
                embryo_data[ui_name] = {"x": x, "y": y}

        stage_data[fname] = embryo_data
    return stage_data

def read_stage_folder_smoothed(folder_path, frac=0.1, polyorder=3):
    """Same as read_stage_folder but y values are pre-smoothed at fixed settings.
    Used exclusively by the Blocks plot so block shapes are stable."""
    stage_data = {}
    for fpath in glob.glob(os.path.join(folder_path, "EMB_*.csv")):
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        embryo_data = {}
        x = normalize(df["Distance_(microns)"])
        for gene in STAGE_GENES:
            if gene in df.columns:
                y = normalize(df[gene])
                y = smooth_signal(y, frac=frac, polyorder=polyorder)
                embryo_data[gene] = {"x": x, "y": y}

        for ui_name, csv_col in STAGE_GENE_ALIASES.items():
            if csv_col in df.columns:
                y = normalize(df[csv_col])
                y = smooth_signal(y, frac=frac, polyorder=polyorder)
                embryo_data[ui_name] = {"x": x, "y": y}

        stage_data[fname] = embryo_data
    return stage_data