import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

FILES = {
    "kr":   "KR.xlsx",
    "mlpt": "MLPT.xlsx",
    "gt":   "GT.xlsx",
    "svb":  "SVB.xlsx",
    "eve-en": "eveintant.xlsx",
}

INTRON_GENES = {"svb", "kr", "mlpt", "eve-en"}

COLORS = {
    "kr":    "red",
    "mlpt":  "green",
    "gt":    "gold",
    "svb":   "brown",
    "cad":   "violet",
    "eve-en":   "purple",
    # Intron colors (lightened versions)
    "kr(int)": None,  # Will be computed
    "mlpt(int)": None,
    "svb(int)": None,
    "eve-en(int)": None,
}

def lighten_color(color, amount=0.5):
    c = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple((1 - amount) * c + amount * white)

# Initialize intron colors
COLORS["kr(int)"] = lighten_color(COLORS["kr"], 0.5)
COLORS["mlpt(int)"] = lighten_color(COLORS["mlpt"], 0.5)
COLORS["svb(int)"] = lighten_color(COLORS["svb"], 0.5)
COLORS["eve-en(int)"] = lighten_color(COLORS["eve-en"], 0.5)

def scale_to_unit(data_dict):
    stage_means = []
    for v in data_dict.values():
        vals = np.array(v, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            stage_means.append(vals.mean())
    if len(stage_means) == 0:
        return data_dict
    mn, mx = np.min(stage_means), np.max(stage_means)
    if mx - mn == 0:
        return data_dict
    scaled = {}
    for k, v in data_dict.items():
        vals = np.array(v, dtype=float)
        scaled[k] = (vals - mn) / (mx - mn)
    return scaled

def compute_mean_se(data_dict, x_stages):
    xs, ys, ses = [], [], []
    for i, s in enumerate(x_stages):
        vals = np.array(data_dict.get(s, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        xs.append(i)
        ys.append(vals.mean())
        ses.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(ses)

def smooth_plot(ax, xs, ys, ses, color, label=None, show_variance=True):
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, ses = xs[mask], ys[mask], ses[mask]
    if len(xs) < 3:
        return
    sx = np.linspace(xs.min(), xs.max(), 400)
    spline = PchipInterpolator(xs, ys)
    sy = spline(sx)
    ax.plot(sx, sy, color=color, lw=2, label=label)
    if show_variance and len(ses) > 0:
        se_spline = PchipInterpolator(xs, ses)
        se_vals = se_spline(sx)
        ax.fill_between(sx, sy - se_vals, sy + se_vals, color=color, alpha=0.2)


def read_one(filepath):
    df = pd.read_excel(filepath, sheet_name="BASE")
    df = df.replace(0.0, np.nan)
    df["Stage"] = df["Stage"].astype(str).str.strip()
    stages = df["Stage"].tolist()

    eve_cols = [c for c in df.columns if c.lower().startswith("eve")]
    mrna_cols = [c for c in df.columns if c.lower().startswith("mrna")]
    int_cols = [c for c in df.columns if c.lower().startswith("int")]

    full_stages = [s for s in stages]
    eve_d = {s: [] for s in full_stages}
    mrna_d = {s: [] for s in full_stages}
    int_d = {s: [] for s in full_stages}

    for s in stages:
        row = df[df["Stage"] == s]
        if not row.empty:
            eve_vals = row[eve_cols].values.flatten()
            mrna_vals = row[mrna_cols].values.flatten()
            int_vals = row[int_cols].values.flatten()

            eve_d[s] = [float(v) for v in eve_vals if pd.notna(v)]
            mrna_d[s] = [float(v) for v in mrna_vals if pd.notna(v)]
            int_d[s] = [float(v) for v in int_vals if pd.notna(v)]

    return full_stages, eve_d, mrna_d, int_d

def load_all_data(data_path="data/Temporal"):
    global_stage_order = []
    per_gene = {}
    EXCLUDE_FROM_EVE_POOL = {"eve-en"}
    for gene, fname in FILES.items():
        stages, eve_d, mrna_d, int_d = read_one(f"{data_path}/{fname}")
        per_gene[gene] = {"eve": eve_d, "mrna": mrna_d, "int": int_d if gene in INTRON_GENES else None}
        for s in stages:
            if s not in global_stage_order:
                global_stage_order.append(s)
    pooled_eve = {s: [] for s in global_stage_order}
    for gene, gdict in per_gene.items():
        if gene in EXCLUDE_FROM_EVE_POOL:
            continue
        for s in global_stage_order:
            pooled_eve[s].extend(gdict["eve"].get(s, []))
    pooled_eve_scaled = scale_to_unit(pooled_eve)
    scaled_genes = {}
    for gene in per_gene:
        scaled_genes[gene] = scale_to_unit(per_gene[gene]["mrna"])
    scaled_introns = {}
    for gene, gdict in per_gene.items():
        if gdict["int"] is not None:
            scaled_introns[gene] = scale_to_unit(gdict["int"])
    return (global_stage_order, pooled_eve_scaled, scaled_genes, scaled_introns)



