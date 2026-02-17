import streamlit as st
import matplotlib.pyplot as plt
from utils import temporal_utils, stage_utils, viz_utils
import io
import numpy as np
import os
import glob


@st.cache_data
def load_temporal_data():
    return temporal_utils.load_all_data("data/Temporal")


@st.cache_data
def load_stage_data(stage_folder):
    return stage_utils.read_stage_folder(stage_folder)


st.set_page_config(layout="wide", page_title="Gap Gene Visualizer")

# --- Page header ---
st.title("Interactive Gap Gene Expression Viewer")
st.markdown(
    "Select genes and visualization options to explore normalized expression profiles. "
    "'cad' and 'eve-en' options not available for temporal data source.",
    unsafe_allow_html=True)

# --- Stage folders ---
stage_folders = {
    "Stage 4.2": "data/Stage4_2",
    "Stage 5.2": "data/Stage5_2",
    "Stage 5.3": "data/Stage5_3",
    "Stage 6.2": "data/Stage6_2",
    "Stage 7.3": "data/Stage7_3"
}

# --- Sidebar: Select Stage ---
st.sidebar.title("Select Stage")
col1, col2 = st.sidebar.columns(2)
stages_and_temporal = ["Temporal"] + list(stage_folders.keys())
stage_selection = []

for idx, stage in enumerate(stages_and_temporal):
    col = col1 if idx % 2 == 0 else col2
    with col:
        if st.checkbox(stage, value=False, key=f"chk_{stage}"):
            stage_selection.append(stage)

data_source_options = []
for stage in stage_selection:
    if stage == "Temporal":
        data_source_options.append("Temporal")
    else:
        embryos = [os.path.basename(f).replace(".csv", "")
                   for f in glob.glob(os.path.join(stage_folders[stage], "EMB_*.csv"))]
        data_source_options.extend([f"{stage} ({embryo})" for embryo in embryos])

selected_data_sources = st.sidebar.multiselect(
    "Select sources to visualize",
    options=data_source_options,
    default=data_source_options
)

# --- Parse selection ---
selected_stage_embryos = {}
for item in selected_data_sources:
    if item == "Temporal":
        selected_stage_embryos["Temporal"] = []
    else:
        stage_name, embryo_name = item.split(" (")
        embryo_name = embryo_name.rstrip(")")
        if stage_name not in selected_stage_embryos:
            selected_stage_embryos[stage_name] = []
        selected_stage_embryos[stage_name].append(embryo_name)

# --- Sidebar: Select Genes ---
st.sidebar.title("Select Genes")
stage_only_genes = ["cad", "eve-en"]
all_genes = list(set(list(temporal_utils.COLORS.keys()) + stage_only_genes))
selected_genes = []
all_genes_sorted = sorted(all_genes)

col1, col2 = st.sidebar.columns(2)
for idx, gene in enumerate(all_genes_sorted):
    if idx % 2 == 0:
        with col1:
            if st.checkbox(gene, value=False, key=f"gene_{gene}_{idx}"):
                selected_genes.append(gene)
    else:
        with col2:
            if st.checkbox(gene, value=False, key=f"gene_{gene}_{idx}"):
                selected_genes.append(gene)

# --- Sidebar: Display Type ---
st.sidebar.title("Display Type")
dt_col1, dt_col2 = st.sidebar.columns(2)
with dt_col1:
    show_intensities = st.checkbox("Intensities", value=True, key="show_intensities")
with dt_col2:
    show_visualization = st.checkbox("Visualization", value=False, key="show_visualization")

# --- Sidebar: Temporal Display Options ---
st.sidebar.title("Temporal Display Options")
show_datapoints = st.sidebar.checkbox("Show data points", value=True, key="show_datapoints")
show_variance   = st.sidebar.checkbox("Show Variance",    value=True, key="show_variance")

# --- Sidebar: Stage Display Options ---
st.sidebar.title("Stage Display Options")
smooth_stage = st.sidebar.checkbox("Smooth stage profiles", value=True, key="smooth_stage")
smooth_frac  = st.sidebar.number_input(
    "Smoothing fraction", min_value=0.05, max_value=0.5,
    value=0.1, step=0.05, disabled=not smooth_stage)
smooth_order = st.sidebar.number_input(
    "Polynomial order", min_value=2, max_value=5,
    value=3, step=1, disabled=not smooth_stage)

# --- Load Temporal Data ---
temporal_stages, temporal_eve, temporal_mrna, temporal_introns = load_temporal_data()

# Will be populated during visualization block
visualized_images = []

# ==============================================================================
# INTENSITIES PLOT
# ==============================================================================
if show_intensities:
    fig, ax = plt.subplots(figsize=(18, 6))

    # Temporal traces
    if "Temporal" in selected_stage_embryos:
        temp_x_pos = np.linspace(0, 1, len(temporal_stages))

        for gene in selected_genes:
            if gene.endswith("(int)"):
                base_gene = gene.replace("(int)", "")
                if base_gene in temporal_introns:
                    ys_vals = [np.array(temporal_introns[base_gene].get(s, [np.nan])) for s in temporal_stages]
                    ys  = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                    ses = np.array([v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])
                    if show_datapoints:
                        for xi, vals in zip(temp_x_pos, ys_vals):
                            vals = [v for v in vals if np.isfinite(v)]
                            if vals:
                                ax.scatter([xi]*len(vals), vals,
                                           color=temporal_utils.COLORS.get(gene, "gray"), alpha=0.5, s=20)
                    temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                               temporal_utils.COLORS.get(gene, "gray"),
                                               f"{gene} (Temporal)", show_variance)

            elif gene == "eve-en":
                ys_vals = [np.array(temporal_eve.get(s, [np.nan])) for s in temporal_stages]
                ys  = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                ses = np.array([v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])
                if show_datapoints:
                    for xi, vals in zip(temp_x_pos, ys_vals):
                        vals = [v for v in vals if np.isfinite(v)]
                        if vals:
                            ax.scatter([xi]*len(vals), vals,
                                       color=temporal_utils.COLORS.get("eve-en", "purple"), alpha=0.5, s=20)
                temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                           temporal_utils.COLORS.get("eve-en", "purple"),
                                           "eve-en (Temporal)", show_variance)

            elif gene in temporal_mrna:
                ys_vals = [np.array(temporal_mrna[gene].get(s, [np.nan])) for s in temporal_stages]
                ys  = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                ses = np.array([v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])
                if show_datapoints:
                    for xi, vals in zip(temp_x_pos, ys_vals):
                        vals = [v for v in vals if np.isfinite(v)]
                        if vals:
                            ax.scatter([xi]*len(vals), vals,
                                       color=temporal_utils.COLORS.get(gene, "black"), alpha=0.5, s=20)
                temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                           temporal_utils.COLORS.get(gene, "black"),
                                           f"{gene} (Temporal)", show_variance)

    # Stage traces
    for stage in selected_stage_embryos:
        if stage == "Temporal":
            continue
        stage_data = load_stage_data(stage_folders[stage])
        for embryo_name in selected_stage_embryos[stage]:
            embryo_key  = embryo_name + ".csv"
            embryo_data = stage_data.get(embryo_key, {})
            for gene in selected_genes:
                gdata = embryo_data.get(gene)
                if gdata is not None:
                    x = gdata["x"]
                    y = gdata["y"]
                    if smooth_stage:
                        y = stage_utils.smooth_signal(y, frac=smooth_frac, polyorder=int(smooth_order))
                    suffix = " (smoothed)" if smooth_stage else ""
                    ax.plot(x, y, lw=2,
                            color=temporal_utils.COLORS.get(gene, "black"),
                            label=f"{gene} ({stage} {embryo_name}){suffix}")

    ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel("Scaled intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if any(s.startswith("Stage") for s in selected_stage_embryos):
        ax.set_xlabel("Spatial Position")
    if "Temporal" in selected_stage_embryos:
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(np.linspace(0, 1, len(temporal_stages)))
        ax_top.set_xticklabels(temporal_stages, rotation=45)
        ax_top.set_xlabel("Temporal Stage")

    st.pyplot(fig)

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    st.download_button("Download PNG", buf_png, file_name="gap_gene_plot.png", mime="image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
    buf_pdf.seek(0)
    st.download_button("Download PDF", buf_pdf, file_name="gap_gene_plot.pdf", mime="application/pdf")


# ==============================================================================
# VISUALIZATION
# ==============================================================================
if show_visualization:
    st.header("Image Visualization")

    for stage in selected_stage_embryos:
        if stage == "Temporal":
            continue

        stage_folder = stage_folders[stage]

        for embryo_name in selected_stage_embryos[stage]:
            tiff_path = os.path.join(stage_folder, f"{embryo_name}.tif")
            if not os.path.exists(tiff_path):
                st.warning(f"TIFF file not found: {tiff_path}")
                continue

            st.subheader(f"{stage} — {embryo_name}")
            image = viz_utils.load_tiff(tiff_path)

            cols_per_row = 3
            gene_count   = 0

            for gene in selected_genes:
                stack_num = viz_utils.get_stack_number_for_gene(gene)
                if stack_num is None:
                    continue

                if gene_count % cols_per_row == 0:
                    cols = st.columns(cols_per_row)

                stack = viz_utils.get_stack(image, stack_num)

                # Index of this image in the list (before appending)
                current_idx = len(visualized_images)
                visualized_images.append({
                    "stage":     stage,
                    "embryo":    embryo_name,
                    "gene":      gene,
                    "stack_num": stack_num,
                    "tiff_path": tiff_path
                })

                # Is this the image selected for B&C adjustment?
                selected_idx = st.session_state.get('selected_image_idx', None)
                is_selected  = (selected_idx == current_idx)

                # Apply adjustment
                raw_min = float(stack.min())
                raw_max = float(stack.max())
                if is_selected and 'bc_brightness' in st.session_state and 'bc_contrast' in st.session_state:
                    adjusted = viz_utils.adjust_brightness_contrast(
                        stack,
                        brightness=st.session_state['bc_brightness'],
                        contrast=st.session_state['bc_contrast'],
                        raw_min=raw_min,
                        raw_max=raw_max,
                    )
                else:
                    adjusted = viz_utils.autoadjust_brightness_contrast(stack)

                colored_stack = viz_utils.apply_stack_color(adjusted, stack_num)

                with cols[gene_count % cols_per_row]:
                    fig_img, ax_img = plt.subplots(figsize=(5, 5))
                    ax_img.imshow(colored_stack)
                    ax_img.axis("off")
                    title_suffix = " ✦" if is_selected else ""
                    ax_img.set_title(f"{gene}{title_suffix}", fontsize=10, fontweight='bold')
                    st.pyplot(fig_img)
                    plt.close(fig_img)

                gene_count += 1


# ==============================================================================
# VISUALIZATION OPTIONS SIDEBAR
# ==============================================================================
if show_visualization and len(visualized_images) > 0:
    st.sidebar.title("Visualization Options")

    # Image selector
    image_labels = [
        f"{img['gene']} ({img['stage']} {img['embryo']})"
        for img in visualized_images
    ]

    if 'selected_image_idx' not in st.session_state:
        st.session_state['selected_image_idx'] = len(image_labels) - 1

    current_sel = min(st.session_state['selected_image_idx'], len(image_labels) - 1)

    new_sel = st.sidebar.selectbox(
        "Select image to adjust",
        range(len(image_labels)),
        format_func=lambda i: image_labels[i],
        index=current_sel,
        key='image_selector'
    )

    # Clear B/C state whenever user switches to a different image
    if new_sel != st.session_state.get('selected_image_idx'):
        for k in ('bc_brightness', 'bc_contrast', 'auto_step'):
            st.session_state.pop(k, None)
    st.session_state['selected_image_idx'] = new_sel

    # Load selected stack to derive its pixel range
    sel_info  = visualized_images[new_sel]
    _img      = viz_utils.load_tiff(sel_info["tiff_path"])
    _stack    = viz_utils.get_stack(_img, sel_info["stack_num"])
    s_min     = float(_stack.min())
    s_max     = float(_stack.max())

    # ── Seed B/C defaults (B=C=midpoint → full-range window) ─────────────────
    # Normalised B=0.5, C=0.5 maps to lo=0.25, hi=0.75 ... but we actually
    # want the initial display to be the full range.
    # Full range corresponds to C_n = 0  (window width = 1), B_n = anything.
    # So we seed contrast = raw_min (i.e. C_n = 0) and brightness = midpoint.
    if 'bc_brightness' not in st.session_state:
        st.session_state['bc_brightness'] = (s_min + s_max) / 2.0
    if 'bc_contrast' not in st.session_state:
        st.session_state['bc_contrast'] = s_min   # C_n = 0 → full-range window
    if 'auto_step' not in st.session_state:
        st.session_state['auto_step'] = 1

    st.sidebar.subheader("Brightness / Contrast")

    # ── Derive current min/max from B/C state ─────────────────────────────────
    _cur_lo, _cur_hi = viz_utils.bc_to_minmax(
        st.session_state['bc_brightness'],
        st.session_state['bc_contrast'],
        s_min, s_max
    )
    # ── Minimum slider (writes back to B/C via inverse formula) ──────────────
    new_min = st.sidebar.slider(
        "Minimum",
        min_value=s_min, max_value=s_max,
        value=float(np.clip(_cur_lo, s_min, s_max)),
        key="slider_min"
    )
    if abs(new_min - _cur_lo) > 1e-6:
        # Clamp so min never exceeds current max
        new_min = min(new_min, _cur_hi)
        new_B, new_C = viz_utils.minmax_to_bc(new_min, _cur_hi, s_min, s_max)
        st.session_state['bc_brightness'] = new_B
        st.session_state['bc_contrast'] = new_C
        _cur_lo = new_min

    # ── Maximum slider (writes back to B/C via inverse formula) ──────────────
    new_max = st.sidebar.slider(
        "Maximum",
        min_value=s_min, max_value=s_max,
        value=float(np.clip(_cur_hi, s_min, s_max)),
        key="slider_max"
    )
    if abs(new_max - _cur_hi) > 1e-6:
        # Clamp so max never goes below current min
        new_max = max(new_max, _cur_lo)
        new_B, new_C = viz_utils.minmax_to_bc(_cur_lo, new_max, s_min, s_max)
        st.session_state['bc_brightness'] = new_B
        st.session_state['bc_contrast'] = new_C
    # ── Brightness slider ─────────────────────────────────────────────────────
    new_brightness = st.sidebar.slider(
        "Brightness",
        min_value=s_min, max_value=s_max,
        value=float(np.clip(st.session_state['bc_brightness'], s_min, s_max)),
        key="slider_brightness"
    )
    if new_brightness != st.session_state['bc_brightness']:
        st.session_state['bc_brightness'] = new_brightness
        # Recompute derived window for the readout below
        _cur_lo, _cur_hi = viz_utils.bc_to_minmax(
            new_brightness, st.session_state['bc_contrast'], s_min, s_max)

    # ── Contrast slider ───────────────────────────────────────────────────────
    new_contrast = st.sidebar.slider(
        "Contrast",
        min_value=s_min, max_value=s_max,
        value=float(np.clip(st.session_state['bc_contrast'], s_min, s_max)),
        key="slider_contrast"
    )
    if new_contrast != st.session_state['bc_contrast']:
        st.session_state['bc_contrast'] = new_contrast
        _cur_lo, _cur_hi = viz_utils.bc_to_minmax(
            st.session_state['bc_brightness'], new_contrast, s_min, s_max)

    # ── Auto button ────────────────────────────────────────────────────────────
    if st.sidebar.button("Auto"):
        auto_B, auto_C = viz_utils.auto_bc(
            _stack,
            raw_min=s_min,
            raw_max=s_max,
            step=st.session_state['auto_step']
        )
        st.session_state['bc_brightness'] = auto_B
        st.session_state['bc_contrast']   = auto_C
        # Advance step (cycle every 8)
        st.session_state['auto_step'] = (st.session_state['auto_step'] % 8) + 1
        st.rerun()

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.sidebar.button("Reset"):
        st.session_state['bc_brightness'] = (s_min + s_max) / 2.0
        st.session_state['bc_contrast']   = s_min
        st.session_state['auto_step']     = 1
        st.rerun()