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
    "Select genes and visualization options to explore normalized expression profiles. 'cad' and 'eve-en' options not available for temporal data source.",
    unsafe_allow_html=True)

# --- Stage folders ---
stage_folders = {
    "Stage 4.2": "data/Stage4_2",
    "Stage 5.2": "data/Stage5_2",
    "Stage 5.3": "data/Stage5_3",
    "Stage 6.2": "data/Stage6_2",
    "Stage 7.1": "data/Stage7_1"
}

# --- Sidebar ---
st.sidebar.title("Select Stage")
# Two-column layout for stage checkboxes
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

# --- Parse selection into dict for plotting ---
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

# --- Gene selection ---
st.sidebar.title("Select Genes")
stage_only_genes = ["cad", "eve-en"]  # These appear in stage data but not in temporal FILES

# Build all_genes from COLORS keys (which now includes both base genes and intron variants)
all_genes = list(set(list(temporal_utils.COLORS.keys()) + stage_only_genes))

selected_genes = []
all_genes_sorted = sorted(all_genes)  # Sort for consistent order

# Create 2 columns for gene selection
col1, col2 = st.sidebar.columns(2)

for idx, gene in enumerate(all_genes_sorted):
    # Alternate between columns
    if idx % 2 == 0:
        with col1:
            if st.checkbox(gene, value=False, key=f"gene_{gene}_{idx}"):
                selected_genes.append(gene)
    else:
        with col2:
            if st.checkbox(gene, value=False, key=f"gene_{gene}_{idx}"):
                selected_genes.append(gene)

# --- Display Type Selection ---
st.sidebar.title("Display Type")
show_intensities = st.sidebar.checkbox("Intensities", value=True, key="show_intensities")
show_visualization = st.sidebar.checkbox("Visualization", value=False, key="show_visualization")

# --- Display Options ---
st.sidebar.title("Temporal Display Options")
show_datapoints = st.sidebar.checkbox("Show data points", value=True, key="show_datapoints")
show_variance = st.sidebar.checkbox("Show Variance", value=True, key="show_variance")

# --- Stage Display Options ---
st.sidebar.title("Stage Display Options")

smooth_stage = st.sidebar.checkbox(
    "Smooth stage profiles",
    value=True,
    key="smooth_stage"
)

smooth_frac = st.sidebar.number_input(
    "Smoothing fraction",
    min_value=0.05,
    max_value=0.5,
    value=0.1,
    step=0.05,
    disabled=not smooth_stage
)

smooth_order = st.sidebar.number_input(
    "Polynomial order",
    min_value=2,
    max_value=5,
    value=3,
    step=1,
    disabled=not smooth_stage
)

# --- Load Temporal Data ---
temporal_stages, temporal_eve, temporal_mrna, temporal_introns = load_temporal_data()

# Track all visualized images for visualization options
visualized_images = []

# --- INTENSITIES PLOT ---
if show_intensities:
    # --- Create Figure ---
    fig, ax = plt.subplots(figsize=(18, 6))

    # --- Temporal plotting ---
    if "Temporal" in selected_stage_embryos:
        temp_x_pos = np.linspace(0, 1, len(temporal_stages))

        for gene in selected_genes:
            # Handle intron genes
            if gene.endswith("(int)"):
                base_gene = gene.replace("(int)", "")
                if base_gene in temporal_introns:
                    ys_vals = [np.array(temporal_introns[base_gene].get(s, [np.nan])) for s in temporal_stages]
                    ys = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                    ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])

                    if show_datapoints:
                        for xi, vals in zip(temp_x_pos, ys_vals):
                            vals = [v for v in vals if np.isfinite(v)]
                            if len(vals) > 0:
                                ax.scatter([xi] * len(vals), vals,
                                           color=temporal_utils.COLORS.get(gene, "gray"),
                                           alpha=0.5, s=20)

                    temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                               temporal_utils.COLORS.get(gene, "gray"),
                                               f"{gene} (Temporal)", show_variance)

            # Handle eve-en mRNA (uses pooled eve data)
            elif gene == "eve-en":
                ys_vals = [np.array(temporal_eve.get(s, [np.nan])) for s in temporal_stages]
                ys = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])

                if show_datapoints:
                    for xi, vals in zip(temp_x_pos, ys_vals):
                        vals = [v for v in vals if np.isfinite(v)]
                        if len(vals) > 0:
                            ax.scatter([xi] * len(vals), vals,
                                       color=temporal_utils.COLORS.get("eve-en", "purple"),
                                       alpha=0.5, s=20)

                temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                           temporal_utils.COLORS.get("eve-en", "purple"),
                                           "eve-en (Temporal)", show_variance)

            # Handle other mRNA genes (kr, mlpt, gt, svb)
            elif gene in temporal_mrna:
                ys_vals = [np.array(temporal_mrna[gene].get(s, [np.nan])) for s in temporal_stages]
                ys = np.array([v.mean() if len(v) > 0 else np.nan for v in ys_vals])
                ses = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in ys_vals])

                if show_datapoints:
                    for xi, vals in zip(temp_x_pos, ys_vals):
                        vals = [v for v in vals if np.isfinite(v)]
                        if len(vals) > 0:
                            ax.scatter([xi] * len(vals), vals,
                                       color=temporal_utils.COLORS.get(gene, "black"),
                                       alpha=0.5, s=20)

                temporal_utils.smooth_plot(ax, temp_x_pos, ys, ses,
                                           temporal_utils.COLORS.get(gene, "black"),
                                           f"{gene} (Temporal)", show_variance)

    # --- Stage plotting ---
    for stage in selected_stage_embryos:
        if stage == "Temporal":
            continue  # Temporal handled separately
        stage_data = load_stage_data(stage_folders[stage])
        for embryo_name in selected_stage_embryos[stage]:
            embryo_key = embryo_name + ".csv"
            embryo_data = stage_data.get(embryo_key, {})
            for gene in selected_genes:
                # Look up gene directly (works for both regular genes and intron genes via aliases)
                gdata = embryo_data.get(gene)
                if gdata is not None:
                    x = gdata["x"]
                    y = gdata["y"]
                    if smooth_stage:
                        y = stage_utils.smooth_signal(y, frac=smooth_frac, polyorder=int(smooth_order))
                    suffix = " (smoothed)" if smooth_stage else ""
                    ax.plot(
                        x,
                        y,
                        lw=2,
                        color=temporal_utils.COLORS.get(gene, "black"),  # consistent color
                        label=f"{gene} ({stage} {embryo_name}){suffix}"
                    )

    # --- Axes settings ---
    ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel("Scaled intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Bottom axis for Stage
    if any(s.startswith("Stage") for s in selected_stage_embryos):
        ax.set_xlabel("Spatial Position")

    # Top axis for Temporal
    if "Temporal" in selected_stage_embryos:
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(np.linspace(0, 1, len(temporal_stages)))
        ax_top.set_xticklabels(temporal_stages, rotation=45)
        ax_top.set_xlabel("Temporal Stage")

    # --- Render Plot ---
    st.pyplot(fig)

    # --- Download Buttons ---
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
    buf_png.seek(0)
    st.download_button("Download PNG", buf_png, file_name="gap_gene_plot.png", mime="image/png")

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
    buf_pdf.seek(0)
    st.download_button("Download PDF", buf_pdf, file_name="gap_gene_plot.pdf", mime="application/pdf")

# --- VISUALIZATION OPTIONS (setup before displaying images) ---
selected_img_info = None
use_auto = False
min_brightness = None
max_brightness = None
auto_iterations = 1

if show_visualization:
    # Initialize session state for brightness/contrast
    if 'use_auto' not in st.session_state:
        st.session_state['use_auto'] = False
    if 'auto_iters' not in st.session_state:
        st.session_state['auto_iters'] = 1

# --- VISUALIZATION ---
if show_visualization:
    st.header("Image Visualization")

    # Only visualize stage data (not temporal)
    for stage in selected_stage_embryos:
        if stage == "Temporal":
            continue

        stage_folder = stage_folders[stage]

        for embryo_name in selected_stage_embryos[stage]:
            # Check if TIFF file exists
            tiff_path = os.path.join(stage_folder, f"{embryo_name}.tif")
            if not os.path.exists(tiff_path):
                st.warning(f"TIFF file not found: {tiff_path}")
                continue

            st.subheader(f"{stage} - {embryo_name}")

            # Load TIFF
            image = viz_utils.load_tiff(tiff_path)

            # Display selected genes
            cols_per_row = 3
            gene_count = 0

            for gene in selected_genes:
                # Map gene to stack number
                stack_num = viz_utils.get_stack_number_for_gene(gene)

                if stack_num is None:
                    continue  # Gene not available in stage visualization

                # Create new row if needed
                if gene_count % cols_per_row == 0:
                    cols = st.columns(cols_per_row)

                col_idx = gene_count % cols_per_row

                with cols[col_idx]:
                    # Get the stack
                    stack = viz_utils.get_stack(image, stack_num)

                    # Track this image for brightness/contrast adjustment
                    img_info = {
                        "stage": stage,
                        "embryo": embryo_name,
                        "gene": gene,
                        "stack_num": stack_num,
                        "tiff_path": tiff_path
                    }
                    visualized_images.append(img_info)

                    # Check if this is the selected image for adjustment
                    is_selected = (len(visualized_images) > 0 and
                                   st.session_state.get('selected_image_idx', len(visualized_images) - 1) == len(
                                visualized_images) - 1)

                    # Apply brightness/contrast
                    if is_selected and st.session_state.get('use_auto', False):
                        # Use auto adjustment
                        adjusted = viz_utils.autoadjust_brightness_contrast(
                            stack,
                            iterations=st.session_state.get('auto_iters', 1)
                        )
                    elif is_selected and 'min_brightness' in st.session_state and 'max_brightness' in st.session_state:
                        # Use manual adjustment
                        adjusted = viz_utils.adjust_brightness_contrast(
                            stack,
                            min_val=st.session_state.get('min_brightness'),
                            max_val=st.session_state.get('max_brightness')
                        )
                    else:
                        # Default: auto-adjust with 1 iteration
                        adjusted = viz_utils.autoadjust_brightness_contrast(stack, iterations=1)

                    # Apply color
                    colored_stack = viz_utils.apply_stack_color(adjusted, stack_num)

                    # Display
                    fig_img, ax_img = plt.subplots(figsize=(5, 5))
                    ax_img.imshow(colored_stack)
                    ax_img.axis("off")
                    title_suffix = " *" if is_selected else ""
                    ax_img.set_title(f"{gene}{title_suffix}", fontsize=10, fontweight='bold')
                    st.pyplot(fig_img)
                    plt.close(fig_img)

                gene_count += 1

# --- VISUALIZATION OPTIONS SIDEBAR ---
if show_visualization and len(visualized_images) > 0:
    st.sidebar.title("Visualization Options")

    # Select which image to adjust
    image_labels = [f"{img['gene']} ({img['stage']} {img['embryo']})"
                    for img in visualized_images]

    # Initialize selected_image_idx if not present
    if 'selected_image_idx' not in st.session_state:
        st.session_state['selected_image_idx'] = len(image_labels) - 1

    selected_image_idx = st.sidebar.selectbox(
        "Select image to adjust",
        range(len(image_labels)),
        format_func=lambda i: image_labels[i],
        index=st.session_state.get('selected_image_idx', len(image_labels) - 1),
        key='image_selector'
    )

    # Update session state
    st.session_state['selected_image_idx'] = selected_image_idx

    selected_img_info = visualized_images[selected_image_idx]

    # Brightness/Contrast controls
    st.sidebar.subheader("Brightness/Contrast")

    # Load the image
    image = viz_utils.load_tiff(selected_img_info["tiff_path"])
    stack = viz_utils.get_stack(image, selected_img_info["stack_num"])

    # Get min/max of original stack
    stack_min = float(stack.min())
    stack_max = float(stack.max())

    # Auto button
    auto_iterations = st.sidebar.number_input(
        "Auto adjust iterations",
        min_value=1,
        max_value=20,
        value=st.session_state.get('auto_iters', 1),
        step=1,
        key="auto_iterations"
    )

    if st.sidebar.button("Auto Brightness/Contrast"):
        st.session_state['min_brightness'] = None
        st.session_state['max_brightness'] = None
        st.session_state['use_auto'] = True
        st.session_state['auto_iters'] = auto_iterations
        st.rerun()

    # Manual controls
    use_auto = st.session_state.get('use_auto', False)

    if not use_auto:
        min_brightness = st.sidebar.slider(
            "Minimum",
            min_value=stack_min,
            max_value=stack_max,
            value=st.session_state.get('min_brightness', stack_min),
            key="min_brightness_slider"
        )

        max_brightness = st.sidebar.slider(
            "Maximum",
            min_value=stack_min,
            max_value=stack_max,
            value=st.session_state.get('max_brightness', stack_max),
            key="max_brightness_slider"
        )

        # Update session state
        st.session_state['min_brightness'] = min_brightness
        st.session_state['max_brightness'] = max_brightness
        st.session_state['use_auto'] = False

    # Reset button
    if st.sidebar.button("Reset"):
        st.session_state['min_brightness'] = stack_min
        st.session_state['max_brightness'] = stack_max
        st.session_state['use_auto'] = False
        st.rerun()