import numpy as np
import tifffile
from matplotlib.colors import to_rgb


# ---------------------------------------------------------------------------
# Gene / stack mapping
# ---------------------------------------------------------------------------

SLICE_NAMES = [
    "svbint",   # 1
    "krint",    # 2
    "svb",      # 3
    "mlptint",  # 4
    "eve-en",   # 5
    "cad",      # 6
    "kr",       # 7
    "mlpt",     # 8
    "gt",       # 9
]

GENE_TO_STACK = {
    "svbint":   1, "svb(int)":  1,
    "krint":    2, "kr(int)":   2,
    "svb":      3,
    "mlptint":  4, "mlpt(int)": 4,
    "eve-en":   5,
    "cad":      6,
    "kr":       7,
    "mlpt":     8,
    "gt":       9,
}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def lighten_color(color, amount=0.5):
    try:
        c = np.array(to_rgb(color))
    except Exception:
        c = np.array(to_rgb("gray"))
    return tuple((1 - amount) * c + amount * np.ones(3))


STACK_COLORS = {
    1: lighten_color("saddlebrown", 0.2),  # svbint  – light brown
    2: lighten_color("red", 0.3),          # krint   – light red
    3: "saddlebrown",                      # svb     – brown
    4: lighten_color("green", 0.4),        # mlptint – light green
    5: "purple",                           # eve-en
    6: "violet",                           # cad
    7: "red",                              # kr
    8: "green",                            # mlpt
    9: "gold",                             # gt
}


# ---------------------------------------------------------------------------
# TIFF I/O
# ---------------------------------------------------------------------------

def load_tiff(filename):
    with tifffile.TiffFile(filename) as tif:
        return tif.asarray()


def get_stack(image, stack_number=1):
    if image.ndim == 2:
        return image
    total = image.shape[0]
    if not (1 <= stack_number <= total):
        raise ValueError(f"stack_number must be 1–{total}, got {stack_number}.")
    return image[stack_number - 1]


# ---------------------------------------------------------------------------
# Display mapping
# ---------------------------------------------------------------------------

def apply_window(img, lo, hi):
    """
    Core ImageJ linear display mapping: stretch [lo, hi] → [0, 1].
    Pixels below lo → 0 (black), pixels above hi → 1 (white).
    """
    img   = img.astype(np.float32)
    denom = hi - lo
    if denom <= 0:
        return np.zeros_like(img)
    return np.clip((img - lo) / denom, 0.0, 1.0)


def autoadjust_brightness_contrast(img, saturation=0.35):
    """
    Auto-stretch: clip saturation% of pixels at each tail (ImageJ Auto behaviour).
    """
    lo = float(np.percentile(img, saturation / 2.0))
    hi = float(np.percentile(img, 100.0 - saturation / 2.0))
    if hi <= lo:
        hi = lo + 1.0
    return apply_window(img, lo, hi)


# ---------------------------------------------------------------------------
# Colorisation
# ---------------------------------------------------------------------------

def apply_stack_color(img_norm, stack_number):
    """Convert a normalised [0,1] grayscale image to RGB using gene colour."""
    if stack_number not in STACK_COLORS:
        raise ValueError(f"No colour defined for stack {stack_number}.")
    color   = np.array(to_rgb(STACK_COLORS[stack_number]), dtype=np.float32)
    colored = np.zeros((*img_norm.shape, 3), dtype=np.float32)
    for i in range(3):
        colored[..., i] = img_norm * color[i]
    return colored


# ---------------------------------------------------------------------------
# Gene lookup
# ---------------------------------------------------------------------------

def get_stack_number_for_gene(gene):
    return GENE_TO_STACK.get(gene, None)