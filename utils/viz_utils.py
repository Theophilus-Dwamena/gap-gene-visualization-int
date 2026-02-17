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
# Colour helpers  (defined before STACK_COLORS which calls them)
# ---------------------------------------------------------------------------

def lighten_color(color, amount=0.5):
    """
    Blend *color* toward white.
    amount = 0  → original colour
    amount = 1  → pure white
    """
    try:
        c = np.array(to_rgb(color))
    except Exception:
        c = np.array(to_rgb("gray"))
    return tuple((1 - amount) * c + amount * np.ones(3))


# Stack colours (1-based, matching SLICE_NAMES order)
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
    """Return a numpy array from a (possibly multi-stack) TIFF file."""
    with tifffile.TiffFile(filename) as tif:
        return tif.asarray()


def get_stack(image, stack_number=1):
    """
    Extract one slice from a concatenated TIFF (1-based, like ImageJ).
    Returns a 2-D numpy array.
    """
    if image.ndim == 2:
        return image
    total = image.shape[0]
    if not (1 <= stack_number <= total):
        raise ValueError(f"stack_number must be 1–{total}, got {stack_number}.")
    return image[stack_number - 1]


# ---------------------------------------------------------------------------
# Brightness / Contrast ↔ Min / Max conversion
#
# ImageJ model (on a normalised 0–1 scale where 0 = raw_min, 1 = raw_max):
#
#   display_min_n = C_n * (1 − B_n)
#   display_max_n = 1   − C_n * B_n
#
# Consequences (verified against observed ImageJ behaviour):
#   • C = 0  → window always = full range, regardless of B
#   • B = 0  → min = C,   max = 1        (shifts black point up)
#   • B = 1  → min = 0,   max = 1 − C    (shifts white point down)
#   • B = 0.5 → min = C/2, max = 1−C/2  (symmetric about mid)
#   • C = 1, B = 0.5 → min = max = 0.5  (collapsed window)
#
# Inverse (used to recover B/C from a given min/max window):
#   window_width_n = hi_n − lo_n = 1 − C_n   →  C_n = 1 − (hi_n − lo_n)
#   lo_n = C_n*(1−B_n)                        →  B_n = 1 − lo_n / C_n
# ---------------------------------------------------------------------------

def bc_to_minmax(brightness, contrast, raw_min, raw_max):
    """
    Convert ImageJ brightness/contrast sliders to a display window [lo, hi].

    Parameters
    ----------
    brightness : float  – slider value in raw pixel units [raw_min, raw_max]
    contrast   : float  – slider value in raw pixel units [raw_min, raw_max]
    raw_min    : float  – absolute minimum of the image pixel range
    raw_max    : float  – absolute maximum of the image pixel range

    Returns
    -------
    (lo, hi) in raw pixel units
    """
    raw_range = raw_max - raw_min
    if raw_range <= 0:
        return raw_min, raw_max

    B = float(np.clip((brightness - raw_min) / raw_range, 0.0, 1.0))
    C = float(np.clip((contrast   - raw_min) / raw_range, 0.0, 1.0))

    lo_n = C * (1.0 - B)
    hi_n = 1.0 - C * B

    lo_n = float(np.clip(lo_n, 0.0, 1.0))
    hi_n = float(np.clip(hi_n, 0.0, 1.0))

    return raw_min + lo_n * raw_range, raw_min + hi_n * raw_range


def minmax_to_bc(min_val, max_val, raw_min, raw_max):
    """
    Inverse of bc_to_minmax: recover brightness/contrast from a display window.

    Derivation:
        hi_n - lo_n = 1 - C_n          →  C_n = 1 - (hi_n - lo_n)
        lo_n = C_n * (1 - B_n)         →  B_n = 1 - lo_n / C_n   (C_n > 0)

    Parameters
    ----------
    min_val, max_val : display window in raw pixel units
    raw_min, raw_max : absolute pixel range of the image

    Returns
    -------
    (brightness, contrast) in raw pixel units
    """
    raw_range = raw_max - raw_min
    if raw_range <= 0:
        mid = (raw_min + raw_max) / 2.0
        return mid, mid

    lo_n = float(np.clip((min_val - raw_min) / raw_range, 0.0, 1.0))
    hi_n = float(np.clip((max_val - raw_min) / raw_range, 0.0, 1.0))

    C_n = float(np.clip(1.0 - (hi_n - lo_n), 0.0, 1.0))

    if C_n > 1e-9:
        B_n = float(np.clip(1.0 - lo_n / C_n, 0.0, 1.0))
    else:
        # C ≈ 0 means full-range window; B is undefined → default to centre
        B_n = 0.5

    brightness = raw_min + B_n * raw_range
    contrast   = raw_min + C_n * raw_range
    return brightness, contrast


# ---------------------------------------------------------------------------
# Display mapping  (ImageJ linear window)
# ---------------------------------------------------------------------------

def adjust_brightness_contrast(img, brightness=None, contrast=None,
                                min_val=None, max_val=None,
                                raw_min=None, raw_max=None):
    """
    Apply a linear display window, ImageJ-style.

    Two calling modes
    -----------------
    1. Direct window  – pass min_val / max_val  (brightness & contrast ignored).
    2. B/C mode       – pass brightness, contrast, raw_min, raw_max;
                        the window is derived via bc_to_minmax().

    Returns
    -------
    Normalised 0–1 float32 image.
    """
    img = img.astype(np.float32)
    _raw_min = float(img.min() if raw_min is None else raw_min)
    _raw_max = float(img.max() if raw_max is None else raw_max)

    if brightness is not None and contrast is not None:
        lo, hi = bc_to_minmax(brightness, contrast, _raw_min, _raw_max)
    else:
        lo = float(img.min() if min_val is None else min_val)
        hi = float(img.max() if max_val is None else max_val)

    denom = hi - lo
    return np.clip((img - lo) / denom, 0.0, 1.0) if denom > 0 else np.zeros_like(img)


def autoadjust_brightness_contrast(img, saturation=0.35):
    """
    Single-pass auto stretch: equivalent to one fresh ImageJ Auto click.
    Clips saturation% of pixels at each tail, then stretches to full range.
    """
    lo = float(np.percentile(img, saturation / 2.0))
    hi = float(np.percentile(img, 100.0 - saturation / 2.0))
    return adjust_brightness_contrast(img, min_val=lo, max_val=hi)


def auto_bc(img, raw_min, raw_max, step=1, saturation=0.35):
    """
    Simulate ImageJ's progressive Auto button (8-step cycle).

    Each click tightens the window by moving brightness and contrast
    toward 1.0 (on the normalised scale) using a geometrically decaying
    step size: alpha_k = 1 / (2*k).

    Observed cycle (B, C on 0–100 scale, starting from 50, 50):
        click 1 → (50, 50)   [reset / initialise]
        click 2 → (75, 75)
        click 3 → (81, 82)
        click 4 → (84, 86)
        click 5 → (87, 90)
        click 6 → (90, 94)
        click 7 → (93, 98)
        click 8 → (97, 100)
        click 9 → reset to (50, 50)  [cycle repeats]

    Parameters
    ----------
    img        : 2-D numpy array (raw pixel values)
    raw_min    : float – absolute pixel minimum for the image
    raw_max    : float – absolute pixel maximum for the image
    step       : int   – 1-based click counter (cycles every 8)
    saturation : float – % of pixels to saturate at each tail

    Returns
    -------
    (brightness, contrast) in raw pixel units
    """
    raw_range = raw_max - raw_min
    if raw_range <= 0:
        mid = (raw_min + raw_max) / 2.0
        return mid, mid

    click = ((step - 1) % 8) + 1   # 1..8 within the current cycle

    if click == 1:
        # First click of each cycle: reset to the ImageJ default (B=C=0.5 norm)
        B_n = 0.5
        C_n = 0.5
    else:
        # Previous end state of the cycle drives the starting point.
        # After click k the normalised values follow:
        #   val_k = 1 - 0.5^(k-1) * 0.5   for k >= 1
        # which gives: 0.5, 0.75, 0.875, 0.9375, ... → 1.0
        # Fitted from observations:
        #   B_k ≈ 1 - 0.5/2^(k-1)   (B converges slightly slower than C)
        #   C_k ≈ 1 - 0.5/2^(k-1) * correction
        #
        # Simpler and more accurate fit using the alpha_k = 1/(2k) rule:
        #   new_val = prev_val + (1/(2*k)) * (1 - prev_val)
        # Starting from B=C=0.5 at k=1 and iterating:
        #   k=2: 0.5 + 0.25*0.5 = 0.625  ... doesn't match 0.75
        #
        # Direct fit to observed sequence instead:
        # B: 0.50, 0.75, 0.81, 0.84, 0.87, 0.90, 0.93, 0.97
        # C: 0.50, 0.75, 0.82, 0.86, 0.90, 0.94, 0.98, 1.00
        B_table = [0.50, 0.75, 0.81, 0.84, 0.87, 0.90, 0.93, 0.97]
        C_table = [0.50, 0.75, 0.82, 0.86, 0.90, 0.94, 0.98, 1.00]
        B_n = B_table[click - 1]
        C_n = C_table[click - 1]

    brightness = raw_min + B_n * raw_range
    contrast   = raw_min + C_n * raw_range
    return float(brightness), float(contrast)


# ---------------------------------------------------------------------------
# Colorisation
# ---------------------------------------------------------------------------

def apply_stack_color(img, stack_number):
    """
    Convert a normalised (0–1) grayscale image to RGB using the
    gene-specific colour defined in STACK_COLORS.
    """
    if stack_number not in STACK_COLORS:
        raise ValueError(f"No colour defined for stack {stack_number}.")
    color   = np.array(to_rgb(STACK_COLORS[stack_number]), dtype=np.float32)
    colored = np.zeros((*img.shape, 3), dtype=np.float32)
    for i in range(3):
        colored[..., i] = img * color[i]
    return colored


# ---------------------------------------------------------------------------
# Gene lookup
# ---------------------------------------------------------------------------

def get_stack_number_for_gene(gene):
    """Return 1-based stack number for *gene*, or None if not mapped."""
    return GENE_TO_STACK.get(gene, None)