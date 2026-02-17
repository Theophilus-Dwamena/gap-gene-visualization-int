import numpy as np
import tifffile
from matplotlib.colors import to_rgb

# Slice names mapping (1-based indexing like ImageJ)
SLICE_NAMES = [
    "svbint",  # 1
    "krint",  # 2
    "svb",  # 3
    "mlptint",  # 4
    "eve-en",  # 5
    "cad",  # 6
    "kr",  # 7
    "mlpt",  # 8
    "gt"  # 9
]

# Gene to slice/stack mapping
GENE_TO_STACK = {
    "svbint": 1,
    "svb(int)": 1,
    "krint": 2,
    "kr(int)": 2,
    "svb": 3,
    "mlptint": 4,
    "mlpt(int)": 4,
    "eve-en": 5,
    "cad": 6,
    "kr": 7,
    "mlpt": 8,
    "gt": 9
}


def lighten_color(color, amount=0.5):
    """
    Lighten a color by mixing with white.

    Parameters:
    - color: named color or RGB tuple
    - amount: 0 -> original color, 1 -> white

    Returns:
    - RGB tuple
    """
    try:
        c = np.array(to_rgb(color))
    except Exception:
        c = np.array(to_rgb("gray"))  # fallback

    white = np.ones(3)
    return tuple((1 - amount) * c + amount * white)


# Stack colors (1-based indexing) - defined after lighten_color function
STACK_COLORS = {
    1: lighten_color("saddlebrown", 0.2),  # svbint - lightbrown
    2: lighten_color("red", 0.3),  # krint - lightred
    3: "saddlebrown",  # svb - brown
    4: lighten_color("green", 0.4),  # mlptint - lightgreen
    5: "purple",  # eve-en
    6: "violet",  # cad
    7: "red",  # kr
    8: "green",  # mlpt
    9: "gold",  # gt
}


def load_tiff(filename):
    """
    Load a TIFF file.

    Parameters:
    - filename: path to TIFF file

    Returns:
    - numpy array (possibly multi-dimensional for stacks)
    """
    with tifffile.TiffFile(filename) as tif:
        img = tif.asarray()
    return img


def get_stack(image, stack_number=1):
    """
    Extract a specific stack from an image (1-based indexing like ImageJ).

    Parameters:
    - image: numpy array
    - stack_number: 1-based stack index

    Returns:
    - 2D numpy array (single slice)
    """
    if image.ndim == 2:
        return image

    total_stacks = image.shape[0]

    if stack_number < 1 or stack_number > total_stacks:
        raise ValueError(f"Stack must be between 1 and {total_stacks}")

    return image[stack_number - 1]


def adjust_brightness_contrast(img, min_val=None, max_val=None):
    """
    Apply linear brightness/contrast adjustment (ImageJ-style).

    Parameters:
    - img: input image
    - min_val: minimum value for clipping
    - max_val: maximum value for clipping

    Returns:
    - normalized image (0-1 range)
    """
    img = img.astype(np.float32)

    if min_val is None:
        min_val = img.min()
    if max_val is None:
        max_val = img.max()

    img = np.clip(img, min_val, max_val)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def autoadjust_brightness_contrast(img, iterations=1, saturation=0.35):
    """
    Auto-adjust brightness/contrast similar to ImageJ Auto button.

    This method uses percentile-based clipping to automatically set
    brightness/contrast levels.

    Parameters:
    - img: input grayscale image
    - iterations: number of auto-adjust passes
    - saturation: percentage of pixels to saturate (default 0.35%)

    Returns:
    - contrast-adjusted image scaled 0–1
    """
    img_work = img.astype(np.float32)

    # Calculate percentiles for clipping
    lower_pct = saturation / 2
    upper_pct = 100 - (saturation / 2)

    for i in range(iterations):
        # Get current range
        lower = np.percentile(img_work, lower_pct)
        upper = np.percentile(img_work, upper_pct)

        # Clip
        img_work = np.clip(img_work, lower, upper)

        # If not last iteration, rescale to full range for next iteration
        if i < iterations - 1:
            min_val = img_work.min()
            max_val = img_work.max()
            if max_val - min_val > 0:
                img_work = (img_work - min_val) / (max_val - min_val) * (img.max() - img.min()) + img.min()

    # Final linear scaling to 0-1
    min_val = img_work.min()
    max_val = img_work.max()

    if max_val - min_val > 0:
        img_work = (img_work - min_val) / (max_val - min_val)
    else:
        img_work = np.zeros_like(img_work)

    return img_work


def apply_stack_color(img, stack_number):
    """
    Convert grayscale image to RGB using stack-specific color.

    Parameters:
    - img: grayscale image (0-1 range)
    - stack_number: 1-based stack index

    Returns:
    - RGB image
    """
    if stack_number not in STACK_COLORS:
        raise ValueError(f"Stack color not defined for stack {stack_number}")

    color = np.array(to_rgb(STACK_COLORS[stack_number]))

    # Expand grayscale to RGB
    colored = np.zeros((*img.shape, 3))

    for i in range(3):
        colored[..., i] = img * color[i]

    return colored


def get_stack_number_for_gene(gene):
    """
    Get the stack number for a given gene name.

    Parameters:
    - gene: gene name (e.g., "kr", "kr(int)", "eve-en")

    Returns:
    - stack number (1-based) or None if not found
    """
    return GENE_TO_STACK.get(gene, None)