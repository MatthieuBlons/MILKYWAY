from copy import copy
import numpy as np
from skimage.transform import resize
from pathlib import Path
from typing import Any, Sequence

def as_model_list(
    model: str | Path | Sequence[str | Path],
) -> list[str | Path]:
    """Accept one checkpoint path or a list of them."""
    if isinstance(model, (str, Path)):
        return [model]

    models = list(model)

    if not models:
        raise ValueError("`model` must contain at least one checkpoint.")

    return models

def make_background_neutral(
    heatmap: np.ndarray, ref_min: float | None = None, ref_max: float | None = None
):
    """
    For the sake of visibility, puts the background neural.

    Parameters
    ----------
    heatmap : np.ndarray.
        heatmap to process

    ref_min : float
        fixed min value of the cmap, if none, fixed to min(heatmap).

    ref_max : float
        fixed max value of the cmap, if none, fixed to min(heatmap).
    """
    heatmap_neutral = copy(heatmap)

    ref_min = heatmap_neutral.min() if ref_min is None else ref_min
    ref_max = heatmap_neutral.max() if ref_max is None else ref_max

    heatmap_neutral[heatmap_neutral == 0] = np.mean([ref_min, ref_max])

    return heatmap_neutral


def add_titlebox(ax, text):
    """
    Add text box on a axis

    Parameters
    ----------
    ax : matplotlib axes object

    text : str
        text to write at the bottom-left corner of the axes.
    """
    ax.text(
        0.05,
        0.05,
        text,
        horizontalalignment="left",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=20,
    )
    return ax


def set_axes_color(ax, color="orange"):
    """
    Set axes color.

    Parameters
    ----------
    ax : matplotlib axes object

    color : str
        color for the axes.
    """
    dirs = ["bottom", "top", "left", "right"]
    args = {x: False for x in dirs}
    args.update({"label" + x: False for x in dirs})
    args.update({"axis": "both", "which": "both"})
    ax.tick_params(**args)
    for sp in ax.spines:
        ax.spines[sp].set_color(color)
        ax.spines[sp].set_linewidth(5)
    return ax


def blend_images(im1, im2, background_color=(0, 0, 0), alpha=0.5):
    """
    Blend two RGB images, ignoring the background of the second image if needed.
    Returns:
    - np.ndarray: Blended RGB image.
    """
    # TODO assert both rgb
    # assert same shape
    shape = im1.shape
    if im2.shape != shape:
        im2 = resize(im2, shape, preserve_range=True, order=0)
    # Create a mask where im2 is not background
    mask = np.any(im2 != np.array(background_color), axis=-1)
    # Blend the images
    blended_image = im1.copy()
    for c in range(3):  # Iterate over RGB channels
        blended_image[:, :, c] = np.where(
            mask, (alpha) * im2[:, :, c] + (1 - alpha) * im1[..., c], im1[..., c]
        )
    # TODO add the possibility to Scale back to 0-255 range and convert to uint8 (or other dtype) if needed
    return blended_image
