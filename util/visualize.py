"""Visualization helpers for few-shot segmentation predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import rasterio


DEFAULT_PALETTE: Sequence[Tuple[int, int, int]] = (
    (0, 0, 0),  # background
    (0, 197, 255),
    (255, 94, 87),
    (255, 211, 0),
    (141, 255, 128),
    (174, 103, 255),
    (255, 159, 243),
    (64, 255, 203),
    (250, 177, 160),
    (162, 155, 254),
)


EXP_DISASTER_BINARY_PALETTE: Sequence[Tuple[int, int, int]] = (
    (0, 0, 0),  # background
    (255, 0, 0),  # landslide
    (0, 0, 255),  # flood
)


def tensor_to_u8_image(
    tensor: torch.Tensor,
    *,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Convert a CxHxW or HxW tensor into an uint8 numpy image."""
    if tensor.ndimension() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "tensor_to_u8_image expects a singleton batch when receiving a 4D tensor; "
                f"got batch dimension {tensor.shape[0]}."
            )
        tensor = tensor.squeeze(0)

    if tensor.ndimension() == 3 and tensor.shape[0] in (1, 3):
        array = tensor.detach().cpu().numpy()
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        array = np.transpose(array, (1, 2, 0))
    elif tensor.ndimension() == 2:
        array = tensor.detach().cpu().numpy()[..., None]
        array = np.repeat(array, 3, axis=2)
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}.")

    if mean is not None and std is not None:
        mean_arr = np.array(mean).reshape(1, 1, -1)
        std_arr = np.array(std).reshape(1, 1, -1)
        array = array * std_arr + mean_arr
        array = array * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def build_palette(num_classes: int, colormap: str = "default") -> List[Tuple[int, int, int]]:
    """Return a palette large enough for ``num_classes`` labels."""

    if colormap == "default":
        palette: List[Tuple[int, int, int]] = list(DEFAULT_PALETTE)
        if num_classes > len(palette):
            repeats = int(np.ceil(num_classes / len(palette)))
            palette = (palette * repeats)[:num_classes]
        else:
            palette = palette[:num_classes]
        return palette

    if colormap == "exp_disaster_binary":
        if num_classes > len(EXP_DISASTER_BINARY_PALETTE):
            raise ValueError(
                "exp_disaster_binary colormap only supports up to 3 classes " f"but received {num_classes}."
            )
        return list(EXP_DISASTER_BINARY_PALETTE[:num_classes])

    raise ValueError(f"Unsupported colormap '{colormap}'.")


def mask_to_color(
    mask: np.ndarray,
    palette: Sequence[Tuple[int, int, int]],
    *,
    ignore_index: Optional[int] = None,
    ignore_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Map label ids in ``mask`` to RGB colors using ``palette``."""
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    mask_max = int(mask.max()) if mask.size > 0 else 0
    if mask_max >= len(palette):
        raise ValueError("Palette does not cover all labels in the mask.")

    palette_arr = np.asarray(palette, dtype=np.uint8)
    color_mask = palette_arr[np.clip(mask, 0, len(palette_arr) - 1)]

    if ignore_index is not None:
        color_mask[mask == ignore_index] = np.array(ignore_color, dtype=np.uint8)
    return color_mask


def overlay_mask(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Blend ``color_mask`` on top of ``image`` using factor ``alpha``."""
    if image.shape != color_mask.shape:
        raise ValueError("Image and mask must have matching shapes.")
    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_png(path: str | Path, array: np.ndarray) -> None:
    """Persist an RGB array as a PNG image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(str(path))


def save_geotiff(path: str | Path, mask: np.ndarray, profile: dict) -> None:
    """Store ``mask`` as a GeoTIFF based on a reference ``profile``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    updated_profile = profile.copy()
    updated_profile.update(
        dtype=mask.dtype,
        count=1,
        nodata=profile.get("nodata", None),
    )
    with rasterio.open(path, "w", **updated_profile) as dst:
        dst.write(mask, 1)
