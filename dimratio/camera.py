"""Fixed orthographic camera protocol used by the verification demo."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


VIEW_ORDER = ("front", "right", "back", "left", "top", "bottom")

# Each tuple contains image-right, image-up, and camera-forward vectors.
# Camera-forward points from the camera toward the object.
VIEW_BASES = {
    "front": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "right": ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "left": ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "bottom": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
}


def _scale_vector(condition: Mapping[str, float]) -> np.ndarray:
    return np.asarray(
        [condition["width"], condition["depth"], condition["height"]],
        dtype=np.float64,
    )


def fixed_projection_scales(
    source_extent_xyz: np.ndarray,
    allowed_conditions: Iterable[Mapping[str, float]],
    *,
    margin: float = 1.16,
    global_across_views: bool = True,
) -> dict[str, float]:
    """Compute a projection range once from every allowed dimensional variant.

    The source mesh is assumed to be centered. For every standard view, the
    largest horizontal/vertical projected extent across the allowed condition
    set is multiplied by a fixed safety margin. With ``global_across_views``,
    one maximum is shared by all six views.
    """
    extent = np.asarray(source_extent_xyz, dtype=np.float64)
    if extent.shape != (3,) or np.any(extent <= 0):
        raise ValueError("source_extent_xyz must contain three positive values")
    if margin <= 1.0:
        raise ValueError("margin must be greater than 1.0")

    conditions = list(allowed_conditions)
    if not conditions:
        raise ValueError("allowed_conditions must not be empty")

    ranges: dict[str, float] = {}
    for view in VIEW_ORDER:
        right, up, _ = (np.asarray(v, dtype=np.float64) for v in VIEW_BASES[view])
        largest = 0.0
        for condition in conditions:
            variant_extent = extent * _scale_vector(condition)
            horizontal = float(np.dot(np.abs(right), variant_extent))
            vertical = float(np.dot(np.abs(up), variant_extent))
            largest = max(largest, horizontal, vertical)
        ranges[view] = margin * largest

    if global_across_views:
        shared = max(ranges.values())
        return {view: shared for view in VIEW_ORDER}
    return ranges


def project_vertices(vertices: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Project centered XYZ vertices into orthographic image coordinates."""
    if view not in VIEW_BASES:
        raise KeyError(f"Unknown view: {view}")
    right, up, forward = (np.asarray(v, dtype=np.float64) for v in VIEW_BASES[view])
    vertices = np.asarray(vertices, dtype=np.float64)
    image_xy = np.column_stack((vertices @ right, vertices @ up))
    depth = vertices @ forward
    return image_xy, depth
