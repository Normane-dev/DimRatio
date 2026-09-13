"""Detail-aware, extent-exact, monotone one-axis mesh deformation.

The implementation follows the DAAW specification used by DimRatio.  It keeps
high-normal-variation regions locally rigid and allocates more of the requested
dimension change to geometrically smoother intervals along the target axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


AXIS_INDEX = {"width": 0, "depth": 1, "height": 2, "x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class DetailAwareWarpConfig:
    bins: int = 16
    alpha: float = 4.0
    smoothness: float = 1.0
    empty_rigidity: float = 0.1
    q_min: float = 0.5
    q_max: float = 1.5
    ftol: float = 1e-9
    maxiter: int = 200
    saliency_percentile: float = 75.0


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)
    normals = np.zeros_like(vertices)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(length, 1e-12)
    return normals


def normal_variation_saliency(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean one-ring normal difference per vertex and unique edges."""
    vertices = np.asarray(vertices, dtype=np.float64)
    normals = vertex_normals(vertices, faces) if normals is None else np.asarray(normals, dtype=np.float64)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    edges = _unique_edges(faces)
    edge_saliency = np.clip(1.0 - np.einsum("ij,ij->i", normals[edges[:, 0]], normals[edges[:, 1]]), 0.0, 2.0)
    total = np.zeros(len(vertices), dtype=np.float64)
    count = np.zeros(len(vertices), dtype=np.int64)
    np.add.at(total, edges[:, 0], edge_saliency)
    np.add.at(total, edges[:, 1], edge_saliency)
    np.add.at(count, edges[:, 0], 1)
    np.add.at(count, edges[:, 1], 1)
    score = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return score, edges, edge_saliency


def _bin_statistics(
    coordinates: np.ndarray,
    saliency: np.ndarray,
    cfg: DetailAwareWarpConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = float(coordinates.min()), float(coordinates.max())
    boundaries = np.linspace(lo, hi, cfg.bins + 1, dtype=np.float64)
    bin_id = np.minimum(np.searchsorted(boundaries, coordinates, side="right") - 1, cfg.bins - 1)
    bin_id = np.maximum(bin_id, 0)
    occupied = np.zeros(cfg.bins, dtype=bool)
    bin_saliency = np.zeros(cfg.bins, dtype=np.float64)
    for k in range(cfg.bins):
        values = saliency[bin_id == k]
        if values.size:
            occupied[k] = True
            bin_saliency[k] = np.percentile(values, cfg.saliency_percentile)
    robust = np.zeros(cfg.bins, dtype=np.float64)
    populated = bin_saliency[occupied]
    if populated.size:
        p10, p90 = np.percentile(populated, [10.0, 90.0])
        if p90 > p10 + 1e-12:
            robust[occupied] = np.clip((populated - p10) / (p90 - p10), 0.0, 1.0)
    rigidity = np.full(cfg.bins, cfg.empty_rigidity, dtype=np.float64)
    rigidity[occupied] = 1.0 + cfg.alpha * robust[occupied]
    return boundaries, bin_id, bin_saliency, rigidity


def _solve_local_scales(
    rigidity: np.ndarray,
    widths: np.ndarray,
    target_scale: float,
    cfg: DetailAwareWarpConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    k_count = len(rigidity)

    def objective(q: np.ndarray) -> float:
        return float(np.dot(rigidity, (q - 1.0) ** 2) + cfg.smoothness * np.sum(np.diff(q) ** 2))

    def jacobian(q: np.ndarray) -> np.ndarray:
        grad = 2.0 * rigidity * (q - 1.0)
        if k_count > 1:
            diff = np.diff(q)
            grad[:-1] -= 2.0 * cfg.smoothness * diff
            grad[1:] += 2.0 * cfg.smoothness * diff
        return grad

    target_length = float(target_scale * widths.sum())
    constraint = {
        "type": "eq",
        "fun": lambda q: float(np.dot(widths, q) - target_length),
        "jac": lambda q: widths,
    }
    initial = np.full(k_count, target_scale, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=[(cfg.q_min, cfg.q_max)] * k_count,
        constraints=[constraint],
        options={"ftol": cfg.ftol, "maxiter": cfg.maxiter, "disp": False},
    )
    q = np.asarray(result.x, dtype=np.float64)
    residual = float(np.dot(widths, q) - target_length)
    valid = bool(result.success and np.all(np.isfinite(q)) and np.all(q > 0.0) and abs(residual) < 1e-8)
    return q, {
        "success": valid,
        "optimizer_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": objective(q),
        "constraint_residual": residual,
    }


def piecewise_axis_warp(
    coordinates: np.ndarray,
    boundaries: np.ndarray,
    local_scales: np.ndarray,
) -> np.ndarray:
    """Evaluate the centered continuous cumulative piecewise-linear warp."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    widths = np.diff(boundaries)
    cumulative = np.concatenate(([0.0], np.cumsum(widths * local_scales)))
    bin_id = np.minimum(np.searchsorted(boundaries, coordinates, side="right") - 1, len(local_scales) - 1)
    bin_id = np.maximum(bin_id, 0)
    mapped_from_min = cumulative[bin_id] + local_scales[bin_id] * (coordinates - boundaries[bin_id])
    source_center = 0.5 * (boundaries[0] + boundaries[-1])
    target_min = source_center - 0.5 * cumulative[-1]
    return target_min + mapped_from_min


def uniform_axis_warp(vertices: np.ndarray, axis: int | str, target_scale: float) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    axis_index = AXIS_INDEX[axis] if isinstance(axis, str) else int(axis)
    out = vertices.copy()
    center = 0.5 * (vertices[:, axis_index].min() + vertices[:, axis_index].max())
    out[:, axis_index] = center + target_scale * (vertices[:, axis_index] - center)
    return out


def detail_aware_axis_warp(
    vertices: np.ndarray,
    faces: np.ndarray,
    axis: int | str,
    target_scale: float,
    *,
    normals: np.ndarray | None = None,
    config: DetailAwareWarpConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply DAAW and return transformed vertices plus serializable diagnostics."""
    cfg = config if isinstance(config, DetailAwareWarpConfig) else DetailAwareWarpConfig(**(config or {}))
    axis_index = AXIS_INDEX[axis] if isinstance(axis, str) else int(axis)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not (cfg.q_min <= target_scale <= cfg.q_max):
        raise ValueError(f"target_scale={target_scale} is outside [{cfg.q_min}, {cfg.q_max}]")
    before_extent = np.ptp(vertices, axis=0)
    if before_extent[axis_index] <= 1e-12:
        raise ValueError("Target axis has zero extent")

    saliency, edges, edge_saliency = normal_variation_saliency(vertices, faces, normals)
    boundaries, _, bin_saliency, rigidity = _bin_statistics(vertices[:, axis_index], saliency, cfg)
    widths = np.diff(boundaries)
    q, solver = _solve_local_scales(rigidity, widths, float(target_scale), cfg)
    fallback = not solver["success"]
    if fallback:
        warped = uniform_axis_warp(vertices, axis_index, target_scale)
        q = np.full(cfg.bins, target_scale, dtype=np.float64)
    else:
        warped = vertices.copy()
        warped[:, axis_index] = piecewise_axis_warp(vertices[:, axis_index], boundaries, q)

    after_extent = np.ptp(warped, axis=0)
    expected = before_extent.copy()
    expected[axis_index] *= target_scale
    extent_error = after_extent - expected
    monotone = bool(np.all(q > 0.0))
    diagnostics = {
        "method": "detail_aware" if not fallback else "uniform_fallback",
        "axis": axis_index,
        "target_scale": float(target_scale),
        "config": asdict(cfg),
        "source_extent_xyz": before_extent.tolist(),
        "target_extent_xyz": after_extent.tolist(),
        "expected_extent_xyz": expected.tolist(),
        "extent_error_xyz": extent_error.tolist(),
        "max_abs_extent_error": float(np.max(np.abs(extent_error))),
        "non_target_max_abs_extent_error": float(np.max(np.abs(np.delete(extent_error, axis_index)))),
        "monotone": monotone,
        "fallback": fallback,
        "boundaries": boundaries.tolist(),
        "bin_saliency": bin_saliency.tolist(),
        "rigidity": rigidity.tolist(),
        "local_scales": q.tolist(),
        "solver": solver,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "edge_count": int(len(edges)),
        "detail_distortion": detail_distortion(vertices, warped, edges, edge_saliency),
    }
    return warped, diagnostics


def detail_distortion(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    edges: np.ndarray,
    edge_saliency: np.ndarray | None = None,
) -> float:
    """Saliency-weighted absolute log change of edge length."""
    source_length = np.linalg.norm(source_vertices[edges[:, 0]] - source_vertices[edges[:, 1]], axis=1)
    target_length = np.linalg.norm(target_vertices[edges[:, 0]] - target_vertices[edges[:, 1]], axis=1)
    valid = (source_length > 1e-12) & (target_length > 1e-12)
    if not np.any(valid):
        return 0.0
    change = np.abs(np.log(target_length[valid] / source_length[valid]))
    # With saliency supplied, this metric measures deformation specifically at
    # geometric details; adding a unit baseline would let the many smooth
    # tessellation edges dominate the intended detail-preservation measure.
    weights = np.ones_like(change) if edge_saliency is None else np.asarray(edge_saliency)[valid] + 1e-8
    return float(np.average(change, weights=weights))


def transform_by_axis_scales(
    vertices: np.ndarray,
    faces: np.ndarray,
    scales_xyz: dict[str, float],
    *,
    mode: str = "detail_aware",
    normals: np.ndarray | None = None,
    config: DetailAwareWarpConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform one changed paper axis; keep uniform mode for ablation."""
    scales = np.array([scales_xyz["width"], scales_xyz["depth"], scales_xyz["height"]], dtype=np.float64)
    changed = np.flatnonzero(np.abs(scales - 1.0) > 1e-10)
    if len(changed) > 1:
        raise ValueError("DAAW showcase expects one independently changed axis")
    if len(changed) == 0:
        return np.asarray(vertices, dtype=np.float64).copy(), {"method": mode, "axis": None, "target_scale": 1.0}
    axis_index = int(changed[0])
    if mode == "uniform":
        warped = uniform_axis_warp(vertices, axis_index, float(scales[axis_index]))
        return warped, {"method": "uniform", "axis": axis_index, "target_scale": float(scales[axis_index])}
    if mode != "detail_aware":
        raise ValueError(f"Unsupported transform mode: {mode}")
    return detail_aware_axis_warp(
        vertices,
        faces,
        axis_index,
        float(scales[axis_index]),
        normals=normals,
        config=config,
    )
