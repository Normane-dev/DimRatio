"""Procedural demonstration mesh; no paper benchmark asset is distributed."""

from __future__ import annotations

import numpy as np
import trimesh


def _placed_box(extents, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _placed_cylinder(radius, height, center) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=20)
    mesh.apply_translation(center)
    return mesh


def make_demo_chair() -> trimesh.Trimesh:
    """Create a simple chair-like mesh with cushions, arms, and four legs."""
    parts: list[trimesh.Trimesh] = []

    seat = trimesh.creation.uv_sphere(count=[24, 16])
    seat.apply_scale([0.68, 0.55, 0.13])
    seat.apply_translation([0.0, 0.0, 0.72])
    parts.append(seat)

    back = trimesh.creation.uv_sphere(count=[24, 16])
    back.apply_scale([0.68, 0.12, 0.58])
    back.apply_translation([0.0, 0.48, 1.18])
    parts.append(back)

    parts.extend([
        _placed_box([0.16, 1.04, 0.48], [-0.68, 0.0, 0.91]),
        _placed_box([0.16, 1.04, 0.48], [0.68, 0.0, 0.91]),
    ])

    for x in (-0.56, 0.56):
        for y in (-0.39, 0.39):
            parts.append(_placed_cylinder(0.045, 0.62, [x, y, 0.31]))

    mesh = trimesh.util.concatenate(parts)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def normalize_longest_extent(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center a mesh and normalize its longest XYZ extent to one."""
    mesh = mesh.copy()
    center = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    mesh.apply_translation(-center)
    longest = float(np.max(mesh.extents))
    if longest <= 0:
        raise ValueError("Mesh has zero extent")
    mesh.apply_scale(1.0 / longest)
    return mesh

