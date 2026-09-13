"""Small CPU orthographic renderer for review-time geometry verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .camera import VIEW_BASES, VIEW_ORDER, project_vertices


BACKGROUND = (128, 128, 128)


def _face_colors(mesh: trimesh.Trimesh, kind: str, view: str) -> np.ndarray:
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if kind == "normal_rgb":
        return np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if kind == "position_rgb":
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        lo, hi = mesh.bounds
        normalized = (centers - lo) / np.maximum(hi - lo, 1e-12)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    if kind != "shaded":
        raise ValueError(f"Unknown render kind: {kind}")
    forward = np.asarray(VIEW_BASES[view][2], dtype=np.float64)
    light = -forward + np.asarray([0.25, -0.15, 0.45])
    light /= np.linalg.norm(light)
    intensity = 0.35 + 0.65 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray([190.0, 52.0, 62.0])
    return np.clip(intensity[:, None] * base[None, :], 0, 255).astype(np.uint8)


def render_view(
    mesh: trimesh.Trimesh,
    view: str,
    ortho_scale: float,
    *,
    resolution: int = 512,
    kind: str = "shaded",
    supersample: int = 2,
) -> Image.Image:
    """Render one centered view using a fixed square orthographic range."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    image_xy, depth = project_vertices(vertices, view)
    face_depth = depth[faces].mean(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    forward = np.asarray(VIEW_BASES[view][2], dtype=np.float64)
    visible = (normals @ (-forward)) > 1e-8
    face_order = np.argsort(face_depth)[::-1]
    face_order = face_order[visible[face_order]]
    colors = _face_colors(mesh, kind, view)

    size = int(resolution * supersample)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    scale_px = size / float(ortho_scale)
    pixels = np.empty_like(image_xy)
    pixels[:, 0] = size * 0.5 + image_xy[:, 0] * scale_px
    pixels[:, 1] = size * 0.5 - image_xy[:, 1] * scale_px

    for face_index in face_order:
        polygon = [tuple(point) for point in pixels[faces[face_index]]]
        color = tuple(int(value) for value in colors[face_index])
        draw.polygon(polygon, fill=color)

    return canvas.resize((resolution, resolution), Image.Resampling.LANCZOS)


def render_six_views(
    mesh: trimesh.Trimesh,
    scales: dict[str, float],
    output_dir: Path,
    *,
    resolution: int,
    kind: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in VIEW_ORDER:
        image = render_view(mesh, view, scales[view], resolution=resolution, kind=kind)
        path = output_dir / f"{view}.png"
        image.save(path)
        paths.append(path)
    return paths


def make_strip(paths: list[Path], target: Path, label: str) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    header = 30
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (view, image) in enumerate(zip(VIEW_ORDER, images, strict=True)):
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 8, 8), view.title(), fill="black")
        image.close()
    draw.text((8, height + header - 18), label, fill="white")
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def make_comparison(rows: list[tuple[str, list[Path]]], target: Path) -> None:
    opened = [[Image.open(path).convert("RGB") for path in paths] for _, paths in rows]
    width, height = opened[0][0].size
    label_width, header = 120, 28
    canvas = Image.new("RGB", (label_width + 6 * width, len(rows) * height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for column, view in enumerate(VIEW_ORDER):
        draw.text((label_width + column * width + 8, 8), view.title(), fill="black")
    for row_index, ((label, _), images) in enumerate(zip(rows, opened, strict=True)):
        y = header + row_index * height
        draw.text((8, y + height // 2), label, fill="black")
        for column, image in enumerate(images):
            canvas.paste(image, (label_width + column * width, y))
            image.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)

