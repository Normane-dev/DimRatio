#!/usr/bin/env python3
"""Run the anonymous DimRatio geometry and fixed-camera verification demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from dimratio.camera import fixed_projection_scales
from dimratio.daaw import DetailAwareWarpConfig, transform_by_axis_scales
from dimratio.demo_mesh import make_demo_chair, normalize_longest_extent
from dimratio.render import make_comparison, render_six_views


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("width", "depth", "height"), default="width")
    parser.add_argument("--scale", type=float, default=1.2)
    parser.add_argument("--mode", choices=("detail_aware", "uniform"), default="detail_aware")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/demo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0.5 <= args.scale <= 1.5:
        raise ValueError("The review demo supports scale factors in [0.5, 1.5]")

    source = normalize_longest_extent(make_demo_chair())
    scales_xyz = {"width": 1.0, "depth": 1.0, "height": 1.0}
    scales_xyz[args.axis] = args.scale
    warped_vertices, diagnostics = transform_by_axis_scales(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        scales_xyz,
        mode=args.mode,
        normals=np.asarray(source.vertex_normals),
        config=DetailAwareWarpConfig(**config["daaw"]),
    )
    target = trimesh.Trimesh(vertices=warped_vertices, faces=source.faces, process=False)
    target.fix_normals()

    camera_cfg = config["camera"]
    camera_scales = fixed_projection_scales(
        np.asarray(source.extents),
        camera_cfg["allowed_conditions"],
        margin=float(camera_cfg["margin"]),
        global_across_views=bool(camera_cfg["global_across_views"]),
    )
    resolution = int(camera_cfg["resolution"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source.export(output / "source_normalized.obj")
    target.export(output / f"target_{args.mode}.obj")

    baseline_paths = render_six_views(
        source, camera_scales, output / "baseline/shaded", resolution=resolution, kind="shaded"
    )
    shaded_paths = render_six_views(
        target, camera_scales, output / "target/shaded", resolution=resolution, kind="shaded"
    )
    render_six_views(
        target, camera_scales, output / "target/normal_rgb", resolution=resolution, kind="normal_rgb"
    )
    render_six_views(
        target, camera_scales, output / "target/position_rgb", resolution=resolution, kind="position_rgb"
    )
    make_comparison(
        [("Source", baseline_paths), (f"{args.axis.title()} × {args.scale:g}", shaded_paths)],
        output / "comparison.png",
    )

    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output / "camera.json").write_text(
        json.dumps(
            {
                "axis_contract": config["axis_contract"],
                "allowed_conditions": camera_cfg["allowed_conditions"],
                "margin": camera_cfg["margin"],
                "global_across_views": camera_cfg["global_across_views"],
                "ortho_scale_by_view": camera_scales,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "camera": camera_scales, "diagnostics": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

