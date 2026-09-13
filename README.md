# DimRatio — Anonymous Review Artifact

This repository accompanies **DimRatio**, a framework for **Geometric Multi-view Generation**. Given an object image and target dimensions, the task is to generate a complete set of standard-view product images whose proportions correctly reflect the requested dimensional changes.

Existing multi-view generation methods often provide limited viewpoint coverage, while text- or layout-based proportion controls do not impose a shared geometric condition across views. DimRatio addresses these limitations through two coordinated blocks:

1. **ViewForge Block** completes the missing standard views and establishes a complete shared geometric representation of the input object.
2. **ShapeSync Block** transforms the shared geometry according to the target dimension ratios. Its **Detail-Aware Axis Warp (DAAW)** adaptively distributes the deformation while preserving salient geometric details, and the transformed geometry provides consistent conditions for all affected views.

![DimRatio pipeline](assets/pipeline.png)

## Method overview

During input preparation, the reference image is expanded into horizontal views and reconstructed as a 3D proxy. The ViewForge Block centers and normalizes this proxy, locks a shared six-view camera setup, and renders geometric conditions for completing the basic six views. Given a target proportion, the ShapeSync Block converts the source and target dimensions into axis-wise scale ratios, applies DAAW to the shared proxy, and renders transformed normal and position conditions with the same cameras. These conditions guide the same geometry-conditioned multi-view generation model to produce the dimension-controlled views.

![Qualitative dimension-control results](assets/complete_function.png)

The figure shows the basic six-view completion and independent `+20%` changes in depth, height, and width. Dashed boxes indicate the views affected by each dimension change.

## Review artifact scope

The included CPU demo verifies the released geometric components of ShapeSync. It starts from a procedurally generated proxy mesh, applies a requested width, depth, or height ratio using DAAW, fixes the cameras over the complete allowed condition set, and exports shaded, normal-RGB, and position-RGB six-view renderings.

This anonymous artifact intentionally does **not** distribute the paper benchmark, commercial product imagery, complete experiment repository, model checkpoints, or large-scale inference/evaluation pipeline. The full implementation, benchmark construction tools, trained artifacts where redistribution is permitted, and complete qualitative results will be released after publication.

## Coordinate and view convention

- `+X`: width
- `+Y`: depth
- `+Z`: height
- front camera: located on `-Y`, looking toward the object
- standard view order: `front, right, back, left, top, bottom`

The demo normalizes the longest source extent to `1.0`. For the predefined condition set

```text
baseline, width/depth/height × {0.8, 1.2}
```

and safety margin `m = 1.16`, all views and variants share the same projection range:

```text
1.0 × 1.2 × 1.16 = 1.392
```

The camera is computed once and is not refitted for an individual target condition.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py --axis width --scale 1.2 --output outputs/width_p20
```

Try a negative height condition:

```bash
python run_demo.py --axis height --scale 0.8 --output outputs/height_m20
```

Run the verification tests:

```bash
pytest -q
```

## Demo outputs

Each run writes:

```text
outputs/<condition>/
├── source_normalized.obj
├── target_detail_aware.obj
├── diagnostics.json
├── camera.json
├── comparison.png
├── baseline/shaded/{front,right,back,left,top,bottom}.png
└── target/{shaded,normal_rgb,position_rgb}/...
```

`diagnostics.json` records the exact source/target extents, local axis scales, monotonicity check, optimizer residual, and detail-distortion value.

## Minimal API

```python
from dimratio.daaw import transform_by_axis_scales

warped_vertices, diagnostics = transform_by_axis_scales(
    vertices,
    faces,
    {"width": 1.2, "depth": 1.0, "height": 1.0},
    mode="detail_aware",
    normals=vertex_normals,
)
```

## GitHub Pages

An accompanying static project page is provided in [`docs/index.html`](docs/index.html). After uploading the repository, enable GitHub Pages with the `/docs` folder as its source.

## Anonymity and release statement

The repository contains no author names, affiliations, private server paths, credentials, dataset account information, or identifying metadata. During peer review, please keep the repository history anonymous as well.

The complete codebase and evaluation resources will be released after publication.
