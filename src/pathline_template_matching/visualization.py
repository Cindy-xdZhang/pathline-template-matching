"""Pure rendering for the audited three-panel template-matching figure.

The renderer consumes already evaluated seeds and predictions.  It does not
fit a model, query a template library, open flow files, or alter the evaluated
seed population.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


FIGURE_SIZE_INCHES = (21.0, 5.0)
DEFAULT_DPI = 360
DEFAULT_VIEW = (22.0, -58.0)
IVD_PERCENTILE = 95.0
OUTER_MARGIN = 0.001
PANEL_GAP = 0.0
BOTTOM_MARGIN = 0.03
TOP_MARGIN = 0.08
PANEL_ZOOM = 1.12

COLORS = {
    "ivd": "#f4a261",
    "vortex": "#d62728",
    "non_vortex": "#2468b4",
    "false_positive": "#9c27b0",
    "false_negative": "#f4a261",
}

PANEL_TITLES = (
    "IVD p95 + center pathlines",
    "FMT exact-1NN class assignment",
    "FMT TP / FP / FN / TN against IVD p95",
)
PANEL_LABELS = ("a", "b", "c")
ALIGNMENT_TOLERANCE_POINTS = 1.5


@dataclass(frozen=True, slots=True)
class VisualizationScene:
    """Validated, immutable view of the renderer's scene contract."""

    dataset: str
    title: str
    regime: str
    source_ordinal: int
    bounds: np.ndarray
    seeds: np.ndarray
    reference: np.ndarray
    prediction: np.ndarray
    display_pathlines: tuple[np.ndarray, ...]
    ivd_points: np.ndarray | None
    ivd_mesh_vertices: np.ndarray | None
    ivd_mesh_faces: np.ndarray | None
    ivd_mesh_level: float | None


def _nonempty_text(scene: Mapping[str, object], key: str) -> str:
    if key not in scene:
        raise KeyError(f"scene is missing required field {key!r}")
    value = str(scene[key]).strip()
    if not value:
        raise ValueError(f"scene field {key!r} must be non-empty text")
    return value


def _strict_binary(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "buif":
        raise ValueError(f"{name} must be a one-dimensional numeric 0/1 array")
    if not np.isfinite(array).all() or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only finite 0/1 values")
    return np.asarray(array, dtype=bool)


def validate_scene(scene: Mapping[str, object]) -> VisualizationScene:
    """Validate shape, finiteness, and one-seed/one-label correspondence."""

    if not isinstance(scene, Mapping):
        raise TypeError("scene must be a mapping")
    dataset = _nonempty_text(scene, "dataset")
    title = _nonempty_text(scene, "title")
    regime = _nonempty_text(scene, "regime")
    if "source_ordinal" not in scene:
        raise KeyError("scene is missing required field 'source_ordinal'")
    source_ordinal_value = scene["source_ordinal"]
    if isinstance(source_ordinal_value, (bool, np.bool_)) or not isinstance(
        source_ordinal_value, (int, np.integer)
    ):
        raise ValueError("source_ordinal must be an integer")
    source_ordinal = int(source_ordinal_value)
    if source_ordinal < 0:
        raise ValueError("source_ordinal must be non-negative")

    bounds = np.asarray(scene.get("bounds"), dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("bounds must be a finite array with shape [2,3]")
    if not np.all(bounds[1] > bounds[0]):
        raise ValueError("bounds upper corner must be strictly above its lower corner")

    seeds = np.asarray(scene.get("seeds"), dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1:] != (3,) or len(seeds) == 0:
        raise ValueError("seeds must be a non-empty array with shape [N,3]")
    if not np.isfinite(seeds).all():
        raise ValueError("seeds must contain only finite coordinates")
    reference = _strict_binary(scene.get("reference"), name="reference")
    prediction = _strict_binary(scene.get("prediction"), name="prediction")
    expected_shape = (len(seeds),)
    if reference.shape != expected_shape:
        raise ValueError("reference must contain exactly one label per seed")
    if prediction.shape != expected_shape:
        raise ValueError("prediction must contain exactly one assignment per seed")

    # Optional explicit seed copies make mismatched producer artifacts fail
    # closed, while the compact contract still needs only one canonical array.
    for key in ("reference_seeds", "prediction_seeds"):
        if key not in scene or scene[key] is None:
            continue
        compared = np.asarray(scene[key], dtype=np.float64)
        if compared.shape != seeds.shape or not np.isfinite(compared).all():
            raise ValueError(f"{key} must be a finite array with shape {seeds.shape}")
        if not np.array_equal(compared, seeds):
            raise ValueError(f"{key} must exactly equal the canonical seeds")

    raw_pathlines = scene.get("display_pathlines")
    if not isinstance(raw_pathlines, (list, tuple)) or not raw_pathlines:
        raise ValueError("display_pathlines must be a non-empty list of [Li,4] arrays")
    display_pathlines = []
    for index, value in enumerate(raw_pathlines):
        pathline = np.asarray(value, dtype=np.float64)
        if pathline.ndim != 2 or pathline.shape[1:] != (4,) or len(pathline) < 2:
            raise ValueError(
                f"display_pathlines[{index}] must have shape [Li,4] with Li >= 2"
            )
        if not np.isfinite(pathline).all():
            raise ValueError(f"display_pathlines[{index}] contains NaN or Inf")
        if np.any(np.diff(pathline[:, 3]) < 0):
            raise ValueError(f"display_pathlines[{index}] time must be non-decreasing")
        display_pathlines.append(pathline)

    raw_ivd_points = scene.get("ivd_points")
    ivd_points = None
    if raw_ivd_points is not None:
        ivd_points = np.asarray(raw_ivd_points, dtype=np.float64)
        if ivd_points.ndim != 2 or ivd_points.shape[1:] != (4,) or len(ivd_points) == 0:
            raise ValueError("ivd_points must be None or a non-empty [M,4] array")
        if not np.isfinite(ivd_points).all():
            raise ValueError("ivd_points must contain only finite xyz and IVD values")

    raw_ivd_mesh = scene.get("ivd_mesh")
    mesh_vertices = mesh_faces = None
    mesh_level = None
    if raw_ivd_mesh is not None:
        if not isinstance(raw_ivd_mesh, Mapping):
            raise ValueError("ivd_mesh must be None or a vertices/faces/level mapping")
        mesh_vertices = np.asarray(raw_ivd_mesh.get("vertices"), dtype=np.float64)
        mesh_faces = np.asarray(raw_ivd_mesh.get("faces"))
        mesh_level = float(raw_ivd_mesh.get("level"))
        if (
            mesh_vertices.ndim != 2
            or mesh_vertices.shape[1:] != (3,)
            or len(mesh_vertices) < 3
            or not np.isfinite(mesh_vertices).all()
        ):
            raise ValueError("ivd_mesh vertices must be a finite [V,3] array")
        if (
            mesh_faces.ndim != 2
            or mesh_faces.shape[1:] != (3,)
            or len(mesh_faces) < 1
            or not np.issubdtype(mesh_faces.dtype, np.integer)
            or int(mesh_faces.min()) < 0
            or int(mesh_faces.max()) >= len(mesh_vertices)
        ):
            raise ValueError("ivd_mesh faces must be valid integer [F,3] indices")
        if not np.isfinite(mesh_level):
            raise ValueError("ivd_mesh level must be finite")
        mesh_faces = mesh_faces.astype(np.int64, copy=False)

    return VisualizationScene(
        dataset=dataset,
        title=title,
        regime=regime,
        source_ordinal=source_ordinal,
        bounds=bounds,
        seeds=seeds,
        reference=reference,
        prediction=prediction,
        display_pathlines=tuple(display_pathlines),
        ivd_points=ivd_points,
        ivd_mesh_vertices=mesh_vertices,
        ivd_mesh_faces=mesh_faces,
        ivd_mesh_level=mesh_level,
    )


def confusion_masks(reference: object, prediction: object) -> dict[str, np.ndarray]:
    """Return a strict, mutually exclusive TP/FP/FN/TN partition."""

    targets = _strict_binary(reference, name="reference")
    assigned = _strict_binary(prediction, name="prediction")
    if not len(targets) or assigned.shape != targets.shape:
        raise ValueError("reference and prediction must be same-length non-empty arrays")
    masks = {
        "true_negative": ~targets & ~assigned,
        "true_positive": targets & assigned,
        "false_positive": ~targets & assigned,
        "false_negative": targets & ~assigned,
    }
    memberships = np.stack(tuple(masks.values()), axis=0).sum(axis=0)
    if not np.all(memberships == 1):
        raise RuntimeError("TP/FP/FN/TN do not form an exact partition")
    if np.any(masks["false_positive"] & targets):
        raise RuntimeError("a reference-positive seed was marked false positive")
    if np.any(masks["false_negative"] & ~targets):
        raise RuntimeError("a reference-negative seed was marked false negative")
    return masks


def _new_horizontal_figure():
    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
    panel_width = (1.0 - 2.0 * OUTER_MARGIN - 2.0 * PANEL_GAP) / 3.0
    panel_height = 1.0 - BOTTOM_MARGIN - TOP_MARGIN
    axes = []
    for panel_index in range(3):
        left = OUTER_MARGIN + panel_index * (panel_width + PANEL_GAP)
        rectangle = (left, BOTTOM_MARGIN, panel_width, panel_height)
        axes.append(figure.add_axes(rectangle, projection="3d"))
    return figure, axes


def _axes_alignment_audit(figure, axes) -> dict[str, object]:
    """Measure and enforce the final three-panel rectangle contract in points."""

    if len(axes) != 3:
        raise ValueError("the triptych alignment audit requires exactly three axes")
    figure.canvas.draw()
    width_inches, height_inches = (float(value) for value in figure.get_size_inches())
    rectangles = []
    for label, axis in zip(PANEL_LABELS, axes, strict=True):
        position = axis.get_position()
        rectangle = {
            "panel": label,
            "left": float(position.x0 * width_inches * 72.0),
            "bottom": float(position.y0 * height_inches * 72.0),
            "width": float(position.width * width_inches * 72.0),
            "height": float(position.height * height_inches * 72.0),
            "right": float(position.x1 * width_inches * 72.0),
            "top": float(position.y1 * height_inches * 72.0),
        }
        rectangles.append(rectangle)
    widths = np.asarray([item["width"] for item in rectangles], dtype=np.float64)
    heights = np.asarray([item["height"] for item in rectangles], dtype=np.float64)
    bottoms = np.asarray([item["bottom"] for item in rectangles], dtype=np.float64)
    tops = np.asarray([item["top"] for item in rectangles], dtype=np.float64)
    gutters = np.asarray(
        [
            rectangles[index + 1]["left"] - rectangles[index]["right"]
            for index in range(2)
        ],
        dtype=np.float64,
    )
    label_anchors = [
        {
            "panel": item["panel"],
            "x": float(item["left"] + 0.012 * item["width"]),
            "y": float(item["bottom"] + 0.985 * item["height"]),
        }
        for item in rectangles
    ]
    tolerance = ALIGNMENT_TOLERANCE_POINTS
    checks = {
        "equal_width": float(np.ptp(widths)) <= tolerance,
        "equal_height": float(np.ptp(heights)) <= tolerance,
        "common_bottom": float(np.ptp(bottoms)) <= tolerance,
        "common_top": float(np.ptp(tops)) <= tolerance,
        "nonnegative_gutters": bool(np.all(gutters >= -1e-9)),
        "uniform_gutters": float(np.ptp(gutters)) <= tolerance,
        "common_panel_label_y": float(
            np.ptp([item["y"] for item in label_anchors])
        )
        <= tolerance,
    }
    passed = all(checks.values())
    audit = {
        "schema": "pathline-template-matching.panel-alignment.v1",
        "schema_version": 1,
        "status": "PASS" if passed else "FIX BEFORE DELIVERY",
        "backend": "python-matplotlib",
        "measurement": "final rendered axes rectangles",
        "units": "points",
        "tolerance_points": tolerance,
        "gutter_tolerance_points": tolerance,
        "figure_size_inches": [width_inches, height_inches],
        "figure": {
            "width_pt": width_inches * 72.0,
            "height_pt": height_inches * 72.0,
        },
        "comparable_row": list(PANEL_LABELS),
        "panels": [
            {
                "id": item["panel"],
                "bbox_pt": [
                    item["left"],
                    item["bottom"],
                    item["right"],
                    item["top"],
                ],
                "grid_id": "triptych-grid",
                "row_start": 0,
                "row_stop": 1,
                "col_start": index,
                "col_stop": index + 1,
                "panel_label": item["panel"],
                "panel_label_anchor_pt": [
                    label_anchors[index]["x"],
                    label_anchors[index]["y"],
                ],
            }
            for index, item in enumerate(rectangles)
        ],
        "row_groups": [
            {"id": "triptych-row", "panels": list(PANEL_LABELS)}
        ],
        "column_groups": [],
        "exemptions": [],
        "rectangles": rectangles,
        "horizontal_gutters": gutters.tolist(),
        "panel_label_anchors": label_anchors,
        "maximum_width_deviation_points": float(np.ptp(widths)),
        "maximum_height_deviation_points": float(np.ptp(heights)),
        "maximum_gutter_points": float(np.max(gutters)),
        "checks": checks,
    }
    if not passed:
        raise RuntimeError(f"triptych axes alignment failed: {audit}")
    return audit


def _write_json_without_overwrite(path: Path, value: Mapping[str, object]) -> None:
    """Publish a UTF-8 JSON artifact without replacing existing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_axis(ax, bounds: np.ndarray, view: tuple[float, float], title: str) -> None:
    lower, upper = bounds
    span = upper - lower
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect(span, zoom=PANEL_ZOOM)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("y", labelpad=-2)
    ax.set_zlabel("z", labelpad=-2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-1)
    ax.grid(False)
    ax.set_title(title, fontsize=10, pad=-2)


def _draw_ivd_background(ax, scene: VisualizationScene) -> dict[str, object]:
    if scene.ivd_mesh_vertices is not None:
        ax.add_collection3d(
            Poly3DCollection(
                scene.ivd_mesh_vertices[scene.ivd_mesh_faces],
                facecolor=COLORS["ivd"],
                edgecolor="none",
                alpha=0.27,
                rasterized=True,
            )
        )
        threshold = float(scene.ivd_mesh_level)
        input_count = int(len(scene.ivd_mesh_vertices))
        points = np.empty((0, 3), dtype=np.float32)
        fallback = {
            "used": False,
            "reason": None,
            "source": "whole-loaded-volume IVD p95 isosurface",
        }
        sampling = "complete Marching Cubes IVD p95 isosurface; no triangle subsampling"
        rendered_count = int(len(scene.ivd_mesh_faces))
    elif scene.ivd_points is None:
        points = scene.seeds[scene.reference]
        threshold = None
        input_count = 0
        fallback = {
            "used": True,
            "reason": "ivd_points is None",
            "source": "positive reference seeds",
        }
        sampling = "all positive reference seeds; no interpolation"
        rendered_count = int(len(points))
        ax.text2D(
            0.02,
            0.96,
            "seed-reference fallback (raw IVD unavailable)",
            transform=ax.transAxes,
            fontsize=8,
            color="#7a4b00",
            va="top",
        )
    else:
        threshold = float(np.percentile(scene.ivd_points[:, 3], IVD_PERCENTILE))
        high = scene.ivd_points[:, 3] >= threshold
        points = scene.ivd_points[high, :3]
        input_count = int(len(scene.ivd_points))
        fallback = {"used": False, "reason": None, "source": "ivd_points"}
        sampling = "all loaded-volume points with IVD >= p95; no subsampling"
        rendered_count = int(len(points))
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=COLORS["ivd"],
            s=3.0,
            alpha=0.16,
            depthshade=False,
            linewidths=0,
            rasterized=True,
        )
    return {
        "fallback": fallback,
        "input_count": input_count,
        "rendered_high_count": rendered_count,
        "threshold": threshold,
        "sampling": sampling,
    }


def _draw_center_pathlines(ax, pathlines: tuple[np.ndarray, ...]) -> dict[str, object]:
    relative_times = [pathline[:, 3] - pathline[0, 3] for pathline in pathlines]
    maximum_time = max(float(values[-1]) for values in relative_times)
    color_maximum = maximum_time if maximum_time > 0.0 else 1.0
    time_norm = Normalize(vmin=0.0, vmax=color_maximum)
    time_cmap = plt.get_cmap("viridis")
    for pathline, times in zip(pathlines, relative_times):
        xyz = np.asarray(pathline[:, :3], dtype=np.float64)
        segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
        segment_times = 0.5 * (times[:-1] + times[1:])
        collection = Line3DCollection(
            segments,
            cmap=time_cmap,
            norm=time_norm,
            linewidths=0.58,
            alpha=0.66,
            zorder=3,
            rasterized=True,
        )
        collection.set_array(np.asarray(segment_times, dtype=np.float64))
        ax.add_collection3d(collection)
        ax.scatter(
            pathline[0, 0],
            pathline[0, 1],
            pathline[0, 2],
            color=[time_cmap(time_norm(0.0))],
            s=4.5,
            alpha=0.68,
            depthshade=False,
            zorder=4,
        )
    return {
        "pathline_count": int(len(pathlines)),
        "point_count": int(sum(len(pathline) for pathline in pathlines)),
        "segment_count": int(sum(len(pathline) - 1 for pathline in pathlines)),
        "relative_time_min": 0.0,
        "relative_time_max": maximum_time,
        "color_map": "viridis",
    }


def _draw_evaluated_seed_context(ax, seeds: np.ndarray) -> None:
    """Show the same evaluated seed population used by panels two and three."""

    ax.scatter(
        seeds[:, 0],
        seeds[:, 1],
        seeds[:, 2],
        color="#263238",
        s=1.0,
        alpha=0.045,
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )


def _draw_template_assignment(ax, scene: VisualizationScene) -> None:
    non_vortex = ~scene.prediction
    vortex = scene.prediction
    ax.scatter(
        scene.seeds[non_vortex, 0],
        scene.seeds[non_vortex, 1],
        scene.seeds[non_vortex, 2],
        color=COLORS["non_vortex"],
        s=4.0,
        alpha=0.24,
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        scene.seeds[vortex, 0],
        scene.seeds[vortex, 1],
        scene.seeds[vortex, 2],
        color=COLORS["vortex"],
        s=10.0,
        alpha=0.92,
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )


def _draw_confusion(ax, seeds: np.ndarray, masks: Mapping[str, np.ndarray]) -> None:
    categories = (
        ("true_negative", COLORS["non_vortex"], "o", 2.0, 0.035),
        ("true_positive", COLORS["vortex"], "o", 13.0, 0.92),
        ("false_positive", COLORS["false_positive"], "^", 15.0, 0.90),
        ("false_negative", COLORS["false_negative"], "x", 21.0, 0.95),
    )
    for name, color, marker, size, alpha in categories:
        mask = masks[name]
        ax.scatter(
            seeds[mask, 0],
            seeds[mask, 1],
            seeds[mask, 2],
            color=color,
            marker=marker,
            s=size,
            alpha=alpha,
            depthshade=False,
            linewidths=1.0 if marker == "x" else 0,
            rasterized=True,
        )


def _camera_signature(ax) -> np.ndarray:
    return np.asarray(
        [
            *ax.get_xlim3d(),
            *ax.get_ylim3d(),
            *ax.get_zlim3d(),
            float(ax.elev),
            float(ax.azim),
        ],
        dtype=np.float64,
    )


def _seed_digest(seeds: np.ndarray) -> str:
    canonical = np.ascontiguousarray(seeds, dtype="<f8")
    return sha256(canonical.tobytes()).hexdigest()


def render_template_matching_triptych(
    scene: Mapping[str, object],
    output_path: str | Path,
    *,
    view: tuple[float, float] = DEFAULT_VIEW,
    dpi: int = DEFAULT_DPI,
    pdf_output_path: str | Path | None = None,
    alignment_output_path: str | Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Render the audited triptych and return its PNG path and metadata.

    ``pdf_output_path`` and ``alignment_output_path`` are optional for
    compatibility with the frozen 1.2 report path.  The 2.1 renderer supplies
    both, which exports editable PDF text, rasterized three-dimensional marks,
    and a blocking physical-point alignment audit.
    """

    validated = validate_scene(scene)
    view_values = np.asarray(view, dtype=np.float64)
    if view_values.shape != (2,) or not np.isfinite(view_values).all():
        raise ValueError("view must contain finite (elevation, azimuth) values")
    if not -90.0 <= float(view_values[0]) <= 90.0:
        raise ValueError("view elevation must be within [-90,90] degrees")
    camera_view = (float(view_values[0]), float(view_values[1]))
    if isinstance(dpi, (bool, np.bool_)) or not isinstance(dpi, (int, np.integer)):
        raise ValueError("dpi must be an integer")
    dpi = int(dpi)
    if dpi < 10:
        raise ValueError("dpi must be at least 10")
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png")
    pdf_path = None if pdf_output_path is None else Path(pdf_output_path)
    if pdf_path is not None and pdf_path.suffix.lower() != ".pdf":
        raise ValueError("pdf_output_path must end in .pdf")
    alignment_path = (
        None if alignment_output_path is None else Path(alignment_output_path)
    )
    if alignment_path is not None and alignment_path.suffix.lower() != ".json":
        raise ValueError("alignment_output_path must end in .json")
    requested_paths = [path]
    if pdf_path is not None:
        requested_paths.append(pdf_path)
    if alignment_path is not None:
        requested_paths.append(alignment_path)
    if len({value.resolve() for value in requested_paths}) != len(requested_paths):
        raise ValueError("PNG, PDF, and alignment outputs must use distinct paths")
    existing = [value for value in requested_paths if value.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    for requested in requested_paths:
        requested.parent.mkdir(parents=True, exist_ok=True)

    masks = confusion_masks(validated.reference, validated.prediction)
    with matplotlib.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = _new_horizontal_figure()
        try:
            ivd_audit = _draw_ivd_background(axes[0], validated)
            _draw_evaluated_seed_context(axes[0], validated.seeds)
            pathline_audit = _draw_center_pathlines(
                axes[0], validated.display_pathlines
            )
            _draw_template_assignment(axes[1], validated)
            _draw_confusion(axes[2], validated.seeds, masks)
            for panel_label, axis, panel_title in zip(
                PANEL_LABELS, axes, PANEL_TITLES, strict=True
            ):
                _prepare_axis(axis, validated.bounds, camera_view, panel_title)
                axis.text2D(
                    0.012,
                    0.985,
                    panel_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="#111111",
                    zorder=20,
                )

            figure.text(
                0.5,
                0.98,
                f"{validated.title} | {validated.regime} | source ordinal "
                f"{validated.source_ordinal} | development-only exposed cache",
                ha="center",
                va="top",
                fontsize=9,
            )

            signatures = [_camera_signature(axis) for axis in axes]
            if any(
                not np.array_equal(signatures[0], value)
                for value in signatures[1:]
            ):
                raise RuntimeError(
                    "all three panels must use identical camera parameters"
                )
            alignment_audit = _axes_alignment_audit(figure, axes)
            figure.savefig(path, dpi=dpi, facecolor="white", edgecolor="none")
            if pdf_path is not None:
                figure.savefig(pdf_path, facecolor="white", edgecolor="none")
        finally:
            plt.close(figure)

    if alignment_path is not None:
        _write_json_without_overwrite(alignment_path, alignment_audit)

    confusion_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    camera_parameters = {
        "projection": "orthographic",
        "elevation_degrees": camera_view[0],
        "azimuth_degrees": camera_view[1],
        "physical_bounds": validated.bounds.tolist(),
        "box_aspect": (validated.bounds[1] - validated.bounds[0]).tolist(),
        "panel_zoom": PANEL_ZOOM,
    }
    counts = {
        "sample_count": int(len(validated.seeds)),
        "reference_positive": int(validated.reference.sum()),
        "reference_negative": int((~validated.reference).sum()),
        "prediction_vortex": int(validated.prediction.sum()),
        "prediction_non_vortex": int((~validated.prediction).sum()),
        **confusion_counts,
        "ivd_input_point_count": int(ivd_audit["input_count"]),
        "ivd_rendered_high_point_count": int(ivd_audit["rendered_high_count"]),
        "display_pathline_count": int(pathline_audit["pathline_count"]),
        "display_pathline_point_count": int(pathline_audit["point_count"]),
    }
    metadata = {
        "schema": "pathline-template-matching.triptych.v2",
        "dataset": validated.dataset,
        "title": validated.title,
        "regime": validated.regime,
        "source_ordinal": validated.source_ordinal,
        "image": str(path),
        "pdf": None if pdf_path is None else str(pdf_path),
        "alignment_audit": (
            alignment_audit
            if alignment_path is None
            else {**alignment_audit, "path": str(alignment_path)}
        ),
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "dpi": dpi,
        "layout": {
            "outer_margin_fraction": OUTER_MARGIN,
            "panel_gap_fraction": PANEL_GAP,
            "bottom_margin_fraction": BOTTOM_MARGIN,
            "top_margin_fraction": TOP_MARGIN,
            "header_y_fraction": 0.98,
        },
        "panel_order": list(PANEL_TITLES),
        "panel_labels": list(PANEL_LABELS),
        "prediction_semantics": "precomputed FMT exact-1NN binary assignment",
        "visual_encoding": {
            "ivd_isosurface_and_false_negative": COLORS["false_negative"],
            "predicted_vortex_and_true_positive": COLORS["vortex"],
            "predicted_non_vortex_and_true_negative": COLORS["non_vortex"],
            "false_positive": COLORS["false_positive"],
            "false_positive_marker": "triangle",
            "false_negative_marker": "x",
            "pathline_colormap": "viridis_relative_integration_time",
        },
        "counts": counts,
        "camera": {
            **camera_parameters,
            "identical_across_panels": True,
            "panels": [dict(camera_parameters) for _ in range(3)],
        },
        "fallback": ivd_audit["fallback"],
        "render_sampling": {
            "seed_method": "exact evaluated seeds; no interpolation or densification",
            "original_seed_count": int(len(validated.seeds)),
            "rendered_seed_count_per_assignment_panel": int(len(validated.seeds)),
            "all_panels_use_identical_seed_coordinates": True,
            "panel_one_context_seed_count": int(len(validated.seeds)),
            "seed_sha256": _seed_digest(validated.seeds),
            "ivd_method": ivd_audit["sampling"],
            "ivd_percentile": IVD_PERCENTILE,
            "ivd_threshold": ivd_audit["threshold"],
            "pathline_method": "all supplied center pathlines; no resampling",
            **pathline_audit,
        },
        "export_contract": {
            "png_raster_dpi": dpi,
            "pdf_editable_text": True,
            "pdf_fonttype": 42,
            "three_dimensional_marks_rasterized": True,
            "canvas_bbox_inches_tight": False,
            "artifacts_are_non_overwriting": True,
        },
    }
    return path, metadata


__all__ = [
    "COLORS",
    "ALIGNMENT_TOLERANCE_POINTS",
    "DEFAULT_DPI",
    "DEFAULT_VIEW",
    "FIGURE_SIZE_INCHES",
    "PANEL_TITLES",
    "PANEL_LABELS",
    "VisualizationScene",
    "confusion_masks",
    "render_template_matching_triptych",
    "validate_scene",
]
