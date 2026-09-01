"""Rank-likelihood adapters and rendering for the fixed four-flow report.

The inherited parent scenes remain immutable.  This module only renames the
RankLikelihood primary prediction fields into the already-audited exact join
used by :mod:`source_centered_visualization`, then renders the same three-panel
spatial evidence without describing the classifier as FMT.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .portable_flow import canonical_array_sha256
from .source_centered_visualization import (
    BLOCKS,
    BLOCK_INDEX,
    CENTER_COUNT,
    PATHLINES_PER_BLOCK,
    SOURCE_ORDINAL,
    CombinedCenterScene,
    ValidProjection,
    _block_legend_handles,
    _draw_block_pathlines,
    bind_valid_projection,
    combine_parent_block_scenes,
)
from .visualization import (
    COLORS,
    DEFAULT_VIEW,
    FIGURE_SIZE_INCHES,
    PANEL_LABELS,
    _axes_alignment_audit,
    _camera_signature,
    _draw_confusion,
    _draw_evaluated_seed_context,
    _draw_ivd_background,
    _draw_template_assignment,
    _new_horizontal_figure,
    _prepare_axis,
    _seed_digest,
    _write_json_without_overwrite,
    confusion_masks,
    validate_scene,
)


PNG_DPI = 360
HEADER_AXES_TOP = 0.84
PANEL_TITLES = (
    "IVD p95 + legacy/expanded center pathlines",
    "Source-rank likelihood template classification",
    "TP / FP / FN / TN against IVD p95",
)
SCENE_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_combined_scene.v1"
)
SCENE_ARRAY_NAMES = (
    "bounds",
    "seeds",
    "reference",
    "prediction",
    "center_seed_index",
    "primary_score",
    "legacy_valid",
    "expanded_valid",
    "display_pathlines",
    "display_pathline_block_index",
    "ivd_mesh_vertices",
    "ivd_mesh_faces",
    "ivd_mesh_normals",
    "ivd_mesh_values",
    "ivd_mesh_level",
    "metadata_json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def publish_bytes_without_overwrite(path: str | Path, payload: bytes) -> None:
    """Atomically publish complete bytes without any replace race."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
        if os.name != "nt":
            directory_descriptor = os.open(
                destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file_without_overwrite(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Atomically hard-link one complete same-filesystem file into place."""

    source_path = Path(source)
    destination_path = Path(destination)
    _require(source_path.is_file(), f"publication source is missing: {source_path}")
    _require(
        source_path.resolve() != destination_path.resolve(),
        "publication source and destination must differ",
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {destination_path}"
        )
    with source_path.open("r+b") as complete:
        complete.flush()
        os.fsync(complete.fileno())
    os.link(source_path, destination_path)
    if os.name != "nt":
        directory_descriptor = os.open(
            destination_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _savefig_without_overwrite(
    figure: Any,
    path: Path,
    *,
    dpi: int | None = None,
) -> None:
    """Render to a same-directory temporary file, then link it into place."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        options: dict[str, Any] = {
            "format": path.suffix.removeprefix("."),
            "facecolor": "white",
            "edgecolor": "none",
        }
        if dpi is not None:
            options["dpi"] = int(dpi)
        figure.savefig(temporary, **options)
        with temporary.open("r+b") as rendered:
            rendered.flush()
            os.fsync(rendered.fileno())
        os.link(temporary, path)
        if os.name != "nt":
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def adapt_unique_primary_prediction(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expose only the frozen primary arrays to the inherited exact join."""

    mapping = {
        "unique_dataset": "unique_dataset",
        "unique_source_ordinal": "unique_source_ordinal",
        "unique_source_index": "unique_source_index",
        "unique_center_seed_index": "unique_center_seed_index",
        "paired_score": "unique_primary_spatial_score",
        "legacy_valid": "unique_legacy_valid",
        "expanded_valid": "unique_expanded_valid",
        "paired_prediction": "unique_primary_prediction",
    }
    _require(
        all(source in arrays for source in mapping.values()),
        "RankLikelihood unique primary projection is incomplete",
    )
    result = {target: np.asarray(arrays[source]) for target, source in mapping.items()}
    _require(
        np.array_equal(
            np.asarray(arrays.get("unique_combined_valid"), dtype=np.bool_),
            np.asarray(result["legacy_valid"], dtype=np.bool_)
            | np.asarray(result["expanded_valid"], dtype=np.bool_),
        ),
        "RankLikelihood combined-valid mask differs from the two block masks",
    )
    return result


def adapt_valid_primary_prediction(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expose only primary valid-row fields; controls cannot enter the plot."""

    mapping = {
        "valid_dataset": "valid_dataset",
        "valid_source_ordinal": "valid_source_ordinal",
        "valid_source_index": "valid_source_index",
        "valid_scale_id": "valid_scale_id",
        "valid_center_seed_index": "valid_center_seed_index",
        "valid_scale_block_index": "valid_scale_block_index",
        "valid_assigned_row_index": "valid_assigned_row_index",
        "valid_paired_score": "valid_primary_score",
        "valid_paired_prediction": "valid_primary_prediction",
    }
    _require(
        all(source in arrays for source in mapping.values()),
        "RankLikelihood valid primary projection is incomplete",
    )
    return {target: np.asarray(arrays[source]) for target, source in mapping.items()}


def combine_rank_likelihood_parent_scenes(
    *,
    legacy_metadata: Mapping[str, Any],
    legacy_arrays: Mapping[str, np.ndarray],
    expanded_metadata: Mapping[str, Any],
    expanded_arrays: Mapping[str, np.ndarray],
    unique_prediction: Mapping[str, np.ndarray],
    title: str,
) -> CombinedCenterScene:
    """Run the inherited union join with an explicit primary-only adapter."""

    return combine_parent_block_scenes(
        legacy_metadata=legacy_metadata,
        legacy_arrays=legacy_arrays,
        expanded_metadata=expanded_metadata,
        expanded_arrays=expanded_arrays,
        unique_prediction=adapt_unique_primary_prediction(unique_prediction),
        title=title,
    )


def bind_rank_likelihood_valid_projection(
    *,
    legacy_metadata: Mapping[str, Any],
    legacy_arrays: Mapping[str, np.ndarray],
    expanded_metadata: Mapping[str, Any],
    expanded_arrays: Mapping[str, np.ndarray],
    unique_prediction: Mapping[str, np.ndarray],
    valid_prediction: Mapping[str, np.ndarray],
) -> ValidProjection:
    """Exact-join primary center predictions back to all parent-valid rows."""

    return bind_valid_projection(
        legacy_metadata=legacy_metadata,
        legacy_arrays=legacy_arrays,
        expanded_metadata=expanded_metadata,
        expanded_arrays=expanded_arrays,
        unique_prediction=adapt_unique_primary_prediction(unique_prediction),
        valid_prediction=adapt_valid_primary_prediction(valid_prediction),
    )


def bind_rank_likelihood_table_only_projection(
    *,
    arm: str,
    legacy_metadata: Mapping[str, Any],
    legacy_arrays: Mapping[str, np.ndarray],
    expanded_metadata: Mapping[str, Any],
    expanded_arrays: Mapping[str, np.ndarray],
    unique_prediction: Mapping[str, np.ndarray],
    valid_prediction: Mapping[str, np.ndarray],
) -> ValidProjection:
    """Exact-join a frozen non-primary arm for table metrics, never plotting."""

    fields = {
        "negative_ecdf": (
            "unique_control_spatial_score",
            "unique_control_prediction",
            "valid_control_score",
            "valid_control_prediction",
        ),
        "direct_rank_mean_top5": (
            "unique_direct_rank_mean_score",
            "unique_direct_rank_mean_prediction",
            "valid_direct_rank_mean_score",
            "valid_direct_rank_mean_prediction",
        ),
    }
    _require(arm in fields, f"unsupported table-only arm: {arm}")
    unique_score, unique_class, valid_score, valid_class = fields[arm]
    _require(
        all(
            name in unique_prediction
            for name in (unique_score, unique_class)
        )
        and all(name in valid_prediction for name in (valid_score, valid_class)),
        f"table-only {arm} projection is incomplete",
    )
    unique = adapt_unique_primary_prediction(unique_prediction)
    valid = adapt_valid_primary_prediction(valid_prediction)
    unique["paired_score"] = np.asarray(unique_prediction[unique_score])
    unique["paired_prediction"] = np.asarray(unique_prediction[unique_class])
    valid["valid_paired_score"] = np.asarray(valid_prediction[valid_score])
    valid["valid_paired_prediction"] = np.asarray(valid_prediction[valid_class])
    return bind_valid_projection(
        legacy_metadata=legacy_metadata,
        legacy_arrays=legacy_arrays,
        expanded_metadata=expanded_metadata,
        expanded_arrays=expanded_arrays,
        unique_prediction=unique,
        valid_prediction=valid,
    )


def scene_arrays(scene: CombinedCenterScene, metadata_json: str) -> dict[str, np.ndarray]:
    """Return the frozen pickle-free RankLikelihood scene payload."""

    arrays = {
        "bounds": np.asarray(scene.bounds, dtype=np.float64),
        "seeds": np.asarray(scene.seeds, dtype=np.float64),
        "reference": np.asarray(scene.reference, dtype=np.bool_),
        "prediction": np.asarray(scene.prediction, dtype=np.bool_),
        "center_seed_index": np.asarray(scene.center_seed_index, dtype=np.int64),
        "primary_score": np.asarray(scene.paired_score, dtype=np.float64),
        "legacy_valid": np.asarray(scene.legacy_valid, dtype=np.bool_),
        "expanded_valid": np.asarray(scene.expanded_valid, dtype=np.bool_),
        "display_pathlines": np.asarray(scene.display_pathlines, dtype=np.float64),
        "display_pathline_block_index": np.asarray(
            scene.display_pathline_block_index, dtype=np.int8
        ),
        "ivd_mesh_vertices": np.asarray(scene.ivd_mesh_vertices),
        "ivd_mesh_faces": np.asarray(scene.ivd_mesh_faces),
        "ivd_mesh_normals": np.asarray(scene.ivd_mesh_normals),
        "ivd_mesh_values": np.asarray(scene.ivd_mesh_values),
        "ivd_mesh_level": np.asarray(scene.ivd_mesh_level),
        "metadata_json": np.asarray(str(metadata_json)),
    }
    _require(tuple(arrays) == SCENE_ARRAY_NAMES, "RankLikelihood scene order changed")
    _require(arrays["metadata_json"].dtype.kind == "U", "scene metadata must be Unicode")
    return arrays


def render_source_rank_likelihood_triptych(
    scene: CombinedCenterScene,
    *,
    png_path: str | Path,
    pdf_path: str | Path,
    svg_path: str | Path,
    alignment_path: str | Path,
    view: tuple[float, float] = DEFAULT_VIEW,
    dpi: int = PNG_DPI,
) -> dict[str, Any]:
    """Render one primary-only triptych with editable text exports."""

    _require(all("FMT" not in title for title in PANEL_TITLES), "Panel B must not be labelled FMT")
    requested = tuple(Path(value) for value in (png_path, pdf_path, svg_path, alignment_path))
    suffixes = (".png", ".pdf", ".svg", ".json")
    _require(
        all(path.suffix.lower() == suffix for path, suffix in zip(requested, suffixes, strict=True)),
        "render output suffix contract changed",
    )
    _require(len({path.resolve() for path in requested}) == 4, "render output paths must be distinct")
    existing = [path for path in requested if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite rendered artifacts: {existing}")
    for path in requested:
        path.parent.mkdir(parents=True, exist_ok=True)
    _require(int(dpi) == PNG_DPI, "frozen PNG DPI changed")
    view_array = np.asarray(view, dtype=np.float64)
    _require(view_array.shape == (2,) and np.isfinite(view_array).all(), "camera view is invalid")
    mapping = {
        "dataset": scene.dataset,
        "title": scene.title,
        "regime": "family-held-out exposed-development source-rank likelihood",
        "source_ordinal": SOURCE_ORDINAL,
        "bounds": scene.bounds,
        "seeds": scene.seeds,
        "reference": scene.reference,
        "prediction": scene.prediction,
        "display_pathlines": list(scene.display_pathlines),
        "ivd_points": None,
        "ivd_mesh": {
            "vertices": scene.ivd_mesh_vertices,
            "faces": scene.ivd_mesh_faces,
            "level": float(np.asarray(scene.ivd_mesh_level).reshape(())),
        },
    }
    validated = validate_scene(mapping)
    masks = confusion_masks(validated.reference, validated.prediction)
    with matplotlib.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    ):
        figure, axes = _new_horizontal_figure()
        try:
            for axis in axes:
                position = axis.get_position()
                axis.set_position(
                    (position.x0, position.y0, position.width, HEADER_AXES_TOP - position.y0)
                )
            ivd_audits = [_draw_ivd_background(axis, validated) for axis in axes]
            _require(all(item == ivd_audits[0] for item in ivd_audits[1:]), "IVD background differs by panel")
            _draw_evaluated_seed_context(axes[0], validated.seeds)
            pathline_audit = _draw_block_pathlines(
                axes[0], validated.display_pathlines, scene.display_pathline_block_index
            )
            _draw_template_assignment(axes[1], validated)
            _draw_confusion(axes[2], validated.seeds, masks)
            camera_view = (float(view_array[0]), float(view_array[1]))
            for label, axis, title in zip(PANEL_LABELS, axes, PANEL_TITLES, strict=True):
                _prepare_axis(axis, validated.bounds, camera_view, title)
                axis.text2D(
                    0.012,
                    0.985,
                    label,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="#111111",
                    zorder=20,
                )
            figure.legend(
                handles=_block_legend_handles(),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.925),
                ncol=2,
                fontsize=7.0,
                frameon=False,
                handlelength=2.2,
                borderaxespad=0.0,
                columnspacing=2.0,
            )
            figure.text(
                0.5,
                0.99,
                f"{scene.title} | primary source-rank likelihood template prediction | "
                f"source ordinal {SOURCE_ORDINAL} | exposed development",
                ha="center",
                va="top",
                fontsize=9,
            )
            signatures = [_camera_signature(axis) for axis in axes]
            _require(
                all(np.array_equal(signatures[0], item) for item in signatures[1:]),
                "camera or bounds differ across panels",
            )
            alignment = _axes_alignment_audit(figure, axes)
            _savefig_without_overwrite(figure, requested[0], dpi=PNG_DPI)
            _savefig_without_overwrite(figure, requested[1])
            _savefig_without_overwrite(figure, requested[2])
        finally:
            plt.close(figure)
    _write_json_without_overwrite(requested[3], alignment)
    camera = {
        "projection": "orthographic",
        "elevation_degrees": float(view_array[0]),
        "azimuth_degrees": float(view_array[1]),
        "physical_bounds": validated.bounds.tolist(),
        "box_aspect": (validated.bounds[1] - validated.bounds[0]).tolist(),
        "identical_across_panels": True,
    }
    return {
        "schema": "pathline_template_matching.source_centered_rank_likelihood_triptych.v1",
        "dataset": scene.dataset,
        "title": scene.title,
        "source_ordinal": SOURCE_ORDINAL,
        "source_index": scene.source_index,
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "dpi": int(dpi),
        "panel_order": list(PANEL_TITLES),
        "prediction_semantics": (
            "primary dual_histogram_llr prediction per combined-valid unique center"
        ),
        "counts": {
            "combined_valid_center_count": int(len(scene.center_seed_index)),
            "reference_positive": int(scene.reference.sum()),
            "reference_negative": int((~scene.reference).sum()),
            "prediction_positive": int(scene.prediction.sum()),
            "prediction_negative": int((~scene.prediction).sum()),
            **{name: int(mask.sum()) for name, mask in masks.items()},
        },
        "camera": camera,
        "alignment_audit": {**alignment, "path": str(requested[3])},
        "ivd_audit": {"identical_across_panels": True, "panels": ivd_audits},
        "pathline_audit": pathline_audit,
        "render_sampling": {
            "center_method": "all combined-valid unique centers in ascending center_seed_index order",
            "center_sha256": canonical_array_sha256(scene.center_seed_index),
            "seed_sha256": _seed_digest(scene.seeds),
            "pathline_method": "fixed first 120 from each immutable parent block scene",
        },
        "visual_encoding": {
            "predicted_vortex_and_true_positive": COLORS["vortex"],
            "predicted_non_vortex_and_true_negative": COLORS["non_vortex"],
            "false_positive": COLORS["false_positive"],
            "false_negative": COLORS["false_negative"],
        },
        "export_contract": {
            "png": str(requested[0]),
            "pdf": str(requested[1]),
            "svg": str(requested[2]),
            "alignment": str(requested[3]),
            "png_dpi": int(dpi),
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
            "editable_text": True,
            "three_dimensional_marks_rasterized": True,
            "artifacts_are_non_overwriting": True,
        },
        "content_sha256_inputs": {
            "center_seed_index": canonical_array_sha256(scene.center_seed_index),
            "primary_score": canonical_array_sha256(scene.paired_score),
            "prediction": canonical_array_sha256(scene.prediction),
            "reference": canonical_array_sha256(scene.reference),
            "display_pathlines": canonical_array_sha256(scene.display_pathlines),
            "display_pathline_block_index": canonical_array_sha256(
                scene.display_pathline_block_index
            ),
        },
    }


__all__ = [
    "BLOCKS",
    "BLOCK_INDEX",
    "CENTER_COUNT",
    "PANEL_TITLES",
    "PATHLINES_PER_BLOCK",
    "SCENE_ARRAY_NAMES",
    "SCENE_SCHEMA",
    "SOURCE_ORDINAL",
    "CombinedCenterScene",
    "ValidProjection",
    "adapt_unique_primary_prediction",
    "adapt_valid_primary_prediction",
    "bind_rank_likelihood_valid_projection",
    "bind_rank_likelihood_table_only_projection",
    "combine_rank_likelihood_parent_scenes",
    "publish_bytes_without_overwrite",
    "publish_file_without_overwrite",
    "render_source_rank_likelihood_triptych",
    "scene_arrays",
]
