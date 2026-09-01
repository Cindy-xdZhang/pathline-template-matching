"""Combined-center scenes and rendering for the source-centered report.

The two Phase 3.1 scale blocks are immutable scene inputs, not two
classifiers.  This module joins their valid center populations to the one
paired-center prediction, preserves the parent IVD geometry, and renders one
three-panel figure per flow.  It never fits a template model or selects a
candidate, source, flow, pathline, threshold, or scale block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .portable_flow import canonical_array_sha256
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


# Publication exports must retain editable SVG/PDF text.  These settings are
# repeated inside the render-time rc_context so importing another module cannot
# silently change them.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


CENTER_COUNT = 64_000
SOURCE_ORDINAL = 2
BLOCKS = ("legacy_2_1", "expanded_3_1")
BLOCK_INDEX = {"legacy_2_1": 0, "expanded_3_1": 1}
PATHLINES_PER_BLOCK = 120
PNG_DPI = 360
FINAL_WIDTH_MM = 533.4
HEADER_AXES_TOP = 0.84
PANEL_TITLES = (
    "IVD p95 + legacy/expanded center pathlines",
    "FMT source-centered paired-scale classification",
    "TP / FP / FN / TN against IVD p95",
)
SCENE_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_combined_scene.v1"
)
SCENE_ARRAY_NAMES = (
    "bounds",
    "seeds",
    "reference",
    "prediction",
    "center_seed_index",
    "paired_score",
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
BLOCK_STYLE = {
    0: {"name": "legacy_2_1", "color": "#2468b4", "linestyle": "solid"},
    1: {"name": "expanded_3_1", "color": "#8e5ea2", "linestyle": "dashed"},
}


@dataclass(frozen=True, slots=True)
class CombinedCenterScene:
    """One immutable four-flow report scene before serialization."""

    dataset: str
    title: str
    source_index: int
    bounds: np.ndarray
    seeds: np.ndarray
    reference: np.ndarray
    prediction: np.ndarray
    center_seed_index: np.ndarray
    paired_score: np.ndarray
    legacy_valid: np.ndarray
    expanded_valid: np.ndarray
    display_pathlines: np.ndarray
    display_pathline_block_index: np.ndarray
    ivd_mesh_vertices: np.ndarray
    ivd_mesh_faces: np.ndarray
    ivd_mesh_normals: np.ndarray
    ivd_mesh_values: np.ndarray
    ivd_mesh_level: np.ndarray


@dataclass(frozen=True, slots=True)
class ValidProjection:
    """Primary row-projection population, kept out of the plotted panels."""

    reference: np.ndarray
    prediction: np.ndarray
    score: np.ndarray
    dataset: np.ndarray
    source_ordinal: np.ndarray
    source_index: np.ndarray
    scale_id: np.ndarray
    center_seed_index: np.ndarray
    scale_block_index: np.ndarray
    assigned_row_index: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metadata_identity(metadata: Mapping[str, Any], *, block: str) -> tuple[str, int, int]:
    _require(metadata.get("scale_block_id") == block, f"parent block identity changed: {block}")
    dataset = str(metadata.get("dataset", "")).strip()
    _require(bool(dataset), "parent scene dataset is missing")
    source_ordinal = int(metadata.get("source_ordinal", -1))
    source_index = int(metadata.get("source_index", -1))
    _require(source_ordinal == SOURCE_ORDINAL, "parent scene source ordinal changed")
    _require(source_index >= 0, "parent scene source index is invalid")
    return dataset, source_ordinal, source_index


def _parent_block_projection(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    block: str,
) -> dict[str, np.ndarray | int | str]:
    dataset, source_ordinal, source_index = _metadata_identity(metadata, block=block)
    block_index = BLOCK_INDEX[block]
    required = (
        "bounds",
        "seeds",
        "reference",
        "valid_scale_id",
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "display_pathlines",
        "ivd_mesh_vertices",
        "ivd_mesh_faces",
        "ivd_mesh_normals",
        "ivd_mesh_values",
        "ivd_mesh_level",
    )
    _require(all(name in arrays for name in required), f"parent scene arrays are incomplete: {block}")
    seeds = np.asarray(arrays["seeds"])
    reference = np.asarray(arrays["reference"])
    center = np.asarray(arrays["valid_center_seed_index"])
    assigned = np.asarray(arrays["valid_assigned_row_index"])
    scale_id = np.asarray(arrays["valid_scale_id"])
    scale_block = np.asarray(arrays["valid_scale_block_index"])
    count = len(center)
    _require(
        seeds.shape == (count, 3)
        and reference.shape == (count,)
        and assigned.shape == scale_id.shape == scale_block.shape == (count,),
        f"parent scene valid-row shapes changed: {block}",
    )
    _require(np.isfinite(seeds).all(), f"parent scene seeds are non-finite: {block}")
    _require(
        center.dtype.kind in "iu"
        and assigned.dtype.kind in "iu"
        and scale_id.dtype.kind in "iu"
        and scale_block.dtype.kind in "iu",
        f"parent scene identity dtype changed: {block}",
    )
    center = np.asarray(center, dtype=np.int64)
    assigned = np.asarray(assigned, dtype=np.int64)
    scale_id = np.asarray(scale_id, dtype=np.int32)
    scale_block = np.asarray(scale_block, dtype=np.int8)
    _require(count > 0 and len(np.unique(center)) == count, f"duplicate parent center: {block}")
    _require(len(np.unique(assigned)) == count, f"duplicate parent assigned row: {block}")
    _require(
        np.all((center >= 0) & (center < CENTER_COUNT)),
        f"parent center is outside 40-cubed grid: {block}",
    )
    _require(
        np.array_equal(assigned, block_index * CENTER_COUNT + center),
        f"parent assigned identity changed: {block}",
    )
    _require(np.all(scale_block == block_index), f"parent block index changed: {block}")
    _require(
        np.array_equal(scale_block.astype(np.int32), scale_id // 1000),
        f"parent scale/block identity changed: {block}",
    )
    binary = np.asarray(reference)
    _require(
        binary.dtype.kind in "buif"
        and np.isfinite(binary).all()
        and np.all(np.isin(binary, (0, 1))),
        f"parent reference is not strict binary: {block}",
    )
    pathlines = np.asarray(arrays["display_pathlines"])
    _require(
        pathlines.ndim == 3
        and pathlines.shape[0] >= PATHLINES_PER_BLOCK
        and pathlines.shape[2] == 4
        and pathlines.shape[1] >= 2
        and np.isfinite(pathlines).all(),
        f"parent pathline array cannot supply fixed prefix: {block}",
    )
    return {
        "dataset": dataset,
        "source_ordinal": source_ordinal,
        "source_index": source_index,
        "bounds": np.asarray(arrays["bounds"]),
        "seeds": np.asarray(seeds),
        "reference": np.asarray(binary, dtype=np.bool_),
        "center": center,
        "assigned": assigned,
        "scale_id": scale_id,
        "scale_block": scale_block,
        "pathlines": np.asarray(pathlines[:PATHLINES_PER_BLOCK]),
        "ivd_mesh_vertices": np.asarray(arrays["ivd_mesh_vertices"]),
        "ivd_mesh_faces": np.asarray(arrays["ivd_mesh_faces"]),
        "ivd_mesh_normals": np.asarray(arrays["ivd_mesh_normals"]),
        "ivd_mesh_values": np.asarray(arrays["ivd_mesh_values"]),
        "ivd_mesh_level": np.asarray(arrays["ivd_mesh_level"]),
    }


def _require_equal_parent_backgrounds(
    legacy: Mapping[str, Any], expanded: Mapping[str, Any]
) -> None:
    for field in (
        "dataset",
        "source_ordinal",
        "source_index",
    ):
        _require(legacy[field] == expanded[field], f"parent scene {field} differs by block")
    for field in (
        "bounds",
        "ivd_mesh_vertices",
        "ivd_mesh_faces",
        "ivd_mesh_normals",
        "ivd_mesh_values",
        "ivd_mesh_level",
    ):
        _require(
            np.array_equal(np.asarray(legacy[field]), np.asarray(expanded[field])),
            f"parent IVD/background array differs by block: {field}",
        )


def combine_parent_block_scenes(
    *,
    legacy_metadata: Mapping[str, Any],
    legacy_arrays: Mapping[str, np.ndarray],
    expanded_metadata: Mapping[str, Any],
    expanded_arrays: Mapping[str, np.ndarray],
    unique_prediction: Mapping[str, np.ndarray],
    title: str,
) -> CombinedCenterScene:
    """Create one combined-valid center scene after a fail-closed exact join."""

    legacy = _parent_block_projection(
        legacy_metadata, legacy_arrays, block="legacy_2_1"
    )
    expanded = _parent_block_projection(
        expanded_metadata, expanded_arrays, block="expanded_3_1"
    )
    _require_equal_parent_backgrounds(legacy, expanded)
    required_unique = (
        "unique_dataset",
        "unique_source_ordinal",
        "unique_source_index",
        "unique_center_seed_index",
        "paired_score",
        "legacy_valid",
        "expanded_valid",
        "paired_prediction",
    )
    _require(
        all(name in unique_prediction for name in required_unique),
        "unique-center prediction arrays are incomplete",
    )
    centers = np.asarray(unique_prediction["unique_center_seed_index"])
    _require(
        centers.dtype.kind in "iu"
        and centers.shape == (CENTER_COUNT,)
        and np.array_equal(centers, np.arange(CENTER_COUNT, dtype=centers.dtype)),
        "unique-center prediction order must be exactly ascending 0..63999",
    )
    dataset = str(legacy["dataset"])
    source_index = int(legacy["source_index"])
    _require(
        np.all(np.asarray(unique_prediction["unique_dataset"]) == dataset)
        and np.all(np.asarray(unique_prediction["unique_source_ordinal"]) == SOURCE_ORDINAL)
        and np.all(np.asarray(unique_prediction["unique_source_index"]) == source_index),
        "unique-center dataset/source identity differs from parent scenes",
    )
    legacy_valid = np.asarray(unique_prediction["legacy_valid"], dtype=np.bool_)
    expanded_valid = np.asarray(unique_prediction["expanded_valid"], dtype=np.bool_)
    prediction = np.asarray(unique_prediction["paired_prediction"], dtype=np.bool_)
    score = np.asarray(unique_prediction["paired_score"], dtype=np.float64)
    for name, values in (
        ("legacy_valid", legacy_valid),
        ("expanded_valid", expanded_valid),
        ("paired_prediction", prediction),
        ("paired_score", score),
    ):
        _require(values.shape == (CENTER_COUNT,), f"unique-center shape changed: {name}")
    _require(np.isfinite(score).all(), "paired score contains NaN or Inf")
    expected_legacy = np.zeros(CENTER_COUNT, dtype=np.bool_)
    expected_expanded = np.zeros(CENTER_COUNT, dtype=np.bool_)
    expected_legacy[np.asarray(legacy["center"], dtype=np.int64)] = True
    expected_expanded[np.asarray(expanded["center"], dtype=np.int64)] = True
    _require(
        np.array_equal(legacy_valid, expected_legacy),
        "prediction legacy_valid mask differs from parent scene membership",
    )
    _require(
        np.array_equal(expanded_valid, expected_expanded),
        "prediction expanded_valid mask differs from parent scene membership",
    )

    seed_lookup = np.full((CENTER_COUNT, 3), np.nan, dtype=np.float64)
    reference_lookup = np.full(CENTER_COUNT, -1, dtype=np.int8)
    for block in (legacy, expanded):
        block_centers = np.asarray(block["center"], dtype=np.int64)
        block_seeds = np.asarray(block["seeds"], dtype=np.float64)
        block_reference = np.asarray(block["reference"], dtype=np.bool_)
        occupied = np.isfinite(seed_lookup[block_centers, 0])
        if occupied.any():
            overlap = block_centers[occupied]
            _require(
                np.array_equal(seed_lookup[overlap], block_seeds[occupied]),
                "same center has different coordinates in the two parent scenes",
            )
            _require(
                np.array_equal(
                    reference_lookup[overlap], block_reference[occupied].astype(np.int8)
                ),
                "same center has different IVD-p95 reference across blocks",
            )
        seed_lookup[block_centers] = block_seeds
        reference_lookup[block_centers] = block_reference.astype(np.int8)
    combined = legacy_valid | expanded_valid
    _require(combined.any(), "combined-valid center population is empty")
    selected_centers = np.flatnonzero(combined).astype(np.int64)
    _require(
        np.isfinite(seed_lookup[selected_centers]).all()
        and np.all(np.isin(reference_lookup[selected_centers], (0, 1))),
        "combined-valid parent lookup is incomplete",
    )
    pathlines = np.concatenate(
        (
            np.asarray(legacy["pathlines"]),
            np.asarray(expanded["pathlines"]),
        ),
        axis=0,
    )
    pathline_blocks = np.concatenate(
        (
            np.zeros(PATHLINES_PER_BLOCK, dtype=np.int8),
            np.ones(PATHLINES_PER_BLOCK, dtype=np.int8),
        )
    )
    return CombinedCenterScene(
        dataset=dataset,
        title=str(title),
        source_index=source_index,
        bounds=np.asarray(legacy["bounds"], dtype=np.float64),
        seeds=np.asarray(seed_lookup[selected_centers], dtype=np.float64),
        reference=np.asarray(reference_lookup[selected_centers], dtype=np.bool_),
        prediction=np.asarray(prediction[selected_centers], dtype=np.bool_),
        center_seed_index=selected_centers,
        paired_score=np.asarray(score[selected_centers], dtype=np.float64),
        legacy_valid=np.asarray(legacy_valid[selected_centers], dtype=np.bool_),
        expanded_valid=np.asarray(expanded_valid[selected_centers], dtype=np.bool_),
        display_pathlines=np.asarray(pathlines),
        display_pathline_block_index=pathline_blocks,
        ivd_mesh_vertices=np.asarray(legacy["ivd_mesh_vertices"]),
        ivd_mesh_faces=np.asarray(legacy["ivd_mesh_faces"]),
        ivd_mesh_normals=np.asarray(legacy["ivd_mesh_normals"]),
        ivd_mesh_values=np.asarray(legacy["ivd_mesh_values"]),
        ivd_mesh_level=np.asarray(legacy["ivd_mesh_level"]),
    )


def bind_valid_projection(
    *,
    legacy_metadata: Mapping[str, Any],
    legacy_arrays: Mapping[str, np.ndarray],
    expanded_metadata: Mapping[str, Any],
    expanded_arrays: Mapping[str, np.ndarray],
    unique_prediction: Mapping[str, np.ndarray],
    valid_prediction: Mapping[str, np.ndarray],
) -> ValidProjection:
    """Join the primary projected prediction to both complete valid-row scenes."""

    legacy = _parent_block_projection(
        legacy_metadata, legacy_arrays, block="legacy_2_1"
    )
    expanded = _parent_block_projection(
        expanded_metadata, expanded_arrays, block="expanded_3_1"
    )
    _require_equal_parent_backgrounds(legacy, expanded)
    unique_required = (
        "unique_dataset",
        "unique_source_ordinal",
        "unique_source_index",
        "unique_center_seed_index",
        "paired_score",
        "paired_prediction",
    )
    _require(
        all(name in unique_prediction for name in unique_required),
        "unique prediction arrays are incomplete",
    )
    _require(
        all(
            np.asarray(unique_prediction[name]).shape == (CENTER_COUNT,)
            for name in unique_required
        ),
        "unique prediction arrays do not share the complete center population",
    )
    unique_centers = np.asarray(
        unique_prediction["unique_center_seed_index"], dtype=np.int64
    )
    _require(
        np.array_equal(unique_centers, np.arange(CENTER_COUNT, dtype=np.int64)),
        "unique prediction centers are not ascending 0..63999",
    )
    required = (
        "valid_dataset",
        "valid_source_ordinal",
        "valid_source_index",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
        "valid_paired_score",
        "valid_paired_prediction",
    )
    _require(all(name in valid_prediction for name in required), "valid prediction arrays are incomplete")
    count = len(np.asarray(valid_prediction["valid_assigned_row_index"]))
    _require(
        all(np.asarray(valid_prediction[name]).shape == (count,) for name in required),
        "valid prediction arrays do not share one row population",
    )
    identity_to_position: dict[tuple[int, int, int, int], int] = {}
    for position, values in enumerate(
        zip(
            np.asarray(valid_prediction["valid_scale_block_index"], dtype=np.int8),
            np.asarray(valid_prediction["valid_center_seed_index"], dtype=np.int64),
            np.asarray(valid_prediction["valid_assigned_row_index"], dtype=np.int64),
            np.asarray(valid_prediction["valid_scale_id"], dtype=np.int32),
            strict=True,
        )
    ):
        key = tuple(int(value) for value in values)
        _require(key not in identity_to_position, "duplicate valid prediction identity")
        identity_to_position[key] = position
    dataset = str(legacy["dataset"])
    source_index = int(legacy["source_index"])
    _require(
        np.all(np.asarray(unique_prediction["unique_dataset"]) == dataset)
        and np.all(
            np.asarray(unique_prediction["unique_source_ordinal"]) == SOURCE_ORDINAL
        )
        and np.all(
            np.asarray(unique_prediction["unique_source_index"]) == source_index
        ),
        "unique prediction dataset/source identity differs from parent scenes",
    )
    _require(
        np.all(np.asarray(valid_prediction["valid_dataset"]) == dataset)
        and np.all(np.asarray(valid_prediction["valid_source_ordinal"]) == SOURCE_ORDINAL)
        and np.all(np.asarray(valid_prediction["valid_source_index"]) == source_index),
        "valid prediction dataset/source identity differs from parent scenes",
    )
    valid_centers = np.asarray(
        valid_prediction["valid_center_seed_index"], dtype=np.int64
    )
    _require(
        np.all((valid_centers >= 0) & (valid_centers < CENTER_COUNT)),
        "valid prediction contains an out-of-range unique center",
    )
    _require(
        np.array_equal(
            np.asarray(valid_prediction["valid_paired_prediction"], dtype=np.bool_),
            np.asarray(unique_prediction["paired_prediction"], dtype=np.bool_)[
                valid_centers
            ],
        ),
        "valid_paired_prediction is not the exact unique-center projection",
    )
    _require(
        np.array_equal(
            np.asarray(valid_prediction["valid_paired_score"], dtype=np.float64),
            np.asarray(unique_prediction["paired_score"], dtype=np.float64)[
                valid_centers
            ],
        ),
        "valid_paired_score is not the exact unique-center projection",
    )
    positions: list[int] = []
    references: list[np.ndarray] = []
    datasets: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    scale_ids: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    block_indices: list[np.ndarray] = []
    assigned_rows: list[np.ndarray] = []
    for block in (legacy, expanded):
        block_count = len(np.asarray(block["center"]))
        references.append(np.asarray(block["reference"], dtype=np.bool_))
        datasets.append(np.full(block_count, dataset, dtype="<U64"))
        ordinals.append(np.full(block_count, SOURCE_ORDINAL, dtype=np.int16))
        source_indices.append(np.full(block_count, source_index, dtype=np.int64))
        scale_ids.append(np.asarray(block["scale_id"], dtype=np.int32))
        centers.append(np.asarray(block["center"], dtype=np.int64))
        block_indices.append(np.asarray(block["scale_block"], dtype=np.int8))
        assigned_rows.append(np.asarray(block["assigned"], dtype=np.int64))
        for values in zip(
            block_indices[-1], centers[-1], assigned_rows[-1], scale_ids[-1], strict=True
        ):
            key = tuple(int(value) for value in values)
            _require(key in identity_to_position, "parent valid row is missing from prediction")
            positions.append(identity_to_position.pop(key))
    _require(not identity_to_position, "prediction has extra valid rows outside parent scenes")
    ordered = np.asarray(positions, dtype=np.int64)
    return ValidProjection(
        reference=np.concatenate(references),
        prediction=np.asarray(valid_prediction["valid_paired_prediction"], dtype=np.bool_)[ordered],
        score=np.asarray(valid_prediction["valid_paired_score"], dtype=np.float64)[ordered],
        dataset=np.concatenate(datasets),
        source_ordinal=np.concatenate(ordinals),
        source_index=np.concatenate(source_indices),
        scale_id=np.concatenate(scale_ids),
        center_seed_index=np.concatenate(centers),
        scale_block_index=np.concatenate(block_indices),
        assigned_row_index=np.concatenate(assigned_rows),
    )


def scene_arrays(scene: CombinedCenterScene, metadata_json: str) -> dict[str, np.ndarray]:
    """Return the exact ordered, pickle-free combined-scene payload."""

    arrays = {
        "bounds": np.asarray(scene.bounds, dtype=np.float64),
        "seeds": np.asarray(scene.seeds, dtype=np.float64),
        "reference": np.asarray(scene.reference, dtype=np.bool_),
        "prediction": np.asarray(scene.prediction, dtype=np.bool_),
        "center_seed_index": np.asarray(scene.center_seed_index, dtype=np.int64),
        "paired_score": np.asarray(scene.paired_score, dtype=np.float64),
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
    _require(tuple(arrays) == SCENE_ARRAY_NAMES, "combined scene array order changed")
    _require(arrays["metadata_json"].dtype.kind == "U", "scene metadata must be Unicode")
    return arrays


def _draw_block_pathlines(
    axis: Any,
    pathlines: Sequence[np.ndarray],
    block_indices: np.ndarray,
) -> dict[str, Any]:
    _require(
        len(pathlines) == len(block_indices) == 2 * PATHLINES_PER_BLOCK,
        "rendered pathline population changed",
    )
    counts: dict[str, int] = {}
    for block_index, style in BLOCK_STYLE.items():
        mask = np.asarray(block_indices) == block_index
        _require(int(mask.sum()) == PATHLINES_PER_BLOCK, "pathline block count changed")
        counts[style["name"]] = int(mask.sum())
        for pathline in np.asarray(pathlines)[mask]:
            xyz = np.asarray(pathline[:, :3], dtype=np.float64)
            segments = np.stack((xyz[:-1], xyz[1:]), axis=1)
            collection = Line3DCollection(
                segments,
                colors=style["color"],
                linewidths=0.60,
                linestyles=style["linestyle"],
                alpha=0.62,
                zorder=3,
                rasterized=True,
            )
            axis.add_collection3d(collection)
            axis.scatter(
                xyz[0, 0],
                xyz[0, 1],
                xyz[0, 2],
                color=style["color"],
                s=4.0,
                alpha=0.72,
                depthshade=False,
                zorder=4,
                rasterized=True,
            )
    return {
        "pathline_count": int(len(pathlines)),
        "block_counts": counts,
        "selection": "first 120 immutable parent-scene pathlines per block",
        "block_styles": BLOCK_STYLE,
    }


def _block_legend_handles() -> list[Line2D]:
    """Return the block-context key for one figure-level safe-area legend."""

    return [
        Line2D(
            [0],
            [0],
            color=BLOCK_STYLE[index]["color"],
            linestyle=BLOCK_STYLE[index]["linestyle"],
            linewidth=1.2,
            label=f"{BLOCK_STYLE[index]['name']} first 120",
        )
        for index in (0, 1)
    ]


def render_source_centered_triptych(
    scene: CombinedCenterScene,
    *,
    png_path: str | Path,
    pdf_path: str | Path,
    svg_path: str | Path,
    alignment_path: str | Path,
    view: tuple[float, float] = DEFAULT_VIEW,
    dpi: int = 360,
) -> dict[str, Any]:
    """Render one audited paired-center triptych without changing the scene."""

    requested = tuple(Path(value) for value in (png_path, pdf_path, svg_path, alignment_path))
    expected_suffixes = (".png", ".pdf", ".svg", ".json")
    _require(
        all(path.suffix.lower() == suffix for path, suffix in zip(requested, expected_suffixes, strict=True)),
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
        "regime": "family-held-out exposed-development paired-center",
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
                    (
                        position.x0,
                        position.y0,
                        position.width,
                        HEADER_AXES_TOP - position.y0,
                    )
                )
            ivd_audits = [_draw_ivd_background(axis, validated) for axis in axes]
            _require(
                all(audit == ivd_audits[0] for audit in ivd_audits[1:]),
                "all three panels must draw the same IVD background",
            )
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
                f"{scene.title} | paired legacy/expanded center prediction | "
                f"source ordinal {SOURCE_ORDINAL} | exposed development",
                ha="center",
                va="top",
                fontsize=9,
            )
            signatures = [_camera_signature(axis) for axis in axes]
            _require(
                all(np.array_equal(signatures[0], value) for value in signatures[1:]),
                "all three panels must use identical camera and bounds",
            )
            alignment = _axes_alignment_audit(figure, axes)
            figure.savefig(requested[0], dpi=360, facecolor="white", edgecolor="none")
            figure.savefig(requested[1], facecolor="white", edgecolor="none")
            figure.savefig(requested[2], facecolor="white", edgecolor="none")
        finally:
            plt.close(figure)
    _write_json_without_overwrite(requested[3], alignment)
    confusion_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    camera = {
        "projection": "orthographic",
        "elevation_degrees": float(view_array[0]),
        "azimuth_degrees": float(view_array[1]),
        "physical_bounds": validated.bounds.tolist(),
        "box_aspect": (validated.bounds[1] - validated.bounds[0]).tolist(),
        "identical_across_panels": True,
    }
    return {
        "schema": "pathline_template_matching.source_centered_paired_scale_triptych.v1",
        "dataset": scene.dataset,
        "title": scene.title,
        "source_ordinal": SOURCE_ORDINAL,
        "source_index": scene.source_index,
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "dpi": int(dpi),
        "panel_order": list(PANEL_TITLES),
        "prediction_semantics": (
            "one authenticated source-centered paired-scale template prediction "
            "per combined-valid unique center"
        ),
        "counts": {
            "combined_valid_center_count": int(len(scene.center_seed_index)),
            "reference_positive": int(scene.reference.sum()),
            "reference_negative": int((~scene.reference).sum()),
            "prediction_positive": int(scene.prediction.sum()),
            "prediction_negative": int((~scene.prediction).sum()),
            **confusion_counts,
        },
        "camera": camera,
        "alignment_audit": {**alignment, "path": str(requested[3])},
        "ivd_audit": {
            "identical_across_panels": True,
            "panels": ivd_audits,
        },
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
            "pathline_blocks": BLOCK_STYLE,
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
            "paired_score": canonical_array_sha256(scene.paired_score),
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
    "CombinedCenterScene",
    "PANEL_TITLES",
    "PATHLINES_PER_BLOCK",
    "SCENE_ARRAY_NAMES",
    "SCENE_SCHEMA",
    "SOURCE_ORDINAL",
    "ValidProjection",
    "bind_valid_projection",
    "combine_parent_block_scenes",
    "render_source_centered_triptych",
    "scene_arrays",
]
