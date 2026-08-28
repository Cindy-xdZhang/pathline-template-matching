from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace

import numpy as np

from pathline_template_matching.phase21_visualization import (
    DISPLAY_PATHLINE_COUNT,
    DISPLAY_PER_CLASS,
    build_phase21_visualization_scene,
    load_phase21_scene_artifact,
    ordered_fmt_prediction,
    render_phase21_scene_artifact,
    scale_axis_indices,
    select_display_query_rows,
    write_phase21_scene_artifact,
)
from pathline_template_matching.portable_flow import canonical_array_sha256, sha256_file
import pathline_template_matching.phase21_pipeline as phase21_pipeline


def _cache_and_prediction(*, invert_prediction: bool = False):
    count = 260
    rng = np.random.Generator(np.random.PCG64(92741))
    labels = np.zeros(count, dtype=np.bool_)
    labels[count // 2 :] = True
    seeds = rng.uniform(0.03, 0.90, size=(count, 3)).astype(np.float64)
    seeds[labels, 0] = rng.uniform(0.96, 0.99, size=int(labels.sum()))
    seeds[~labels, 0] = rng.uniform(0.03, 0.90, size=int((~labels).sum()))
    # Make every row identity unambiguous while staying inside the domain.
    seeds[:, 1] = (np.arange(count) % 26 + 0.5) / 26.0
    seeds[:, 2] = (np.arange(count) // 26 + 0.5) / 10.0
    raw = np.zeros((count, 672), dtype=np.float32)
    source_time = 4.25
    center_time = np.broadcast_to(
        np.linspace(0.0, 0.3, 32, dtype=np.float32),
        (count, 32),
    ).copy()
    valid_seed_index = np.arange(count, dtype=np.int64)
    valid_scale_id = ((np.arange(count, dtype=np.int64) * 137) % 1000).astype(
        np.int32
    )
    x = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    ivd_volume = np.broadcast_to(x[None, None, :], (21, 21, 21)).copy()
    threshold = float(np.percentile(ivd_volume, 95.0))
    cache = {
        "raw_features": raw,
        "valid_seed_index": valid_seed_index,
        "valid_scale_id": valid_scale_id,
        "valid_labels": labels,
        "seeds_xyz": seeds,
        "center_sample_time": center_time,
        "ivd_volume": ivd_volume,
        "metadata": {
            "schema": "pathline_template_matching.phase21_cache.v2",
            "experiment": "mainExp_TemplateMatching_2.1",
            "dataset": "tangaroa",
            "split": "test",
            "source_ordinal": 2,
            "source_index": 125,
            "source_time": source_time,
            "loaded_shape_TZYXC": [13, 21, 21, 21, 3],
            "spacing_xyz": [0.05, 0.05, 0.05],
            "domain_min_xyz": [0.0, 0.0, 0.0],
            "domain_max_xyz": [1.0, 1.0, 1.0],
            "ivd_percentile": 95.0,
            "ivd_threshold": threshold,
        },
    }
    prediction_labels = (np.arange(count) % 3 == 0)
    if invert_prediction:
        prediction_labels = ~prediction_labels
    prediction = ordered_fmt_prediction(
        prediction_labels, valid_seed_index, valid_scale_id
    )
    return cache, prediction


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_phase21_scale_decode_and_selection_are_balanced_deterministic_and_prediction_free():
    decoded = scale_axis_indices(np.asarray([0, 9, 10, 100, 999], dtype=np.int32))
    np.testing.assert_array_equal(
        decoded,
        np.asarray(
            [[0, 0, 0], [0, 0, 9], [0, 1, 0], [1, 0, 0], [9, 9, 9]],
            dtype=np.int16,
        ),
    )
    cache, prediction_a = _cache_and_prediction(invert_prediction=False)
    _, prediction_b = _cache_and_prediction(invert_prediction=True)
    scene_a, audit_a = build_phase21_visualization_scene(cache, prediction_a)
    scene_b, audit_b = build_phase21_visualization_scene(cache, prediction_b)
    np.testing.assert_array_equal(
        scene_a["selected_query_row"], scene_b["selected_query_row"]
    )
    assert audit_a["selection"] == audit_b["selection"]
    selected = np.asarray(scene_a["selected_query_row"])
    assert selected.shape == (DISPLAY_PATHLINE_COUNT,)
    assert len(np.unique(selected)) == DISPLAY_PATHLINE_COUNT
    assert int(np.asarray(scene_a["selected_reference"]).sum()) == DISPLAY_PER_CLASS
    assert audit_a["selection"]["positive_count"] == DISPLAY_PER_CLASS
    assert audit_a["selection"]["negative_count"] == DISPLAY_PER_CLASS
    assert "FMT_prediction" in audit_a["selection"]["forbidden_dependencies"]
    assert audit_a[
        "all_panels_use_complete_identical_valid_query_seed_population"
    ] is True
    assert len(set(audit_a["panel_query_seed_sha256"].values())) == 1
    assert len(scene_a["seeds"]) == len(cache["valid_labels"])
    assert len(scene_a["reference"]) == len(scene_a["prediction"]) == len(
        scene_a["seeds"]
    )
    pathlines = np.asarray(scene_a["display_pathlines"])
    assert pathlines.shape == (DISPLAY_PATHLINE_COUNT, 32, 4)
    np.testing.assert_allclose(
        pathlines[:, 0, :3],
        np.asarray(scene_a["seeds"])[selected],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(pathlines[:, 0, 3], 4.25, rtol=0.0, atol=1e-12)
    assert audit_a["ivd_mesh"]["complete_loaded_volume"] is True
    assert audit_a["ivd_mesh"]["triangle_subsampling"] is False
    assert audit_a["ivd_mesh"]["mesh_finite"] is True
    assert audit_a["ivd_mesh"]["level_absolute_difference"] == 0.0


def test_phase21_selection_direct_api_is_repeatable_and_fails_if_a_class_is_short():
    cache, _ = _cache_and_prediction()
    rows_a, audit_a = select_display_query_rows(
        cache["seeds_xyz"],
        cache["valid_labels"],
        cache["valid_scale_id"],
        domain_bounds=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
        dataset="tangaroa",
        source_ordinal=2,
    )
    rows_b, audit_b = select_display_query_rows(
        cache["seeds_xyz"],
        cache["valid_labels"],
        cache["valid_scale_id"],
        domain_bounds=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
        dataset="tangaroa",
        source_ordinal=2,
    )
    np.testing.assert_array_equal(rows_a, rows_b)
    assert audit_a == audit_b
    short_labels = np.asarray(cache["valid_labels"]).copy()
    short_labels[:] = False
    short_labels[:119] = True
    try:
        select_display_query_rows(
            cache["seeds_xyz"],
            short_labels,
            cache["valid_scale_id"],
            domain_bounds=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            dataset="tangaroa",
            source_ordinal=2,
        )
    except RuntimeError as error:
        assert "only 119 positive" in str(error)
    else:
        raise AssertionError("selection accepted fewer than 120 positive queries")


def test_phase21_prediction_order_mismatch_and_nonfixed_source_fail_closed():
    cache, prediction = _cache_and_prediction()
    bad_prediction = dict(prediction)
    bad_prediction["valid_seed_index"] = prediction["valid_seed_index"][::-1]
    try:
        build_phase21_visualization_scene(cache, bad_prediction)
    except ValueError as error:
        assert "row order disagrees" in str(error)
    else:
        raise AssertionError("a reordered prediction was accepted")
    wrong_source = dict(cache)
    wrong_source["metadata"] = {**cache["metadata"], "source_ordinal": 1}
    try:
        build_phase21_visualization_scene(wrong_source, prediction)
    except ValueError as error:
        assert "source ordinal 2" in str(error)
    else:
        raise AssertionError("a metric-selectable source ordinal was accepted")


def test_phase21_scene_roundtrip_hashes_every_array_and_refuses_overwrite():
    cache, prediction = _cache_and_prediction()
    scene, audit = build_phase21_visualization_scene(cache, prediction)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = root / "tangaroa_source2.scene.npz"
        manifest_path = root / "tangaroa_source2.scene.json"
        manifest = write_phase21_scene_artifact(
            scene, audit, scene_path, manifest_path
        )
        assert manifest["scene_npz_sha256"] == sha256_file(scene_path)
        assert manifest["scene_manifest_file_sha256"] == sha256_file(manifest_path)
        assert len(manifest["arrays"]) == 16
        assert manifest["arrays"]["seeds"]["canonical_sha256"] == (
            canonical_array_sha256(scene["seeds"])
        )
        loaded = load_phase21_scene_artifact(scene_path, manifest_path)
        assert loaded.npz_sha256 == sha256_file(scene_path)
        assert loaded.manifest_file_sha256 == sha256_file(manifest_path)
        np.testing.assert_array_equal(loaded.scene["seeds"], scene["seeds"])
        np.testing.assert_array_equal(
            loaded.scene["prediction"], scene["prediction"]
        )
        np.testing.assert_array_equal(
            np.asarray(loaded.scene["display_pathlines"]),
            np.asarray(scene["display_pathlines"]),
        )
        try:
            write_phase21_scene_artifact(scene, audit, scene_path, manifest_path)
        except FileExistsError as error:
            assert "immutable" in str(error)
        else:
            raise AssertionError("scene evidence was overwritten")


def test_phase21_render_writes_png_pdf_metadata_counts_and_alignment():
    cache, prediction = _cache_and_prediction()
    scene, audit = build_phase21_visualization_scene(cache, prediction)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = root / "scene.npz"
        manifest_path = root / "scene.json"
        write_phase21_scene_artifact(scene, audit, scene_path, manifest_path)
        rendered = render_phase21_scene_artifact(
            scene_path, manifest_path, root / "triptych", dpi=20
        )
        assert rendered.png_path.is_file()
        assert rendered.pdf_path.is_file()
        assert rendered.metadata_path.is_file()
        assert rendered.alignment_path.is_file()
        assert _png_dimensions(rendered.png_path) == (21 * 20, 5 * 20)
        assert rendered.pdf_path.read_bytes().startswith(b"%PDF-")
        pdf_bytes = rendered.pdf_path.read_bytes()
        assert b"/Font" in pdf_bytes
        assert b"/Subtype /Image" in pdf_bytes
        stored_metadata = json.loads(rendered.metadata_path.read_text(encoding="utf-8"))
        assert stored_metadata["png_sha256"] == sha256_file(rendered.png_path)
        assert stored_metadata["pdf_sha256"] == sha256_file(rendered.pdf_path)
        assert stored_metadata["counts"]["sample_count"] == 260
        assert stored_metadata["counts"]["display_pathline_count"] == 240
        assert sum(
            stored_metadata["counts"][name]
            for name in (
                "true_negative",
                "true_positive",
                "false_positive",
                "false_negative",
            )
        ) == 260
        assert stored_metadata["renderer"]["panel_labels"] == ["a", "b", "c"]
        assert stored_metadata["renderer"]["export_contract"][
            "pdf_editable_text"
        ] is True
        assert stored_metadata["renderer"]["export_contract"][
            "three_dimensional_marks_rasterized"
        ] is True
        alignment = json.loads(rendered.alignment_path.read_text(encoding="utf-8"))
        assert alignment["status"] == "PASS"
        assert alignment["schema_version"] == 1
        assert alignment["backend"] == "python-matplotlib"
        assert alignment["tolerance_points"] == 1.5
        assert len(alignment["rectangles"]) == 3
        assert [panel["id"] for panel in alignment["panels"]] == ["a", "b", "c"]
        assert alignment["row_groups"] == [
            {"id": "triptych-row", "panels": ["a", "b", "c"]}
        ]
        assert alignment["checks"]["equal_width"] is True
        assert alignment["checks"]["equal_height"] is True
        assert alignment["checks"]["uniform_gutters"] is True
        assert alignment["maximum_width_deviation_points"] <= 1.5
        assert alignment["maximum_height_deviation_points"] <= 1.5
        try:
            render_phase21_scene_artifact(
                scene_path, manifest_path, root / "triptych", dpi=20
            )
        except FileExistsError as error:
            assert "overwrite" in str(error)
        else:
            raise AssertionError("render bundle was overwritten")


def test_phase21_pipeline_hook_writes_two_fixed_visualization_bundles_and_manifest():
    tangaroa_cache, tangaroa_prediction = _cache_and_prediction()
    smoke_cache, smoke_prediction = _cache_and_prediction(invert_prediction=True)
    smoke_cache = {
        **smoke_cache,
        "metadata": {
            **smoke_cache["metadata"],
            "dataset": "smokeBuoyancy",
            "source_index": 98,
        },
    }
    count = len(tangaroa_cache["valid_labels"])
    plan = SimpleNamespace(
        config={
            "visualization": {
                "source_ordinal": 2,
                "metric_based_or_prediction_based_scene_selection": "forbidden",
                "png_dpi": 20,
            }
        },
        test_datasets=("tangaroa", "smokeBuoyancy"),
        family_by_dataset={
            "tangaroa": "tangaroa",
            "smokeBuoyancy": "smoke_buoyancy",
        },
        experiment="mainExp_TemplateMatching_2.1",
        config_sha256="1" * 64,
    )
    rows = (
        {
            "dataset": "tangaroa",
            "source_ordinal": 2,
            "source_index": 125,
            "path": "tangaroa.npz",
            "file_sha256": "2" * 64,
        },
        {
            "dataset": "smokeBuoyancy",
            "source_ordinal": 2,
            "source_index": 98,
            "path": "smoke.npz",
            "file_sha256": "3" * 64,
        },
    )
    caches = {
        "tangaroa.npz": tangaroa_cache,
        "smoke.npz": smoke_cache,
    }
    query = {
        "dataset_index": np.concatenate(
            (np.zeros(count, dtype=np.int16), np.ones(count, dtype=np.int16))
        ),
        "source_ordinal": np.full(2 * count, 2, dtype=np.int16),
        "valid_seed_index": np.concatenate(
            (tangaroa_cache["valid_seed_index"], smoke_cache["valid_seed_index"])
        ),
        "scale_id": np.concatenate(
            (tangaroa_cache["valid_scale_id"], smoke_cache["valid_scale_id"])
        ),
        "labels": np.concatenate(
            (tangaroa_cache["valid_labels"], smoke_cache["valid_labels"])
        ),
    }
    fmt_prediction = np.concatenate(
        (tangaroa_prediction["labels"], smoke_prediction["labels"])
    )
    original_loader = phase21_pipeline._load_cache
    original_validator = phase21_pipeline._validate_cache_provenance
    phase21_pipeline._load_cache = lambda path, expected_sha256=None: caches[path.name]
    phase21_pipeline._validate_cache_provenance = lambda *_args, **_kwargs: None
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = phase21_pipeline._build_phase21_visualization_artifacts(
                plan,
                root=root,
                test_rows=rows,
                query=query,
                fmt_prediction=fmt_prediction,
                git_commit="4" * 40,
                verify_cache_hashes=True,
            )
            assert manifest is not None and manifest["entry_count"] == 2
            assert [row["dataset"] for row in manifest["entries"]] == [
                "tangaroa",
                "smokeBuoyancy",
            ]
            stored = json.loads(
                (root / "visualization_manifest.json").read_text(encoding="utf-8")
            )
            assert stored["manifest_content_sha256"] == manifest[
                "manifest_content_sha256"
            ]
            for row in stored["entries"]:
                for field in (
                    "scene_npz",
                    "scene_manifest",
                    "png",
                    "pdf",
                    "render_metadata",
                    "panel_alignment",
                ):
                    assert (root / row[field]).is_file()
    finally:
        phase21_pipeline._load_cache = original_loader
        phase21_pipeline._validate_cache_provenance = original_validator
