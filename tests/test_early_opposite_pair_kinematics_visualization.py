from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)


CONFIG = ROOT / "config/Other_EarlyOppositePairKinematicsVisualization_1.1.yaml"
CONFIG_SHA256 = "0b5053cdd2342fcd65950b82f08b520de4c8a2717c44ad15a5d13babd0caf1c8"
SCRIPT = ROOT / "scripts/render_early_opposite_pair_kinematics_visualizations.py"

_SPEC = importlib.util.spec_from_file_location(
    "render_early_opposite_pair_kinematics_visualizations_for_test",
    SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = RENDERER
_SPEC.loader.exec_module(RENDERER)


EXPECTED_ARRAYS = (
    "dataset",
    "source_ordinal",
    "source_index",
    "scale_id",
    "center_seed_index",
    "scale_block_index",
    "assigned_row_index",
    "raw_negative_distance",
    "tail_probability",
    "tail_anomaly",
    "spatial_score",
    "spatial_denominator",
    "retrieval_supported",
    "calibration_supported",
    "spatial_imputed",
    "spatial_unimputable",
    "calibration_mode",
    "scaler_mode",
    "prediction",
)
EXPECTED_CANDIDATES = {
    "half_cylinder": {
        "candidate_id": (
            "representation=chirality_all35_plus_seed4|k=31|sigma=0.5|"
            "fixed_top_fraction=0.05"
        ),
        "representation": "chirality_all35_plus_seed4",
        "k": 31,
        "sigma": 0.5,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    },
    "boeing_747": {
        "candidate_id": (
            "representation=real_neighbor36_plus_seed4|k=31|sigma=0.5|"
            "fixed_top_fraction=0.05"
        ),
        "representation": "real_neighbor36_plus_seed4",
        "k": 31,
        "sigma": 0.5,
        "decision_rule": "fixed_top_fraction",
        "decision_value": 0.05,
    },
}
FOLD_DATASETS = {
    "half_cylinder": (
        "cylinder3d",
        "halfcylinderRe640",
        "halfcylinderRe6400",
    ),
    "boeing_747": ("boeing747",),
}


def _write_self_hashed_json(path: Path, payload: dict[str, object]) -> dict[str, object]:
    value = copy.deepcopy(payload)
    assert "content_sha256" not in value
    value["content_sha256"] = canonical_json_sha256(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def _synthetic_arrays(datasets: tuple[str, ...]) -> dict[str, np.ndarray]:
    rows = [
        (dataset, dataset_index, source, block, local_index)
        for dataset_index, dataset in enumerate(datasets)
        for source in range(4)
        for block in range(2)
        for local_index in range(2)
    ]
    count = len(rows)
    dataset = np.asarray([row[0] for row in rows], dtype="<U64")
    source = np.asarray([row[2] for row in rows], dtype="<i2")
    block = np.asarray([row[3] for row in rows], dtype="|i1")
    center = np.asarray(
        [row[1] * 100 + row[2] * 10 + row[4] for row in rows],
        dtype="<i8",
    )
    scale = block.astype("<i4") * 1000 + np.asarray(
        [row[4] for row in rows], dtype="<i4"
    )
    zeros_f4 = np.zeros(count, dtype="<f4")
    zeros_f8 = np.zeros(count, dtype="<f8")
    true = np.ones(count, dtype="|b1")
    false = np.zeros(count, dtype="|b1")
    return {
        "dataset": dataset,
        "source_ordinal": source,
        "source_index": source.astype("<i8") * 10,
        "scale_id": scale,
        "center_seed_index": center,
        "scale_block_index": block,
        "assigned_row_index": center + block.astype("<i8") * 64_000,
        "raw_negative_distance": zeros_f4,
        "tail_probability": zeros_f8,
        "tail_anomaly": zeros_f8,
        "spatial_score": np.linspace(0.0, 1.0, count, dtype="<f8"),
        "spatial_denominator": np.ones(count, dtype="<f8"),
        "retrieval_supported": true,
        "calibration_supported": true,
        "spatial_imputed": false,
        "spatial_unimputable": false,
        "calibration_mode": np.ones(count, dtype="|i1"),
        "scaler_mode": np.ones(count, dtype="|i1"),
        "prediction": np.arange(count, dtype="<i8").astype("|b1"),
    }


def _write_synthetic_fold(
    root: Path,
    *,
    family: str = "half_cylinder",
    candidate: dict[str, object] | None = None,
    prediction_manifest_schema: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    frozen_candidate = copy.deepcopy(
        EXPECTED_CANDIDATES[family] if candidate is None else candidate
    )
    arrays = _synthetic_arrays(FOLD_DATASETS[family])
    assert tuple(arrays) == EXPECTED_ARRAYS
    prediction_path = root / "outer_predictions.npz"
    np.savez_compressed(
        prediction_path,
        **{name: arrays[name] for name in EXPECTED_ARRAYS},
    )

    selected = _write_self_hashed_json(
        root / "selected_candidate.json",
        {
            "schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_selected_candidate.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "git_commit": RENDERER.PREDICTION_COMMIT,
            "config_sha256": RENDERER.PREDICTION_CONFIG_SHA256,
            "outer_family": family,
            "candidate_count": 3060,
            "candidate": frozen_candidate,
        },
    )
    prediction_manifest = _write_self_hashed_json(
        root / "outer_prediction_manifest.json",
        {
            "schema": prediction_manifest_schema
            or (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_outer_prediction_manifest.v1"
            ),
            "prediction_schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_outer_prediction.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "git_commit": RENDERER.PREDICTION_COMMIT,
            "config_sha256": RENDERER.PREDICTION_CONFIG_SHA256,
            "outer_family": family,
            "selected_candidate": frozen_candidate,
            "selected_candidate_artifact": {
                "path": "selected_candidate.json",
                "file_sha256": sha256_file(root / "selected_candidate.json"),
                "content_sha256": selected["content_sha256"],
            },
            "valid_labels_opened": False,
            "metadata_json_opened": False,
            "array_count": len(EXPECTED_ARRAYS),
            "row_count": len(arrays["dataset"]),
            "arrays": {
                name: {
                    "dtype": arrays[name].dtype.str,
                    "shape": list(arrays[name].shape),
                    "sha256": canonical_array_sha256(arrays[name]),
                }
                for name in EXPECTED_ARRAYS
            },
            "prediction_file": {
                "path": "outer_predictions.npz",
                "size_bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
            },
        },
    )

    (root / "outer_group_metrics.csv").write_text(
        "outer_family,dataset,source_ordinal,block\n",
        encoding="utf-8",
    )
    _write_self_hashed_json(
        root / "outer_summary.json",
        {
            "schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_outer_summary.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "outer_family": family,
        },
    )
    _write_self_hashed_json(
        root / "outer_reference_access_audit.json",
        {
            "schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_outer_reference_access.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "outer_family": family,
        },
    )

    artifact_names = (
        "outer_predictions.npz",
        "outer_prediction_manifest.json",
        "outer_group_metrics.csv",
        "outer_reference_access_audit.json",
        "outer_summary.json",
        "selected_candidate.json",
    )
    artifacts = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in artifact_names
    }
    result = _write_self_hashed_json(
        root / "result_manifest.json",
        {
            "schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_result.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "status": "completed",
            "git_commit": RENDERER.PREDICTION_COMMIT,
            "config_sha256": RENDERER.PREDICTION_CONFIG_SHA256,
            "outer_family": family,
            "selected_candidate": frozen_candidate,
            "selected_candidate_file_sha256": artifacts[
                "selected_candidate.json"
            ]["sha256"],
            "selected_candidate_content_sha256": selected["content_sha256"],
            "prediction_manifest_file_sha256": artifacts[
                "outer_prediction_manifest.json"
            ]["sha256"],
            "prediction_file_sha256": artifacts["outer_predictions.npz"]["sha256"],
            "artifacts": artifacts,
        },
    )
    _write_self_hashed_json(
        root / "RUN_COMPLETE.json",
        {
            "schema": (
                "pathline_template_matching."
                "early_opposite_pair_kinematics_run_complete.v1"
            ),
            "experiment": RENDERER.PREDICTION_EXPERIMENT,
            "git_commit": RENDERER.PREDICTION_COMMIT,
            "config_sha256": RENDERER.PREDICTION_CONFIG_SHA256,
            "outer_family": family,
            "result_manifest_file": "result_manifest.json",
            "result_manifest_file_sha256": sha256_file(
                root / "result_manifest.json"
            ),
            "result_manifest_content_sha256": result["content_sha256"],
        },
    )
    assert prediction_manifest["array_count"] == 19


def test_frozen_config_arrays_candidates_eight_figures_and_classification_contract():
    config = RENDERER._authenticate_report_config(CONFIG)
    assert sha256_file(CONFIG) == CONFIG_SHA256
    assert config["experiment"] == RENDERER.REPORT_EXPERIMENT
    assert config["status"] == "frozen_pre_run_not_run"
    assert RENDERER.PREDICTION_COMMIT == "2c3774dca0d81db8edd5645e63576526b9e276f7"
    assert (
        RENDERER.PREDICTION_CONFIG_SHA256
        == "e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767"
    )
    prediction_contract = config["prediction_contract"]
    assert prediction_contract["schema"].endswith(
        "early_opposite_pair_kinematics_outer_prediction.v1"
    )
    assert prediction_contract["manifest_schema"].endswith(
        "early_opposite_pair_kinematics_outer_prediction_manifest.v1"
    )
    assert prediction_contract["result_schema"].endswith(
        "early_opposite_pair_kinematics_result.v1"
    )
    assert prediction_contract["completion_schema"].endswith(
        "early_opposite_pair_kinematics_run_complete.v1"
    )
    assert prediction_contract["complete_array_count"] == 19
    assert tuple(prediction_contract["ordered_array_names"]) == EXPECTED_ARRAYS
    assert tuple(RENDERER.PREDICTION_ARRAY_NAMES) == EXPECTED_ARRAYS

    folds = config["parents"]["early_opposite_pair_folds"]["folds"]
    assert {fold["outer_family"]: fold["candidate"] for fold in folds} == EXPECTED_CANDIDATES
    assert (
        len(config["query"]["datasets"]) * len(config["query"]["scale_blocks"])
        == config["figure_contract"]["expected_figure_count"]
        == 8
    )
    panel_b = " ".join(
        (
            config["figure_contract"]["panel_titles"][1],
            config["figure_contract"]["panel_map"]["b"],
            RENDERER._figure_contract()["panel_map"]["b"],
        )
    ).lower()
    assert "classification" in panel_b
    assert "cluster" not in panel_b
    assert "clustering" in config["evidence_scope"]["forbidden_claims"]


def test_frozen_reporting_config_rejects_any_byte_drift():
    with tempfile.TemporaryDirectory() as directory:
        changed = Path(directory) / CONFIG.name
        changed.write_text(
            CONFIG.read_text(encoding="utf-8").replace(
                "purpose: audited_", "purpose: changed_audited_", 1
            ),
            encoding="utf-8",
        )
        try:
            RENDERER._authenticate_report_config(changed)
        except ValueError as error:
            assert "config" in str(error).lower() and "sha" in str(error).lower()
        else:
            raise AssertionError("modified frozen visualization config was accepted")


def test_synthetic_fold_chain_and_complete_19_array_projection_authenticate():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "half_cylinder"
        _write_synthetic_fold(root)
        authenticated, evidence = RENDERER._authenticate_fold(root, "half_cylinder")
        assert len({row["path"] for row in evidence}) == len(evidence)
        groups = RENDERER._load_prediction_groups(authenticated)
        assert set(groups) == {
            (dataset, block)
            for dataset in FOLD_DATASETS["half_cylinder"]
            for block in ("legacy_2_1", "expanded_3_1")
        }
        assert all(group.candidate == EXPECTED_CANDIDATES["half_cylinder"] for group in groups.values())


def test_fold_authentication_rejects_candidate_schema_and_array_tamper():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)

        wrong_candidate = copy.deepcopy(EXPECTED_CANDIDATES["half_cylinder"])
        wrong_candidate["k"] = 15
        wrong_candidate["candidate_id"] = str(wrong_candidate["candidate_id"]).replace(
            "k=31", "k=15"
        )
        candidate_root = base / "candidate"
        _write_synthetic_fold(candidate_root, candidate=wrong_candidate)
        try:
            RENDERER._authenticate_fold(candidate_root, "half_cylinder")
        except ValueError as error:
            assert "candidate" in str(error).lower()
        else:
            raise AssertionError("non-frozen Early candidate was accepted")

        schema_root = base / "schema"
        _write_synthetic_fold(
            schema_root,
            prediction_manifest_schema="pathline_template_matching.changed.v1",
        )
        try:
            RENDERER._authenticate_fold(schema_root, "half_cylinder")
        except ValueError as error:
            assert "schema" in str(error).lower()
        else:
            raise AssertionError("changed Early prediction schema was accepted")

        tamper_root = base / "tamper"
        _write_synthetic_fold(tamper_root)
        authenticated, _ = RENDERER._authenticate_fold(tamper_root, "half_cylinder")
        with np.load(tamper_root / "outer_predictions.npz", allow_pickle=False) as archive:
            changed_arrays = {name: np.asarray(archive[name]).copy() for name in EXPECTED_ARRAYS}
        changed_arrays["spatial_score"][0] += 1.0
        np.savez_compressed(
            tamper_root / "outer_predictions.npz",
            **{name: changed_arrays[name] for name in EXPECTED_ARRAYS},
        )
        try:
            RENDERER._load_prediction_groups(authenticated)
        except ValueError as error:
            message = str(error).lower()
            assert "sha" in message or "hash" in message
        else:
            raise AssertionError("tampered Early prediction array was accepted")


def test_existing_output_directory_fails_before_checkout_or_input_access():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_root = root / "immutable-output"
        output_root.mkdir()
        try:
            RENDERER.render_bundle(
                parent_root=root / "missing-parent",
                half_fold_root=root / "missing-half-fold",
                boeing_fold_root=root / "missing-boeing-fold",
                output_root=output_root,
                dpi=360,
            )
        except FileExistsError as error:
            assert "exist" in str(error).lower()
        else:
            raise AssertionError("an existing immutable output directory was reused")


def _run_zero_argument_tests() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"early_opposite_pair_kinematics_visualization_tests={len(tests)} PASS")
    return len(tests)


if __name__ == "__main__":
    _run_zero_argument_tests()
