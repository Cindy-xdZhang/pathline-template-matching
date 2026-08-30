from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.portable_flow import canonical_json_sha256, sha256_file
from scripts import aggregate_verify_scale_conditioned_retrieval_1_1 as aggregate_module
from scripts.run_verify_scale_conditioned_retrieval_1_1 import EXPERIMENT, load_plan


CONFIG_PATH = ROOT / "config" / "Verify_ScaleConditionedRetrieval_1.1.yaml"
SYNTHETIC_COMMIT = "1" * 40


def _expect_value_error(function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _outer_summary(f1: float) -> dict[str, float | int]:
    return {
        "accuracy": 0.88,
        "average_precision": 0.66,
        "auroc": 0.76,
        "precision": 0.67,
        "recall": 0.73,
        "f1": f1,
        "balanced_accuracy": 0.74,
        "retrieval_support_fraction": 0.90,
        "spatial_imputed_fraction": 0.08,
        "spatial_unimputable_fraction": 0.02,
        "sample_count": 100,
        "positive_count": 20,
        "negative_count": 80,
        "supported_count": 90,
        "imputed_count": 8,
        "unimputable_count": 2,
        "true_positive": 14,
        "false_positive": 7,
        "true_negative": 73,
        "false_negative": 6,
    }


def _write_fold(
    root: Path,
    *,
    ordinal: int,
    family: str,
    config_sha256: str,
    f1: float,
    git_commit: str = SYNTHETIC_COMMIT,
) -> Path:
    fold = root / f"fold_{ordinal}_{family}"
    fold.mkdir()
    payload_path = fold / "outer_metrics.json"
    payload_path.write_bytes(f"synthetic metrics for {family}\n".encode("utf-8"))
    result = {
        "schema": "pathline_template_matching.scale_conditioned_outer_result.v1",
        "experiment": EXPERIMENT,
        "status": "completed",
        "config_sha256": config_sha256,
        "outer_family": family,
        "git_commit": git_commit,
        "selected_candidate": {"candidate_id": f"candidate_{family}"},
        "outer_summary": _outer_summary(f1),
        "artifacts": {
            payload_path.name: {
                "size_bytes": payload_path.stat().st_size,
                "sha256": sha256_file(payload_path),
            }
        },
    }
    result["manifest_content_sha256"] = canonical_json_sha256(result)
    result_path = fold / "result_manifest.json"
    _write_json(result_path, result)
    completion = {
        "schema": "pathline_template_matching.run_complete.v1",
        "experiment": EXPERIMENT,
        "config_sha256": config_sha256,
        "outer_family": family,
        "git_commit": git_commit,
        "result_manifest_file_sha256": sha256_file(result_path),
    }
    completion["completion_content_sha256"] = canonical_json_sha256(completion)
    _write_json(fold / "RUN_COMPLETE.json", completion)
    return fold


def _make_folds(
    root: Path,
    *,
    families: tuple[str, ...] | None = None,
    commits: tuple[str, ...] | None = None,
) -> tuple[Path, ...]:
    plan = load_plan(CONFIG_PATH)
    fold_families = families or plan.family_order
    fold_commits = commits or (SYNTHETIC_COMMIT,) * 5
    f1_values = (0.80, 0.75, 0.70, 0.65, 0.60)
    return tuple(
        _write_fold(
            root,
            ordinal=ordinal,
            family=family,
            config_sha256=plan.sha256,
            f1=f1,
            git_commit=commit,
        )
        for ordinal, (family, f1, commit) in enumerate(
            zip(fold_families, f1_values, fold_commits)
        )
    )


def _aggregate(run_directories: tuple[Path, ...], output_dir: Path):
    with patch.object(
        aggregate_module,
        "_git_identity",
        return_value=(SYNTHETIC_COMMIT, False),
    ):
        return aggregate_module.aggregate(CONFIG_PATH, run_directories, output_dir)


def test_scale_conditioned_aggregate_writes_five_family_macro_and_passes_stop_rule():
    plan = load_plan(CONFIG_PATH)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary = _aggregate(_make_folds(root), root / "aggregate")

        assert summary["outer_families"] == list(plan.family_order)
        assert summary["outer_family_count"] == 5
        np.testing.assert_allclose(summary["family_macro"]["f1"], 0.70)
        np.testing.assert_allclose(summary["family_macro"]["accuracy"], 0.88)
        assert all(summary["success_stop_rule"].values())
        assert summary["all_success_conditions_pass"] is True
        assert (root / "aggregate" / "outer_family_summary.csv").is_file()
        assert (root / "aggregate" / "aggregate_summary.json").is_file()
        assert (root / "aggregate" / "aggregate_manifest.json").is_file()
        assert (root / "aggregate" / "AGGREGATE_COMPLETE.json").is_file()


def test_scale_conditioned_aggregate_rejects_duplicate_outer_family():
    plan = load_plan(CONFIG_PATH)
    duplicated = (*plan.family_order[:-1], plan.family_order[0])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root, families=duplicated)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="outer family set is incomplete",
        )


def test_scale_conditioned_aggregate_rejects_mixed_fold_commits():
    commits = (SYNTHETIC_COMMIT,) * 4 + ("2" * 40,)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root, commits=commits)
        _expect_value_error(
            _aggregate,
            folds,
            root / "aggregate",
            contains="one numerical Git commit",
        )


def test_scale_conditioned_aggregate_rejects_corrupt_result_completion_and_artifact():
    for corruption in ("result", "completion", "artifact"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = _make_folds(root)
            target = folds[0]
            if corruption == "result":
                with (target / "result_manifest.json").open("ab") as stream:
                    stream.write(b"\n")
                expected = "result manifest hash mismatch"
            elif corruption == "completion":
                completion_path = target / "RUN_COMPLETE.json"
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completion["result_manifest_file_sha256"] = "0" * 64
                _write_json(completion_path, completion)
                expected = "completion content hash mismatch"
            else:
                with (target / "outer_metrics.json").open("ab") as stream:
                    stream.write(b"damage")
                expected = "size drift outer_metrics.json"
            try:
                _expect_value_error(
                    _aggregate,
                    folds,
                    root / "aggregate",
                    contains=expected,
                )
            except AssertionError as error:
                raise AssertionError(f"corruption mode {corruption}: {error}") from error


def test_scale_conditioned_aggregate_refuses_to_overwrite_output_directory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        folds = _make_folds(root)
        output = root / "aggregate"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("do not overwrite", encoding="utf-8")
        _expect_value_error(
            _aggregate,
            folds,
            output,
            contains="immutable output directory exists",
        )
        assert marker.read_text(encoding="utf-8") == "do not overwrite"
        assert tuple(output.iterdir()) == (marker,)
