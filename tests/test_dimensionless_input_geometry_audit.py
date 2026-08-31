from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import run_other_dimensionless_input_geometry_audit_1_1 as runner  # noqa: E402


CONFIG = ROOT / "config" / "Other_DimensionlessInputGeometryAudit_1.1.yaml"
WRAPPER = ROOT / "ibex" / "other_dimensionless_input_geometry_audit_1.1.sh"


def _expect_error(error_type, fragment: str, function) -> None:
    try:
        function()
    except error_type as error:
        assert fragment.lower() in str(error).lower(), str(error)
        return
    raise AssertionError(f"expected {error_type.__name__}: {fragment}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _producer_raw(seeds: np.ndarray, distances: np.ndarray) -> np.ndarray:
    count = len(seeds)
    absolute = np.empty((count, 7, 32, 3), dtype=np.float64)
    absolute[:, 0] = seeds[:, None, :]
    directions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    for direction_index, direction in enumerate(directions, start=1):
        absolute[:, direction_index] = (
            seeds[:, None, :] + distances[:, None, None] * direction[None, None, :]
        )
    stored = absolute.astype(np.float32)
    centered = stored - stored[:, :1, :1, :]
    return np.ascontiguousarray(centered.reshape(count, 672), dtype=np.float32)


def _fixture_arrays(plan: runner.AuditPlan) -> dict[str, np.ndarray]:
    assigned = np.asarray((0, 1, 64000, 64001), dtype=np.int64)
    centers = np.asarray((0, 1, 0, 1), dtype=np.int64)
    blocks = np.asarray((0, 0, 1, 1), dtype=np.int8)
    scales = np.asarray((0, 1, 1080, 1001), dtype=np.int32)
    selected_seeds = np.asarray(
        (
            (0.25, -0.50, 0.75),
            (-0.40, 0.60, -0.80),
            (-0.8530591565829057, -4.758745901859724, -0.8991659581661224),
            (0.10, 0.20, 0.30),
        ),
        dtype=np.float64,
    )
    distances = np.asarray((0.125, 0.25, 0.004587151110172272, 0.075), dtype=np.float64)
    seeds = np.zeros((plan.assigned_seed_count, 3), dtype=np.float64)
    seeds[assigned] = selected_seeds
    return {
        "raw_features": _producer_raw(selected_seeds, distances),
        "valid_scale_id": scales,
        "valid_center_seed_index": centers,
        "valid_scale_block_index": blocks,
        "valid_assigned_row_index": assigned,
        "seeds_xyz": seeds,
    }


def _dummy_row(path: Path, sha256: str = "a" * 64, size: int = 1) -> runner.CacheRow:
    return runner.CacheRow(
        dataset="cylinder3d",
        physical_family="half_cylinder",
        source_ordinal=0,
        source_index=0,
        path=path,
        size_bytes=size,
        sha256=sha256,
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        valid_labels=np.zeros(len(arrays["raw_features"]), dtype=bool),
        reference_labels_all=np.zeros(1, dtype=bool),
        metadata_json=np.asarray("forbidden-sentinel"),
        fmt_features=np.zeros((1, 161), dtype=np.float32),
        ivd_values_all=np.zeros(1, dtype=np.float32),
        ivd_volume=np.zeros((1, 1, 1), dtype=np.float32),
    )


def _write_manifest_fixture(
    root: Path,
    plan: runner.AuditPlan,
    arrays: dict[str, np.ndarray],
) -> tuple[runner.AuditPlan, tuple[Path, ...]]:
    cache_root = root / "primitive_cache" / "train"
    first = cache_root / plan.dataset_order[0] / "source_000000.npz"
    _write_npz(first, arrays)
    cache_sha = _sha256(first)
    cache_size = first.stat().st_size
    paths: list[Path] = []
    rows: list[dict[str, object]] = []
    for dataset in plan.dataset_order:
        for source_ordinal in range(4):
            path = cache_root / dataset / f"source_{source_ordinal:06d}.npz"
            if path != first:
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(first, path)
            paths.append(path)
            rows.append(
                {
                    "cache_file_sha256": cache_sha,
                    "cache_path": str(path),
                    "cache_size_bytes": cache_size,
                    "dataset": dataset,
                    "sidecar_file_sha256": "b" * 64,
                    "sidecar_path": str(path.with_suffix(".summary.json")),
                    "sidecar_size_bytes": 1,
                    "source_index": source_ordinal * 10,
                    "source_ordinal": source_ordinal,
                }
            )
    rows_sha = runner._canonical_json_sha256(rows)
    manifest = {
        "experiment": "Verify_LongArcHorizon_1.1",
        "git_commit": plan.cache_builder_git_commit,
        "input_scope": plan.manifest_input_scope,
        "main_config_sha256": plan.parent_main_config_sha256,
        "parent_experiment": "mainExp_TemplateMatching_3.1",
        "row_count": 32,
        "rows": rows,
        "rows_content_sha256": rows_sha,
        "schema": runner.MANIFEST_SCHEMA,
        "synthetic_pass_file_sha256": "c" * 64,
        "test_dataset_access": False,
        "train_portable_population_pass": {},
        "verify_config_sha256": "d" * 64,
    }
    manifest_path = root / "train_cache_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fixture_plan = replace(
        plan,
        manifest_path=manifest_path,
        manifest_size_bytes=manifest_path.stat().st_size,
        manifest_sha256=_sha256(manifest_path),
        manifest_rows_sha256=rows_sha,
    )
    return fixture_plan, tuple(paths)


def test_geometry_audit_config_and_wrapper_freeze_label_free_scope() -> None:
    plan = runner.load_plan(CONFIG)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256 == _sha256(CONFIG)
    assert plan.allowed_members == runner.ALLOWED_MEMBERS
    assert plan.forbidden_members == runner.REQUIRED_FORBIDDEN_MEMBERS
    assert not set(plan.allowed_members) & set(plan.forbidden_members)
    assert plan.dataset_order == (
        "cylinder3d",
        "halfcylinderRe640",
        "halfcylinderRe6400",
        "deltaWing_resampled",
        "deltaWing_LBM",
        "f22raptor",
        "channel",
        "boeing747",
    )
    assert plan.forbidden_datasets == ("tangaroa", "smokeBuoyancy")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=128G",
        "#SBATCH --constraint=cpu_amd_epyc_7702",
        "conda activate deepvortex",
        runner.EXPECTED_CONFIG_SHA256,
        "EXPECTED_GIT_COMMIT",
        "python tests/test_dimensionless_input_geometry_audit.py",
        '[[ ! -e "$OUTPUT_DIR" ]]',
    ):
        assert required in wrapper


def test_rounding_envelope_explains_float32_cache_quantization_but_not_true_distortion() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    row = _dummy_row(Path("synthetic.npz"))
    audit = runner.audit_geometry_arrays(plan, row, arrays)
    quantized_row = 2
    assert audit.six_norm_fail[quantized_row]
    assert audit.opposite_pair_fail[quantized_row]
    assert audit.rounding_feasible[quantized_row]
    assert audit.rounding_low[quantized_row] <= 0.004587151110172272
    assert audit.rounding_high[quantized_row] >= 0.004587151110172272

    distorted = {name: np.array(value, copy=True) for name, value in arrays.items()}
    primitive = distorted["raw_features"].reshape(-1, 7, 32, 3)
    primitive[quantized_row, 3, 0, 1] *= np.float32(1.25)
    bad = runner.audit_geometry_arrays(plan, row, distorted)
    assert bad.either_gate_fail[quantized_row]
    assert not bad.rounding_feasible[quantized_row]


def test_scale_wide_common_h_is_required_for_the_quantization_conclusion() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    # Rows 1 and 2 are individually produced by valid float32 pipelines, but
    # their true h values differ by orders of magnitude.  Assigning both to one
    # exact scale must therefore falsify the one-common-h producer hypothesis
    # for that scale, even though each row has a non-empty interval by itself.
    arrays["valid_scale_id"][1] = arrays["valid_scale_id"][2]
    arrays["valid_scale_block_index"][1] = arrays["valid_scale_block_index"][2]
    original_seed = np.array(arrays["seeds_xyz"][1], copy=True)
    arrays["valid_center_seed_index"][1] = 2
    arrays["valid_assigned_row_index"][1] = (
        np.int64(arrays["valid_scale_block_index"][1]) * plan.center_seed_count
        + np.int64(arrays["valid_center_seed_index"][1])
    )
    arrays["seeds_xyz"][arrays["valid_assigned_row_index"][1]] = original_seed
    row = _dummy_row(Path("synthetic.npz"))
    audit = runner.audit_geometry_arrays(plan, row, arrays)
    assert audit.rounding_feasible[1]
    assert audit.rounding_feasible[2]
    assert audit.either_gate_fail[2]
    shard, scales = runner.summarize_cache_row(row, audit)
    shared = next(item for item in scales if item["scale_id"] == 1080)
    assert not shared["common_h_envelope_feasible"]
    assert shared["gate_fail_explained_by_rounding_count"] == 0
    assert shared["gate_fail_unexplained_count"] == shared["either_gate_fail_count"]
    assert shard["gate_fail_unexplained_count"] >= shared["gate_fail_unexplained_count"]


def test_positive_h_collapsed_to_zero_is_a_gate_failure_but_can_be_quantization_explained() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    collapsed_seed = np.asarray((1.0e8, 1.0e8, 1.0e8), dtype=np.float64)
    arrays["raw_features"][0] = _producer_raw(
        collapsed_seed[None, :], np.asarray((0.1,), dtype=np.float64)
    )[0]
    arrays["seeds_xyz"][0] = collapsed_seed
    row = _dummy_row(Path("synthetic.npz"))
    audit = runner.audit_geometry_arrays(plan, row, arrays)
    assert audit.zero_dx[0]
    assert audit.either_gate_fail[0]
    assert audit.rounding_feasible[0]
    assert audit.rounding_high[0] > 0.0
    shard, scales = runner.summarize_cache_row(row, audit)
    collapsed_scale = next(item for item in scales if item["scale_id"] == 0)
    assert collapsed_scale["common_h_envelope_feasible"]
    assert collapsed_scale["gate_fail_explained_by_rounding_count"] == 1
    assert collapsed_scale["gate_fail_unexplained_count"] == 0
    assert shard["zero_dx_count"] == 1


def test_completion_guard_runs_before_the_last_marker_is_published() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture_plan, _ = _write_manifest_fixture(root, plan, arrays)
        output = root / "guarded_output"

        def reject_completion() -> None:
            assert (output / "summary.json").is_file()
            assert not (output / "RUN_COMPLETE.json").exists()
            raise RuntimeError("synthetic completion guard rejection")

        _expect_error(
            RuntimeError,
            "completion guard rejection",
            lambda: runner.run_audit(
                fixture_plan,
                output,
                git_commit="a" * 40,
                completion_guard=reject_completion,
            ),
        )
        assert not (output / "RUN_COMPLETE.json").exists()


def test_whole_file_authentication_precedes_exact_allowlisted_member_reads() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "cache.npz"
        _write_npz(path, arrays)
        row = _dummy_row(path, _sha256(path), path.stat().st_size)
        original_load = np.load
        accessed: list[str] = []

        class TrackingArchive:
            def __init__(self, archive):
                self.archive = archive

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.archive.close()

            def __getitem__(self, name):
                accessed.append(str(name))
                return self.archive[name]

        def tracking_load(*args, **kwargs):
            return TrackingArchive(original_load(*args, **kwargs))

        with patch.object(runner.np, "load", side_effect=tracking_load):
            observed = runner._load_cache_members(plan, row)
        assert tuple(observed) == runner.ALLOWED_MEMBERS
        assert tuple(accessed) == runner.ALLOWED_MEMBERS
        assert not set(accessed) & set(runner.REQUIRED_FORBIDDEN_MEMBERS)

        accessed.clear()
        wrong = replace(row, sha256="0" * 64)
        with patch.object(runner.np, "load", side_effect=tracking_load):
            _expect_error(
                ValueError,
                "SHA-256 mismatch",
                lambda: runner._load_cache_members(plan, wrong),
            )
        assert accessed == []


def test_complete_32_shard_audit_is_self_hashed_and_never_overwritten() -> None:
    plan = runner.load_plan(CONFIG)
    arrays = _fixture_arrays(plan)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture_plan, _ = _write_manifest_fixture(root, plan, arrays)
        output = root / "audit_output"
        result = runner.run_audit(fixture_plan, output, git_commit="a" * 40)
        assert {path.name for path in output.iterdir()} == {
            "per_shard_geometry.csv",
            "per_scale_geometry.csv",
            "summary.json",
            "RUN_COMPLETE.json",
        }
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        completion = json.loads((output / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        runner.authenticate_self_hash(summary)
        runner.authenticate_self_hash(completion)
        assert summary["authenticated_cache_file_count"] == 32
        assert summary["opened_cache_members_in_exact_order"] == list(runner.ALLOWED_MEMBERS)
        assert summary["forbidden_cache_member_open_count"] == 0
        assert summary["label_access"] is False
        assert summary["counts"]["shard_count"] == 32
        assert summary["counts"]["row_count"] == 128
        assert summary["counts"]["gate_fail_unexplained_count"] == 0
        assert summary["rounding_envelope"]["hypothesis_status"] == (
            "quantization_explanation_supported_for_all_observed_gate_failures"
        )
        assert completion["summary_file_sha256"] == _sha256(output / "summary.json")
        assert completion["summary_content_sha256"] == summary["content_sha256"]
        assert result["completion_file_sha256"] == _sha256(output / "RUN_COMPLETE.json")
        for csv_name, expected_rows in (
            ("per_shard_geometry.csv", 32),
            ("per_scale_geometry.csv", 128),
        ):
            with (output / csv_name).open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            assert len(rows) == expected_rows
            assert summary["artifacts"][csv_name]["sha256"] == _sha256(output / csv_name)
            assert summary["artifacts"][csv_name]["row_count"] == expected_rows

        _expect_error(
            FileExistsError,
            "exists",
            lambda: runner.run_audit(fixture_plan, output, git_commit="a" * 40),
        )


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} dimensionless input geometry audit tests passed")
