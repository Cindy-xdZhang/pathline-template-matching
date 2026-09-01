from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np

from pathline_template_matching.portable_flow import canonical_array_sha256
from scripts import (
    aggregate_verify_source_centered_paired_scale_template_1_1 as aggregate,
)
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_SourceCenteredPairedScaleTemplate_1.1.yaml"


def _expect_value_error(callable_object: object, *args: object, match: str, **kwargs: object) -> None:
    try:
        callable_object(*args, **kwargs)  # type: ignore[operator]
    except ValueError as error:
        assert match in str(error), (match, str(error))
    else:
        raise AssertionError("expected ValueError")


def test_aggregate_fresh_inner_replay_rejects_tampered_selected_candidate() -> None:
    plan = runner.load_plan(CONFIG)
    correct, tampered = runner.candidate_specs(plan)[:2]
    paths = {
        "inner_group_metrics": (Path("metrics.csv"), "a" * 64),
        "inner_candidate_summary": (Path("summary.csv"), "b" * 64),
        "inner_fit_audits": (Path("audits.json"), "c" * 64),
    }
    summary = runner._candidate_payload(correct)
    with (
        patch.object(runner, "_parse_inner_metric_csv", return_value=[{"fresh": True}]) as parse,
        patch.object(
            runner,
            "aggregate_and_select_inner",
            return_value=([summary], correct, summary),
        ),
        patch.object(runner, "_authenticate_summary_csv"),
        patch.object(runner, "_authenticate_inner_fit_audits"),
    ):
        _expect_value_error(
            aggregate._fresh_inner_selection,
            plan,
            "half_cylinder",
            paths,
            claimed_candidate=tampered,
            claimed_summary=summary,
            match="differs from fresh inner selection",
        )
        parse.assert_called_once()


def _fake_population_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(32):
        rows.append(
            {
                "dataset": f"dataset_{index // 4}",
                "dataset_index": index // 4,
                "physical_family": f"family_{index // 4}",
                "source_ordinal": index % 4,
                "source_index": index,
                "completion_file_sha256": f"{index:064x}",
                "sidecar_file_sha256": f"{index + 32:064x}",
                "sidecar_combined_array_sha256": f"{index + 64:064x}",
                "valid_projection_sha256": f"{index + 96:064x}",
                "assigned_row_count": 128000,
                "valid_projection_row_count": 100000,
                "sidecar_relative_path": f"row_{index}/sidecar.npz",
                "sidecar_size_bytes": 1234,
            }
        )
    return rows


def test_aggregate_evidence_persists_population_and_fold_result_binding() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.json"
        population_path = root / "population.json"
        input_path.write_text("input", encoding="utf-8")
        population_path.write_text("population", encoding="utf-8")
        rows = _fake_population_rows()
        plan = SimpleNamespace(
            sha256="1" * 64,
            family_order=("family_0", "family_1"),
            sidecar_input_manifest_path=input_path,
            sidecar_input_manifest_file_sha256="2" * 64,
            sidecar_input_manifest_content_sha256="3" * 64,
            sidecar_root=root,
            sidecar_population_path=population_path,
            sidecar_population_file_sha256="4" * 64,
            sidecar_population_content_sha256="5" * 64,
            sidecar_population={
                "sidecar_count": 32,
                "rows_content_sha256": "6" * 64,
                "assigned_row_count_total": 4096000,
                "valid_projection_row_count_total": 3200000,
                "rows": rows,
            },
        )
        evidence = aggregate._aggregate_evidence_binding(
            plan, git_commit="7" * 40  # type: ignore[arg-type]
        )
        assert evidence["source_centered_input_manifest"]["path"] == str(input_path)
        assert len(evidence["source_centered_sidecars"]["row_identities"]) == 32

        candidate = runner.CandidateSpec(
            runner.REPRESENTATIONS[0], 1, 0.0, 0.0, 0.025
        )
        fold_evidence = runner._evidence_binding(
            plan,  # type: ignore[arg-type]
            representation=candidate.representation,
            fit_families=["family_1"],
        )
        result = {"source_centered_evidence": fold_evidence}
        aggregate._require_fold_source_evidence(
            plan, result, candidate, "family_0"  # type: ignore[arg-type]
        )
        result["source_centered_evidence"] = {
            **fold_evidence,
            "source_centered_input_manifest": {
                **fold_evidence["source_centered_input_manifest"],
                "file_sha256": "8" * 64,
            },
        }
        _expect_value_error(
            aggregate._require_fold_source_evidence,
            plan,
            result,
            candidate,
            "family_0",
            match="differs from aggregate inputs",
        )


def test_outer_source_mean_binding_is_written_and_freshly_authenticated() -> None:
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        rows = _fake_population_rows()
        rows[0]["dataset"] = "dataset_0"
        rows[0]["physical_family"] = "half_cylinder"
        rows[0]["source_ordinal"] = 0
        rows[0]["source_index"] = 0
        plan = SimpleNamespace(
            sha256="1" * 64,
            sidecar_root=destination,
            sidecar_population_path=destination / "population.json",
            sidecar_population_file_sha256="4" * 64,
            sidecar_population={"rows": rows},
        )
        mean = np.arange(60, dtype=np.float64).reshape(20, 3)
        cache = SimpleNamespace(
            row=SimpleNamespace(
                dataset="dataset_0",
                family="half_cylinder",
                source_ordinal=0,
                source_index=0,
            ),
            sidecar_file_sha256=rows[0]["sidecar_file_sha256"],
            sidecar_combined_array_sha256=rows[0][
                "sidecar_combined_array_sha256"
            ],
            sidecar_group_mean_curl_xyz=mean,
            sidecar_group_mean_curl_xyz_sha256=canonical_array_sha256(mean),
        )
        seal = runner._AUTHENTICATION_SEAL
        selection = SimpleNamespace(seal=seal, file_sha256="a" * 64)
        scaler = SimpleNamespace(
            seal=seal,
            manifest_file_sha256="b" * 64,
            artifact_file_sha256="c" * 64,
        )
        calibration = SimpleNamespace(
            seal=seal,
            manifest_file_sha256="d" * 64,
            artifact_file_sha256="e" * 64,
        )
        candidate = runner.CandidateSpec(
            runner.REPRESENTATIONS[0], 1, 0.0, 0.0, 0.025
        )
        path, digest = runner.write_outer_source_centered_binding(
            destination,
            [cache],
            plan=plan,  # type: ignore[arg-type]
            selected=candidate,
            selection=selection,  # type: ignore[arg-type]
            scaler=scaler,  # type: ignore[arg-type]
            calibration=calibration,  # type: ignore[arg-type]
            outer_family="half_cylinder",
            git_commit="f" * 40,
        )
        verified = runner.authenticate_outer_source_centered_binding(
            path,
            [cache],
            plan=plan,  # type: ignore[arg-type]
            selected=candidate,
            selection=selection,  # type: ignore[arg-type]
            scaler=scaler,  # type: ignore[arg-type]
            calibration=calibration,  # type: ignore[arg-type]
            outer_family="half_cylinder",
            git_commit="f" * 40,
            expected_file_sha256=digest,
        )
        assert verified.manifest["sources"][0]["group_mean_curl_xyz_sha256"] == canonical_array_sha256(mean)
