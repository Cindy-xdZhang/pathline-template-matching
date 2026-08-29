"""Tests for the synthetic H48/2,000-scale verification gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.verify_long_arc_horizon_1_1 import (  # noqa: E402
    _write_phase_a_artifacts,
    load_frozen_inputs,
    verify_fail_closed,
    verify_full_constant_oracle,
    verify_horizon_boundaries,
    verify_scale_union_and_assignments,
    verify_time_varying_field,
)


class TestVerifyLongArcHorizon(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan, cls.legacy, cls.verify = load_frozen_inputs(
            ROOT / "config/mainExp_TemplateMatching_3.1.yaml",
            ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
        )

    def test_scale_union_assignment_and_H48_identity(self) -> None:
        result = verify_scale_union_and_assignments(
            self.plan, self.legacy, self.verify
        )
        self.assertEqual(result["scale_count"], 2_000)
        self.assertEqual(result["assignment_shape"], [128_000])
        self.assertEqual(result["assignment_count_per_scale_minimum"], 64)
        self.assertEqual(result["assignment_count_per_scale_maximum"], 64)
        self.assertEqual(result["maximum_source_frame_intervals"], 48.0)
        self.assertEqual(result["derived_window_frame_count"], 49)
        self.assertEqual(self.plan.scale_table.scale_id.dtype, np.dtype(np.int32))
        self.assertEqual(
            result["legacy_scale_subset_sha256"],
            self.verify["scale_union_gate"][
                "required_legacy_scale_subset_sha256"
            ],
        )
        self.assertEqual(
            result["legacy_assignment_sha256"],
            self.verify["assignment_gate"][
                "required_legacy_assignment_canonical_sha256"
            ],
        )

    def test_phase_a_writes_six_audited_outputs_then_marker(self) -> None:
        scale = verify_scale_union_and_assignments(
            self.plan, self.legacy, self.verify
        )
        main_path = ROOT / "config/mainExp_TemplateMatching_3.1.yaml"
        verify_path = ROOT / "config/Verify_LongArcHorizon_1.1.yaml"
        main_bytes = main_path.read_bytes()
        verify_bytes = verify_path.read_bytes()

        def sha256(payload: bytes) -> str:
            return hashlib.sha256(payload).hexdigest()

        provenance = {
            "git_commit": "1" * 40,
            "worktree_clean": True,
            "main_config_path": str(main_path),
            "main_config_sha256": sha256(main_bytes),
            "verify_config_path": str(verify_path),
            "verify_config_sha256": sha256(verify_bytes),
            "legacy_parent_config_path": "config/mainExp_TemplateMatching_2.1.yaml",
            "legacy_parent_config_sha256": "3" * 64,
            "dataset_registry_path": "config/datasets.yaml",
            "dataset_registry_sha256": "4" * 64,
            "source_sha256": {"synthetic_test": "2" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "phase_a"
            run_dir.mkdir()
            marker = _write_phase_a_artifacts(
                run_dir=run_dir,
                plan=self.plan,
                verify=self.verify,
                evidence={"scale_union_and_assignment": scale},
                main_config_bytes=main_bytes,
                verify_config_bytes=verify_bytes,
                provenance=provenance,
                environment={"device": "cpu"},
                completed_utc="2026-08-29T00:00:00+00:00",
            )
            expected = set(
                self.verify["execution"]["phase_a_synthetic"][
                    "required_outputs"
                ]
            )
            self.assertEqual({item.name for item in run_dir.iterdir()}, expected)
            self.assertEqual(len(marker["outputs"]), 6)
            self.assertFalse(marker["final_verify_pass"])
            self.assertEqual(marker["git_commit"], provenance["git_commit"])
            for row in marker["outputs"]:
                payload = (run_dir / row["path"]).read_bytes()
                self.assertEqual(row["size_bytes"], len(payload))
                self.assertEqual(row["sha256"], sha256(payload))
            on_disk = json.loads(
                (run_dir / "SYNTHETIC_PASS.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk, marker)
            with self.assertRaises(FileExistsError):
                _write_phase_a_artifacts(
                    run_dir=run_dir,
                    plan=self.plan,
                    verify=self.verify,
                    evidence={"scale_union_and_assignment": scale},
                    main_config_bytes=main_bytes,
                    verify_config_bytes=verify_bytes,
                    provenance=provenance,
                    environment={"device": "cpu"},
                    completed_utc="2026-08-29T00:00:00+00:00",
                )

    def test_full_union_constant_oracle_batch_and_permutation(self) -> None:
        result = verify_full_constant_oracle(self.plan, self.verify)
        self.assertEqual(result["valid_count"], 2_000)
        self.assertEqual(result["primitive_shape"], [2_000, 7, 32, 4])
        self.assertTrue(result["external_batches_exact"])
        self.assertTrue(result["input_permutation_exact"])
        self.assertEqual(result["theoretical_int32_step_bound"], 961)

    def test_H12_H48_and_longer_than_H48_boundaries(self) -> None:
        result = verify_horizon_boundaries(self.plan, self.verify)
        self.assertTrue(result["after_H12_before_H48_valid"])
        self.assertTrue(result["exactly_H48_valid"])
        self.assertTrue(result["requires_more_than_H48_invalid"])
        self.assertTrue(result["all_cases_invalid_at_H12"])
        self.assertTrue(result["thirteen_frame_H12_window_rejected_for_H48"])

    def test_time_varying_oracle_uses_frames_after_H12(self) -> None:
        result = verify_time_varying_field(self.plan, self.verify)
        self.assertTrue(result["uses_frames_after_H12"])
        self.assertTrue(result["H48_valid"])
        self.assertTrue(result["H12_invalid"])
        self.assertGreater(result["analytic_endpoint_time"], result["H12_endpoint_time"])

    def test_fail_closed_cases(self) -> None:
        result = verify_fail_closed(self.plan, self.verify)
        self.assertTrue(all(result.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
