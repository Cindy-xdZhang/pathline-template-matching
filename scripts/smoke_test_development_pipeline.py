#!/usr/bin/env python3
"""Run the complete development pipeline on small synthetic cache files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.development_data import (
    build_input_manifest,
    canonical_json_sha256,
)
from pathline_template_matching.development_experiment import evaluate_fold
from pathline_template_matching.development_report import finalize_development_run


def _cache(
    path: Path,
    *,
    dataset: str,
    phase: str,
    ordinal: int,
    scales: list[dict[str, object]],
    scale_set: str,
    config_sha256: str,
    seed: int,
    empty_first_scale: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    scale_id = np.repeat(np.arange(len(scales), dtype=np.int16), 4)
    reference = np.tile(np.asarray([False, False, False, True]), len(scales))
    if empty_first_scale:
        reference[:4] = False
    count = len(reference)
    seeds = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    raw = rng.normal(size=(count, 672)).astype(np.float32)
    # Preserve the cache interpretation raw = primitive_xyz - center seed.
    raw[:, :3] = 0.0
    metadata = {
        "experiment": "mainExp_Task5_3D_1.1",
        "config_sha256": config_sha256,
        "task": "Task5",
        "dataset": dataset,
        "phase": phase,
        "ordinal": ordinal,
        "source_start_index": ordinal * 8,
        "source_time": float(ordinal),
        "frame_count": 2,
        "source_time_step": 1.0,
        "loaded_shape_TZYXC": [2, 2, 2, 2, 3],
        "spatial_strides": {"x": 1, "y": 1, "z": 1},
        "scale_set": scale_set,
        "scale_table": scales,
        "assigned_count_by_scale": [4] * len(scales),
        "valid_count_by_scale": [4] * len(scales),
        "valid_primitives": count,
        "total_primitives": count,
        "sampled_steps": 32,
        "ivd_definition": "whole_field_ivd_percentile",
        "ivd_percentile": 95.0,
        "ivd_threshold": 1.0,
        "ivd_positive_count": int(reference.sum()),
        "ivd_positive_fraction": float(reference.mean()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    integration_steps = np.asarray(
        [int(scales[index]["integration_steps"]) for index in scale_id],
        dtype=np.int16,
    )
    np.savez_compressed(
        path,
        raw_features=raw,
        fmt_features=rng.normal(size=(count, 161)).astype(np.float32),
        reference=reference,
        seeds=seeds,
        scale_id=scale_id,
        physical_dt=np.full(count, 0.1, dtype=np.float32),
        integration_steps=integration_steps,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def run() -> None:
    development = yaml.safe_load(
        (ROOT / "config/mainExp_TemplateMatching_1.2_development.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = yaml.safe_load(
        (ROOT / "config/mainExp_TemplateMatching_1.1.yaml").read_text(encoding="utf-8")
    )
    development["library"]["maximum_templates_per_class_per_stratum"] = 1
    development["bootstrap"]["replicates"] = 10
    development["visualization"]["display_center_pathlines"]["count"] = 10
    config_digest = "b" * 64
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_dir = root / "config"
        config_dir.mkdir()
        development["base_config"] = "config/base.yaml"
        development["dataset_registry"] = "config/registry.yaml"
        development_path = config_dir / "development.yaml"
        development_path.write_text(
            yaml.safe_dump(development, sort_keys=False), encoding="utf-8"
        )
        (config_dir / "base.yaml").write_text(
            yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
        )
        datasets = []
        counter = 0
        for family, family_datasets in development["physical_families"].items():
            for dataset in family_datasets:
                development_root = root / "cache" / "development" / dataset
                confirmation_root = root / "cache" / "confirmation" / dataset
                for ordinal in range(6):
                    role = "library" if ordinal < 4 else "descriptor_selection_only"
                    _cache(
                        development_root / f"slice_{ordinal:02d}_index_{ordinal * 8:04d}.npz",
                        dataset=dataset,
                        phase="development",
                        ordinal=ordinal,
                        scales=base["scale_sets"][role],
                        scale_set="train" if ordinal < 4 else "validation",
                        config_sha256=config_digest,
                        seed=counter,
                        empty_first_scale=(dataset == "channel" and ordinal == 0),
                    )
                    counter += 1
                for ordinal in range(4):
                    _cache(
                        confirmation_root / f"slice_{ordinal:02d}_index_{ordinal * 8:04d}.npz",
                        dataset=dataset,
                        phase="confirmation",
                        ordinal=ordinal,
                        scales=base["scale_sets"]["unseen_scale_evaluation"],
                        scale_set="confirmation",
                        config_sha256=config_digest,
                        seed=counter,
                    )
                    counter += 1
                datasets.append(
                    {
                        "id": dataset,
                        "physical_family": family,
                        "kind": "netcdf",
                        "raw_paths": {"synthetic": []},
                        "legacy_task5_cache": {
                            "evidence_scope": "exposed_development_only",
                            "synthetic": {
                                "development": str(development_root),
                                "confirmation": str(confirmation_root),
                            },
                        },
                    }
                )
        registry = {
            "schema_version": 1,
            "legacy_task5_cache_contract": {
                "experiment": "mainExp_Task5_3D_1.1",
                "config_sha256": config_digest,
                "fmt_descriptor_id": "fmt_independent_3d_161d_sha256_25fce29499c9089e",
                "raw_feature_width": 672,
                "fmt_feature_width": 161,
            },
            "datasets": datasets,
        }
        (config_dir / "registry.yaml").write_text(
            yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
        )
        run_dir = root / "run"
        run_dir.mkdir()
        manifest, digest_by_path = build_input_manifest(
            development_path, environment="synthetic"
        )
        manifest.update({"git_commit": "synthetic-smoke", "created_utc": "synthetic"})
        manifest["manifest_content_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key != "manifest_content_sha256"
            }
        )
        (run_dir / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        for family in development["physical_families"]:
            evaluate_fold(
                development_path,
                run_dir,
                held_out_family=family,
                environment="synthetic",
                digest_by_path=digest_by_path,
                input_manifest_sha256=manifest["manifest_content_sha256"],
                git_commit="synthetic-smoke",
                device="cpu",
            )
            fold_manifest = json.loads(
                (run_dir / "folds" / family / "fold_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            assert (
                fold_manifest["pca"]["sample_count"]
                == fold_manifest["eligible_library_candidate_count"]
            )
            assert (
                fold_manifest["pca"]["sample_count"]
                > fold_manifest["balanced_library_count"]
            )
            expected_skipped = 0 if family == "channel" else 1
            assert fold_manifest["skipped_library_stratum_count"] == expected_skipped
            assert fold_manifest["skipped_library_candidate_count"] == 4 * expected_skipped
            with (run_dir / "folds" / family / "audit_counts.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                audit = list(csv.DictReader(source))
            skipped_rows = [
                row for row in audit if row.get("stratum_status") == "skipped_empty_class"
            ]
            assert len(skipped_rows) == 3 * expected_skipped
            assert sum(int(row["selected_count"]) for row in skipped_rows) == 0
        result = finalize_development_run(
            development_path,
            run_dir,
            render_environment="synthetic",
            figure_dpi=10,
        )
        assert result["status"] == "development_completed_confirmation_not_run"
        assert result["fold_count"] == 7
        assert result["triptych_count"] == 20
        assert len(result["folds"]) == 7
        for fold in result["folds"]:
            expected_skipped = 0 if fold["held_out_family"] == "channel" else 1
            assert fold["skipped_library_stratum_count"] == expected_skipped
            assert fold["skipped_library_candidate_count"] == 4 * expected_skipped
        assert len(result["visualization_artifacts"]) == 20
        assert (run_dir / "main_table.csv").is_file()
        assert (run_dir / "figures/visualization_manifest.json").is_file()
        report = (run_dir / "development_report.md").read_text(encoding="utf-8")
        assert "## Library construction audit" in report
        assert "| channel |" in report
        try:
            finalize_development_run(
                development_path,
                run_dir,
                render_environment="synthetic",
                figure_dpi=10,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("completed synthetic report was overwritten")
        print("synthetic development pipeline smoke: OK")


if __name__ == "__main__":
    run()
