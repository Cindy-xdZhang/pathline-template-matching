from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import yaml

from scripts import (
    run_verify_class_conditional_template_score_resource_smoke_1_1 as smoke,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_audit_record(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    *,
    digest: str | None = None,
) -> dict[str, object]:
    selected_digest = digest or hashlib.sha256(name.encode("utf-8")).hexdigest()
    return {
        "dtype": dtype,
        "shape": list(shape),
        "size_bytes": int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize,
        "sha256": selected_digest,
        "finite_or_nonfloating": True,
    }


def _constructed_model_audit_fixture() -> dict[str, object]:
    scaler = {
        name: _array_audit_record(f"scaler/{name}", dtype, shape)
        for name, (dtype, shape) in smoke.SCALER_ARRAY_SPECS.items()
    }
    scaler["scale_id_int32"]["sha256"] = smoke.canonical_array_sha256(
        np.arange(smoke.SCALE_COUNT, dtype=np.int32)
    )
    class_audits: dict[str, dict[str, dict[str, object]]] = {}
    for family in smoke.FIT_FAMILIES:
        class_audits[family] = {}
        for class_name in smoke.CLASS_NAMES:
            class_audits[family][class_name] = {
                "library_row_count": smoke.K + 1,
                "library_supported_scale_count_k31": 1,
                "loo_reference_row_count_k31": smoke.K + 1,
                "loo_supported_scale_count_k31": 1,
                "zero_distance_loo_reference_count_k31": 0,
                "exact_self_exclusion_count_identity_passed": True,
                "duplicate_rows_retained_in_count_and_zero_distance_references": True,
                "class_scale_counts_sha256": hashlib.sha256(
                    f"counts/{family}/{class_name}".encode("utf-8")
                ).hexdigest(),
            }
    calibrator = {
        name: _array_audit_record(f"calibrator/{name}", dtype, shape)
        for name, (dtype, shape) in smoke.CALIBRATOR_BASE_ARRAY_SPECS.items()
    }
    member_specs = {
        "serialization_version": ("<i2", ()),
        "negative_features": ("<f4", (smoke.K + 1, smoke.FEATURE_WIDTH)),
        "negative_scale_offsets": ("<i8", (smoke.SCALE_COUNT + 1,)),
        "mean": ("<f8", (smoke.FEATURE_WIDTH,)),
        "raw_std": ("<f8", (smoke.FEATURE_WIDTH,)),
        "effective_std": ("<f8", (smoke.FEATURE_WIDTH,)),
        "zero_variance_feature_mask": ("|b1", (smoke.FEATURE_WIDTH,)),
        "ks": ("<i8", (1,)),
        "shrinkage_lambda": ("<f8", ()),
        f"loo_distances_k_{smoke.K}": ("<f4", (smoke.K + 1,)),
        f"loo_scale_offsets_k_{smoke.K}": (
            "<i8",
            (smoke.SCALE_COUNT + 1,),
        ),
    }
    for family_index in range(len(smoke.FIT_FAMILIES)):
        for class_index in range(len(smoke.CLASS_NAMES)):
            prefix = f"calibrator_f{family_index}_c{class_index}__"
            for suffix, (dtype, shape) in member_specs.items():
                name = f"{prefix}{suffix}"
                calibrator[name] = _array_audit_record(
                    f"calibrator/{name}", dtype, shape
                )
    family_dtype = np.asarray(smoke.FIT_FAMILIES, dtype=np.str_).dtype
    family_sha = smoke.canonical_array_sha256(
        np.asarray(smoke.FIT_FAMILIES, dtype=family_dtype)
    )
    calibrator["family_order_unicode"]["sha256"] = family_sha
    calibrator["family_order_copy_unicode"]["sha256"] = family_sha
    calibrator["ks_int64"]["sha256"] = smoke.canonical_array_sha256(
        np.asarray([smoke.K], dtype=np.int64)
    )
    calibrator["serialization_version_int16"]["sha256"] = (
        smoke.canonical_array_sha256(np.asarray(1, dtype=np.int16))
    )
    calibrator["required_family_count_int64"]["sha256"] = (
        smoke.canonical_array_sha256(np.asarray(2, dtype=np.int64))
    )
    calibrator["shrinkage_lambda_float64"]["sha256"] = (
        smoke.canonical_array_sha256(np.asarray(64.0, dtype=np.float64))
    )
    for family_index in range(len(smoke.FIT_FAMILIES)):
        for class_index in range(len(smoke.CLASS_NAMES)):
            prefix = f"calibrator_f{family_index}_c{class_index}__"
            calibrator[f"{prefix}serialization_version"]["sha256"] = calibrator[
                "serialization_version_int16"
            ]["sha256"]
            calibrator[f"{prefix}ks"]["sha256"] = calibrator["ks_int64"][
                "sha256"
            ]
            calibrator[f"{prefix}shrinkage_lambda"]["sha256"] = calibrator[
                "shrinkage_lambda_float64"
            ]["sha256"]
    calibrator["class_present_bool"]["sha256"] = smoke.canonical_array_sha256(
        np.ones(
            (len(smoke.FIT_FAMILIES), len(smoke.CLASS_NAMES)), dtype=np.bool_
        )
    )
    effective_counts_sha = "1" * 64
    calibrator["class_scale_counts_int64"]["sha256"] = effective_counts_sha
    total = len(smoke.FIT_FAMILIES) * len(smoke.CLASS_NAMES) * (smoke.K + 1)
    return {
        "family_order": list(smoke.FIT_FAMILIES),
        "feature_width": smoke.FEATURE_WIDTH,
        "k": smoke.K,
        "strict_majority_family_count": 2,
        "natural_raw_family_class_row_count": total + 8,
        "effective_retained_family_class_library_row_count": total,
        "all_constructed_arrays_finite": True,
        "exact_self_exclusion_duplicate_and_support_count_audits_passed": True,
        "natural_raw_present_exact_scale_count": 2,
        "effective_retained_exact_scale_count": 1,
        "natural_raw_only_no_scaler_exact_scale_count": 1,
        "natural_raw_present_exact_scale_ids_sha256": "c" * 64,
        "effective_retained_exact_scale_ids_sha256": "d" * 64,
        "natural_raw_only_no_scaler_exact_scale_ids_sha256": "e" * 64,
        "natural_raw_class_scale_counts_sha256": "f" * 64,
        "effective_retained_class_scale_counts_sha256": effective_counts_sha,
        "shared_negative_row_count": len(smoke.FIT_FAMILIES) * (smoke.K + 1),
        "full_family_class_library_row_count": total,
        "loo_reference_row_count_k31": total,
        "zero_distance_loo_reference_count_k31": 0,
        "class_library_and_reference_audits": class_audits,
        "scaler_arrays": scaler,
        "family_class_library_and_calibration_arrays": calibrator,
    }


def _synthetic_population(root: Path) -> tuple[list[SimpleNamespace], SimpleNamespace]:
    identities: list[tuple[str, str]] = []
    identities.extend((f"half_{index // 4}", smoke.RESERVED_OUTER_FAMILY) for index in range(12))
    identities.extend((f"delta_{index // 4}", smoke.RESERVED_INNER_FAMILY) for index in range(8))
    fit_datasets = {
        "f22_raptor": "f22raptor",
        "channel": "channel",
        "boeing_747": "boeing747",
    }
    identities.extend(
        (fit_datasets[family], family)
        for family in smoke.FIT_FAMILIES
        for _ in range(4)
    )
    ordinal_counts: dict[str, int] = {}
    rows: list[SimpleNamespace] = []
    population_rows: list[dict[str, object]] = []
    sidecar_root = root / "sidecars"
    sidecar_root.mkdir()
    for global_index, (dataset, family) in enumerate(identities):
        ordinal = ordinal_counts.get(dataset, 0)
        ordinal_counts[dataset] = ordinal + 1
        source_index = ordinal * 7
        parent = root / f"parent_{global_index}.npz"
        sidecar = sidecar_root / f"sidecar_{global_index}.npz"
        parent.write_bytes(f"parent-{global_index}".encode("ascii"))
        sidecar.write_bytes(f"sidecar-{global_index}".encode("ascii"))
        row = SimpleNamespace(
            dataset=dataset,
            family=family,
            source_ordinal=ordinal,
            source_index=source_index,
            path=parent,
            size_bytes=parent.stat().st_size,
            sha256=_digest(parent),
        )
        rows.append(row)
        population_rows.append(
            {
                "dataset": dataset,
                "physical_family": family,
                "source_ordinal": ordinal,
                "source_index": source_index,
                "sidecar_relative_path": sidecar.name,
                "sidecar_size_bytes": sidecar.stat().st_size,
                "sidecar_file_sha256": _digest(sidecar),
            }
        )
    plan = SimpleNamespace(
        sidecar_root=sidecar_root,
        sidecar_population={"sidecar_count": 32, "rows": population_rows},
    )
    return rows, plan


def _valid_release_payloads() -> tuple[dict[str, object], dict[str, object], str]:
    commit = "a" * 40
    source_identity = smoke._source_identity_records()
    frozen = smoke._frozen_release_contract()
    frozen_evidence = frozen["evidence"]
    assert isinstance(frozen_evidence, dict)
    evidence: dict[str, dict[str, object]] = {}
    for name, value in frozen_evidence.items():
        assert isinstance(value, dict)
        evidence[name] = dict(value)
        evidence_path = Path(str(value["path"]))
        evidence[name].setdefault(
            "size_bytes", evidence_path.stat().st_size if evidence_path.is_file() else 1
        )
    opened_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    family_datasets = {
        "f22_raptor": "f22raptor",
        "channel": "channel",
        "boeing_747": "boeing747",
    }
    train_metadata: dict[tuple[str, int], dict[str, object]] = {}
    sidecar_metadata: dict[tuple[str, int], dict[str, object]] = {}
    train_path = Path(str(evidence["train_cache_input_manifest"]["path"]))
    population_path = Path(str(evidence["sealed_sidecar_population"]["path"]))
    if train_path.is_file() and population_path.is_file():
        train_value = json.loads(train_path.read_text(encoding="utf-8"))
        population_value = json.loads(population_path.read_text(encoding="utf-8"))
        train_metadata = {
            (str(row["dataset"]), int(row["source_ordinal"])): row
            for row in train_value["rows"]
        }
        sidecar_metadata = {
            (str(row["dataset"]), int(row["source_ordinal"])): row
            for row in population_value["rows"]
        }
    for family in smoke.FIT_FAMILIES:
        dataset = family_datasets[family]
        for ordinal in range(4):
            train_row = train_metadata.get((dataset, ordinal))
            sidecar_row = sidecar_metadata.get((dataset, ordinal))
            fit_row_index = len(opened_rows)
            joined_row_count = (
                int(sidecar_row["sidecar_row_count"])
                if sidecar_row is not None
                else (17 if fit_row_index < 8 else 16)
            )
            source_index = (
                int(train_row["source_index"])
                if train_row is not None
                else ordinal * 7
            )
            parent_digest = (
                str(train_row["cache_file_sha256"])
                if train_row is not None
                else hashlib.sha256(
                    f"parent-{dataset}-{ordinal}".encode("ascii")
                ).hexdigest()
            )
            sidecar_digest = (
                str(sidecar_row["sidecar_file_sha256"])
                if sidecar_row is not None
                else hashlib.sha256(
                    f"sidecar-{dataset}-{ordinal}".encode("ascii")
                ).hexdigest()
            )
            opened = {
                "dataset": dataset,
                "physical_family": family,
                "source_ordinal": ordinal,
                "source_index": source_index,
                "joined_row_count": joined_row_count,
                "parent_cache": {
                    "path": (
                        str(Path(str(train_row["cache_path"])).resolve())
                        if train_row is not None
                        else f"/frozen/{dataset}/parent_{ordinal}.npz"
                    ),
                    "size_bytes": (
                        int(train_row["cache_size_bytes"])
                        if train_row is not None
                        else 11
                    ),
                    "sha256": parent_digest,
                },
                "parent_archive_members": list(smoke.PARENT_ARCHIVE_MEMBERS),
                "sidecar": {
                    "path": (
                        str(
                            (
                                population_path.parent
                                / str(sidecar_row["sidecar_relative_path"])
                            ).resolve()
                        )
                        if sidecar_row is not None
                        else f"/frozen/{dataset}/sidecar_{ordinal}.npz"
                    ),
                    "size_bytes": (
                        int(sidecar_row["sidecar_size_bytes"])
                        if sidecar_row is not None
                        else 13
                    ),
                    "sha256": sidecar_digest,
                },
                "sidecar_archive_members": list(smoke.SIDECAR_MEMBERS),
                "exact_identity_join_passed": True,
            }
            opened_rows.append(opened)
            fit_rows.append(
                {
                    **copy.deepcopy(opened),
                    "valid_row_count": joined_row_count,
                    "negative_row_count": 8,
                    "positive_row_count": joined_row_count - 8,
                }
            )
    audit = {
        "schema": smoke.AUDIT_SCHEMA,
        "experiment": smoke.EXPERIMENT,
        "stage": smoke.STAGE,
        "status": "passed",
        "evidence_scope": "resource_and_exact_path_only_no_method_quality_evidence",
        "git": {"git_commit": commit, "worktree_clean": True},
        "config": {"path": str(smoke.CONFIG_PATH), "sha256": smoke.EXPECTED_CONFIG_SHA256},
        "production_source_identity": {
            "files": dict(source_identity["files"]),
            "files_content_sha256": source_identity["files_content_sha256"],
        },
        "runtime": {
            "platform": "Linux-test",
            "python": "3.11.0",
            "slurm_job_id": "123456",
            "requested_device": "cpu",
            "gpu_requested": False,
            "slurm_cpus_per_task": 32,
            "slurm_job_partition": "cpu",
            "slurm_job_account": "pi-hadwigm",
            "slurm_scontrol_features": "rome",
            "slurm_memory_per_node": str(smoke.SLURM_MEMORY_PER_NODE_MIB),
            "slurm_num_nodes": 1,
            "slurm_time_limit": smoke.RESOURCE_SMOKE_TIME_LIMIT,
            "slurm_gpu_allocation": "none",
            "slurm_scontrol_source": "scontrol_show_job_one_line",
            "slurm_scontrol_record_sha256": "2" * 64,
            "slurm_scontrol_alloc_tres": "cpu=32,mem=128G,node=1,billing=32",
            "slurm_scontrol_req_tres": "cpu=32,mem=128G,node=1,billing=32",
            "slurm_scontrol_tres_per_node": "(not_set)",
            "slurm_scontrol_gres": "(not_set)",
            "cpu_model": "AMD EPYC 7702 64-Core Processor",
        },
        "authenticated_evidence": evidence,
        "data_access": {
            "semantic_separation": {
                "population_envelope_whole_file_authentication": "all sealed sidecar byte files were hashed without archive-member deserialization",
                "archive_member_deserialization": "only the three fixed fit families below",
            },
            "population_envelope_sidecar_count": 32,
            "member_open_counts_by_family": {
                "half_cylinder": 0,
                "delta_wing": 0,
                "f22_raptor": 4,
                "channel": 4,
                "boeing_747": 4,
            },
            "reserved_family_zero_member_open": {
                "half_cylinder": True,
                "delta_wing": True,
            },
            "reserved_parent_cache_whole_file_open_count": 0,
            "reserved_sidecar_archive_member_open_count": 0,
            "reserved_sidecar_envelope_hash_note": "sealed byte files are whole-file hashed by the mandatory 32-file population envelope gate without NPZ member deserialization",
            "opened_fit_rows": opened_rows,
        },
        "fit_input_rows": fit_rows,
        "synthetic_arithmetic_gate": {
            "feature_width": smoke.FEATURE_WIDTH,
            "family_count": len(smoke.FIT_FAMILIES),
            "rows_per_family_class": smoke.K + 1,
            "self_exclusion_forces_positive_k31_distance": True,
            "thirty_one_equal_duplicates_retained_per_family_class": True,
            "artifact_round_trip_exact": True,
            "strict_threshold_tie_contract_passed": True,
            "positive_only_no_scaler_exact_scale_all_class_family_support_false": True,
        },
        "constructed_model": _constructed_model_audit_fixture(),
        "synthetic_query_path": {
            "construction": "deterministic_integer_index_formula_independent_of_fit_rows_and_labels",
            "query_row_count": 2,
            "query_scale_domain": "union_of_all_natural_raw_exact_scales_before_scaler_filter",
            "all_natural_raw_present_exact_scales_exercised": True,
            "natural_raw_present_exact_scale_count": 2,
            "effective_retained_exact_scale_count": 1,
            "natural_raw_only_no_scaler_exact_scale_count": 1,
            "natural_raw_present_exact_scale_ids_sha256": "c" * 64,
            "effective_retained_exact_scale_ids_sha256": "d" * 64,
            "natural_raw_only_no_scaler_exact_scale_ids_sha256": "e" * 64,
            "all_natural_raw_only_no_scaler_scales_have_all_class_family_retrieval_and_calibration_unsupported": True,
            "strict_majority_joint_supported_row_count": 1,
            "strict_majority_retrieval_supported_row_count": 1,
            "joint_family_count_histogram": {"0": 1, "1": 0, "2": 0, "3": 1},
            "support_count_arithmetic_passed": True,
            "all_supported_numerical_values_finite": True,
            "reference_labels_consulted_by_query_path": False,
        },
        "resource": {
            "elapsed_seconds_before_audit_publish": 1.0,
            "linux_peak_rss_bytes_before_audit_publish": 1024,
            "memory_limit_bytes_exclusive": smoke.MEMORY_LIMIT_BYTES,
            "walltime_limit_seconds_inclusive": smoke.WALLTIME_LIMIT_SECONDS,
            "peak_memory_strictly_below_limit": True,
            "elapsed_at_or_below_limit": True,
        },
        "final_gates": {
            "clean_git_identity_unchanged": True,
            "reserved_family_zero_member_open": True,
            "forbidden_dataset_member_open": False,
            "all_constructed_arrays_finite": True,
            "exact_self_exclusion_duplicate_and_support_counts": True,
            "raw_and_effective_scale_domains_distinguished": True,
            "synthetic_queries_only": True,
            "quality_outputs_absent": True,
        },
    }
    marker = {
        "schema": smoke.PASS_SCHEMA,
        "experiment": smoke.EXPERIMENT,
        "stage": smoke.STAGE,
        "status": "passed",
        "git_commit": commit,
        "worktree_clean": True,
        "config_sha256": smoke.EXPECTED_CONFIG_SHA256,
        "elapsed_seconds": 2.0,
        "linux_peak_rss_bytes": 2048,
        "memory_limit_bytes_exclusive": smoke.MEMORY_LIMIT_BYTES,
        "walltime_limit_seconds_inclusive": smoke.WALLTIME_LIMIT_SECONDS,
        "reserved_family_zero_member_open": True,
        "forbidden_dataset_member_open": False,
        "all_constructed_arrays_finite": True,
        "exact_path_and_resource_gates_passed": True,
    }
    return audit, marker, commit


@contextmanager
def _temporary_frozen_output_root(root: Path):
    original = smoke._frozen_release_contract
    base = original()
    metadata_root = root.parent / "authenticated_metadata"
    sidecar_root = metadata_root / "sidecars"
    sidecar_root.mkdir(parents=True)
    dataset_families = (
        ("cylinder3d", "half_cylinder"),
        ("halfcylinderRe640", "half_cylinder"),
        ("halfcylinderRe6400", "half_cylinder"),
        ("deltaWing_resampled", "delta_wing"),
        ("deltaWing_LBM", "delta_wing"),
        ("f22raptor", "f22_raptor"),
        ("channel", "channel"),
        ("boeing747", "boeing_747"),
    )
    train_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    for dataset, family in dataset_families:
        for ordinal in range(4):
            source_index = ordinal * 7
            if family in smoke.FIT_FAMILIES:
                fit_row_index = smoke.FIT_FAMILIES.index(family) * 4 + ordinal
                sidecar_row_count = 17 if fit_row_index < 8 else 16
            else:
                sidecar_row_count = 3
            parent_digest = hashlib.sha256(
                f"parent-{dataset}-{ordinal}".encode("ascii")
            ).hexdigest()
            sidecar_digest = hashlib.sha256(
                f"sidecar-{dataset}-{ordinal}".encode("ascii")
            ).hexdigest()
            train_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": family,
                    "source_ordinal": ordinal,
                    "source_index": source_index,
                    "cache_path": str(
                        (metadata_root / "cache" / dataset / f"parent_{ordinal}.npz").resolve()
                    ),
                    "cache_size_bytes": 11,
                    "cache_file_sha256": parent_digest,
                }
            )
            population_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": family,
                    "source_ordinal": ordinal,
                    "source_index": source_index,
                    "sidecar_relative_path": f"{dataset}/sidecar_{ordinal}.npz",
                    "sidecar_size_bytes": 13,
                    "sidecar_file_sha256": sidecar_digest,
                    "sidecar_row_count": sidecar_row_count,
                }
            )
    train = {
        "schema": "pathline_template_matching.long_arc_train_cache_input.v1",
        "row_count": 32,
        "rows": train_rows,
        "rows_content_sha256": smoke.canonical_json_sha256(train_rows),
        "test_dataset_access": False,
    }
    train_path = metadata_root / "train_cache_input_manifest.json"
    train_path.write_text(
        json.dumps(train, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    population_base = {
        "schema": smoke.POPULATION_MANIFEST_SCHEMA,
        "sidecar_count": 32,
        "rows": population_rows,
        "rows_content_sha256": smoke.canonical_json_sha256(population_rows),
    }
    population = dict(population_base)
    population["content_sha256"] = smoke.canonical_json_sha256(population_base)
    population_path = sidecar_root / "SIDECAR_POPULATION.json"
    population_path.write_text(
        json.dumps(population, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    kinematic_path = metadata_root / "kinematic_input_manifest.json"
    kinematic_path.write_text(
        json.dumps({"fixture": "kinematic_input_manifest"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent_synthetic_path = metadata_root / "SYNTHETIC_PASS.json"
    parent_synthetic_path.write_text(
        json.dumps({"fixture": "parent_synthetic_pass"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence = copy.deepcopy(base["evidence"])
    evidence["train_cache_input_manifest"] = {
        "path": str(train_path.resolve()),
        "size_bytes": train_path.stat().st_size,
        "sha256": _digest(train_path),
        "schema": train["schema"],
        "rows_content_sha256": train["rows_content_sha256"],
    }
    evidence["sealed_sidecar_population"] = {
        "path": str(population_path.resolve()),
        "sha256": _digest(population_path),
    }
    for name, path in (
        ("kinematic_input_manifest", kinematic_path),
        ("parent_synthetic_pass", parent_synthetic_path),
    ):
        record: dict[str, object] = {
            "path": str(path.resolve()),
            "sha256": _digest(path),
        }
        existing = evidence[name]
        assert isinstance(existing, dict)
        if "size_bytes" in existing:
            record["size_bytes"] = path.stat().st_size
        evidence[name] = record

    def patched() -> dict[str, object]:
        return {
            "output_root": str(root.resolve()),
            "evidence": copy.deepcopy(evidence),
        }

    smoke._frozen_release_contract = patched
    try:
        yield
    finally:
        smoke._frozen_release_contract = original


def _release_output(root: Path, *, commit: str, job_id: str = "123456") -> Path:
    output = root / "resource_smoke" / f"slurm_{job_id}_{commit[:12]}"
    output.mkdir(parents=True)
    return output


def _expect_self_consistent_release_rejection(
    mutate: object, *, path_mode: str = "valid"
) -> None:
    assert callable(mutate)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        root = temporary / "frozen_output"
        with _temporary_frozen_output_root(root):
            audit, marker, commit = _valid_release_payloads()
            mutate(audit)
            if path_mode == "valid":
                output = _release_output(root, commit=commit)
            elif path_mode == "wrong_basename":
                output = root / "resource_smoke" / f"slurm_999999_{commit[:12]}"
                output.mkdir(parents=True)
            elif path_mode == "outside_root":
                output = temporary / "outside" / f"slurm_123456_{commit[:12]}"
                output.mkdir(parents=True)
            else:
                raise AssertionError(f"unknown path mode: {path_mode}")
            _audit_path, pass_path, _audit_sha, pass_sha = (
                smoke._publish_smoke_evidence(output, audit, marker)
            )
            try:
                smoke.authenticate_resource_smoke_release(
                    pass_path, pass_sha, commit, smoke.EXPECTED_CONFIG_SHA256
                )
            except RuntimeError:
                return
            raise AssertionError(
                f"self-consistent forged release was accepted: {path_mode}"
            )


def test_frozen_resource_smoke_contract_matches_exact_config() -> None:
    payload = smoke.CONFIG_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == smoke.EXPECTED_CONFIG_SHA256
    raw = yaml.safe_load(payload.decode("utf-8"))
    contract = raw["resource_smoke"]
    split = contract["fixed_split_identity"]
    worst = contract["worst_case_path"]
    slurm = contract["slurm"]
    assert split == {
        "reserved_outer_family_not_opened": "half_cylinder",
        "reserved_inner_family_not_opened": "delta_wing",
        "fit_families_opened": ["f22_raptor", "channel", "boeing_747"],
    }
    assert worst["representation"] == "fmt161_plus_seed4"
    assert worst["k"] == 31
    assert worst["scales"] == "all_exact_scales_present_in_the_three_fit_families"
    assert worst["query_exercise"] == "deterministic_label_free_synthetic_queries_only"
    assert slurm["cpus_per_task"] == 32
    assert slurm["memory_gb"] == 128
    assert slurm["walltime"] == "04:00:00"
    assert slurm["gpu"] == "none"
    assert slurm["account"] == smoke.FROZEN_CONFIG_SLURM_ACCOUNT == "deepvortex"
    assert smoke.RUNTIME_SLURM_ACCOUNT == "pi-hadwigm"
    wrapper = smoke.SMOKE_WRAPPER_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --constraint=rome\n" in wrapper
    assert "#SBATCH --partition=cpu\n" in wrapper
    assert "#SBATCH --account=pi-hadwigm\n" in wrapper
    assert "#SBATCH --cpus-per-task=32\n" in wrapper
    assert "#SBATCH --mem=128G\n" in wrapper
    assert "#SBATCH --time=04:00:00\n" in wrapper
    assert "cpu_amd_epyc_7702" not in wrapper


def test_frozen_evidence_fixture_matches_real_runtime_identity_shapes() -> None:
    audit, _marker, _commit = _valid_release_payloads()
    evidence = audit["authenticated_evidence"]
    assert isinstance(evidence, dict)
    assert set(evidence["train_cache_input_manifest"]) == {
        "path",
        "size_bytes",
        "sha256",
        "schema",
        "rows_content_sha256",
    }
    for name in (
        "kinematic_input_manifest",
        "parent_synthetic_pass",
        "sealed_sidecar_population",
    ):
        assert set(evidence[name]) == {"path", "size_bytes", "sha256"}
    smoke._validate_authenticated_evidence(
        evidence, frozen_contract=smoke._frozen_release_contract()
    )


def test_reserved_family_selection_fails_closed_before_any_loader_call() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rows, _plan = _synthetic_population(Path(directory))
        calls: list[object] = []

        def forbidden_loader(*args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            raise AssertionError("loader must not be called")

        try:
            smoke._selected_fit_rows(
                rows,
                fit_families=(
                    smoke.RESERVED_OUTER_FAMILY,
                    "channel",
                    "boeing_747",
                ),
            )
        except RuntimeError as error:
            assert "fit-family" in str(error)
        else:
            raise AssertionError("reserved family was accepted as a fit family")
        assert calls == []
        ledger = smoke.AccessLedger()
        try:
            ledger.authorize(smoke.RESERVED_INNER_FAMILY)
        except RuntimeError as error:
            assert "forbidden" in str(error)
        else:
            raise AssertionError("reserved family member access was authorized")
        assert calls == []


def test_fit_loader_records_exact_members_and_reserved_zero_open() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rows, plan = _synthetic_population(Path(directory))
        called_families: list[str] = []

        def loader(_plan: object, row: SimpleNamespace, *, include_labels: bool) -> object:
            assert include_labels is True
            assert row.family in smoke.FIT_FAMILIES
            called_families.append(row.family)
            count = 3
            return SimpleNamespace(
                row=row,
                fmt_features=np.zeros((count, 161), dtype=np.float32),
                seed_kinematic4=np.zeros((count, 4), dtype=np.float32),
                scale_ids=np.arange(count, dtype=np.int32),
                labels=np.asarray([False, True, False], dtype=np.bool_),
                count=count,
                sidecar_file_sha256=smoke._population_row(_plan, row)[
                    "sidecar_file_sha256"
                ],
            )

        ledger = smoke.AccessLedger()
        projections = smoke._load_fit_projections(
            plan, rows, ledger, loader=loader
        )
        assert len(projections) == smoke.EXPECTED_FIT_ROW_COUNT
        assert called_families == [family for family in smoke.FIT_FAMILIES for _ in range(4)]
        ledger.validate()
        audit = ledger.as_json(population_sidecar_count=32)
        assert audit["reserved_family_zero_member_open"] == {
            "half_cylinder": True,
            "delta_wing": True,
        }
        assert audit["reserved_parent_cache_whole_file_open_count"] == 0
        assert audit["reserved_sidecar_archive_member_open_count"] == 0
        for record in audit["opened_fit_rows"]:
            assert tuple(record["parent_archive_members"]) == smoke.PARENT_ARCHIVE_MEMBERS
            assert tuple(record["sidecar_archive_members"]) == smoke.SIDECAR_MEMBERS
            assert record["parent_cache"]["sha256"] == _digest(
                Path(record["parent_cache"]["path"])
            )
            assert record["sidecar"]["sha256"] == _digest(
                Path(record["sidecar"]["path"])
            )
        fit_rows = []
        for projection, opened in zip(projections, ledger.records, strict=True):
            fit_rows.append(
                {
                    **copy.deepcopy(opened),
                    "valid_row_count": projection.count,
                    "negative_row_count": int(np.count_nonzero(~projection.labels)),
                    "positive_row_count": int(np.count_nonzero(projection.labels)),
                }
            )
        smoke._validate_fit_access_cross_binding(ledger.records, fit_rows)


def test_expected_counts_ignore_positive_rows_without_pooled_negative_scale_support() -> None:
    def projections(*, include_unsupported_positive: bool) -> list[SimpleNamespace]:
        result: list[SimpleNamespace] = []
        for family in smoke.FIT_FAMILIES:
            labels = [False, True]
            scales = [0, 0]
            if include_unsupported_positive:
                labels.append(True)
                scales.append(1)
            result.append(
                SimpleNamespace(
                    row=SimpleNamespace(family=family),
                    labels=np.asarray(labels, dtype=np.bool_),
                    scale_ids=np.asarray(scales, dtype=np.int64),
                    count=len(labels),
                )
            )
        return result

    baseline = smoke._expected_class_scale_counts(
        projections(include_unsupported_positive=False)
    )
    augmented = smoke._expected_class_scale_counts(
        projections(include_unsupported_positive=True)
    )
    assert np.array_equal(augmented, baseline)
    assert int(augmented[:, 1, 1].sum()) == 0


def test_absent_positive_family_is_valid_for_model_array_and_query_audits() -> None:
    batches: dict[str, smoke.FamilyFitBatch] = {}
    projections: list[SimpleNamespace] = []
    for family_index, family in enumerate(smoke.FIT_FAMILIES):
        negative = np.zeros((smoke.K + 1, smoke.FEATURE_WIDTH), dtype=np.float32)
        negative[-1, 0] = np.float32(1.0 + family_index * 0.125)
        if family_index == 0:
            features = negative
            labels = np.zeros(smoke.K + 1, dtype=np.bool_)
        else:
            positive = negative.copy()
            features = np.concatenate((negative, positive), axis=0)
            labels = np.concatenate(
                (
                    np.zeros(smoke.K + 1, dtype=np.bool_),
                    np.ones(smoke.K + 1, dtype=np.bool_),
                )
            )
        scales = np.zeros(len(features), dtype=np.int64)
        batches[family] = smoke.FamilyFitBatch(features, scales, labels)
        projections.append(
            SimpleNamespace(
                row=SimpleNamespace(family=family),
                labels=labels,
                scale_ids=scales,
                count=len(labels),
            )
        )

    natural = smoke._natural_class_scale_counts(projections)
    expected = smoke._effective_class_scale_counts(natural)
    assert int(expected[0, 1].sum()) == 0
    model = smoke.ClassConditionalTemplateScoreModel(
        batches,
        family_order=smoke.FIT_FAMILIES,
        ks=(smoke.K,),
        device="cpu",
        query_chunk_size=64,
        library_chunk_size=64,
    )
    model_audit, _scaler_arrays, _calibrator_arrays = smoke._audit_model_arrays(
        model, natural, expected
    )
    smoke._validate_constructed_model_artifact_audit(model_audit)
    assert (
        model_audit["class_library_and_reference_audits"]["f22_raptor"]
        ["positive"]["library_row_count"]
        == 0
    )
    query_audit = smoke._exercise_synthetic_query_path(model, natural, expected)
    assert query_audit["support_count_arithmetic_passed"] is True


def test_positive_only_no_scaler_raw_scale_is_queried_and_all_support_is_false() -> None:
    batches: dict[str, smoke.FamilyFitBatch] = {}
    projections: list[SimpleNamespace] = []
    for family_index, family in enumerate(smoke.FIT_FAMILIES):
        one_class = np.zeros(
            (smoke.K + 1, smoke.FEATURE_WIDTH), dtype=np.float32
        )
        one_class[-1, 0] = np.float32(1.0 + family_index * 0.125)
        features = np.concatenate((one_class, one_class, one_class), axis=0)
        labels = np.concatenate(
            (
                np.zeros(smoke.K + 1, dtype=np.bool_),
                np.ones(2 * (smoke.K + 1), dtype=np.bool_),
            )
        )
        scales = np.concatenate(
            (
                np.zeros(2 * (smoke.K + 1), dtype=np.int64),
                np.ones(smoke.K + 1, dtype=np.int64),
            )
        )
        batches[family] = smoke.FamilyFitBatch(features, scales, labels)
        projections.append(
            SimpleNamespace(
                row=SimpleNamespace(family=family),
                labels=labels,
                scale_ids=scales,
                count=len(labels),
            )
        )

    natural = smoke._natural_class_scale_counts(projections)
    effective = smoke._effective_class_scale_counts(natural)
    assert np.all(natural[:, 1, 1] == smoke.K + 1)
    assert np.all(natural[:, 0, 1] == 0)
    assert int(effective[:, :, 1].sum()) == 0
    model = smoke.ClassConditionalTemplateScoreModel(
        batches,
        family_order=smoke.FIT_FAMILIES,
        ks=(smoke.K,),
        device="cpu",
        query_chunk_size=64,
        library_chunk_size=64,
    )
    model_audit, _scaler_arrays, _calibrator_arrays = smoke._audit_model_arrays(
        model, natural, effective
    )
    assert set(model_audit) == smoke.CONSTRUCTED_MODEL_FIELDS
    smoke._validate_constructed_model_artifact_audit(model_audit)
    assert model_audit["natural_raw_present_exact_scale_count"] == 2
    assert model_audit["effective_retained_exact_scale_count"] == 1
    assert model_audit["natural_raw_only_no_scaler_exact_scale_count"] == 1
    query_audit = smoke._exercise_synthetic_query_path(
        model, natural, effective
    )
    assert set(query_audit) == smoke.SYNTHETIC_QUERY_FIELDS
    assert query_audit["query_row_count"] == 2
    assert query_audit["natural_raw_present_exact_scale_count"] == 2
    assert query_audit["effective_retained_exact_scale_count"] == 1
    assert query_audit["natural_raw_only_no_scaler_exact_scale_count"] == 1
    assert (
        query_audit[
            "all_natural_raw_only_no_scaler_scales_have_all_class_family_retrieval_and_calibration_unsupported"
        ]
        is True
    )


def test_synthetic_core_full_path_self_exclusion_roundtrip_and_strict_tie() -> None:
    audit = smoke._synthetic_core_contract_gate()
    assert audit["self_exclusion_forces_positive_k31_distance"] is True
    assert audit["thirty_one_equal_duplicates_retained_per_family_class"] is True
    assert audit["artifact_round_trip_exact"] is True
    assert audit["strict_threshold_tie_contract_passed"] is True
    assert (
        audit[
            "positive_only_no_scaler_exact_scale_all_class_family_support_false"
        ]
        is True
    )


def test_array_audit_requires_finite_values_and_hashes_exact_bytes() -> None:
    values = np.asarray([[1.0, 2.0]], dtype=np.float32)
    record = smoke._array_record(values)
    assert record["dtype"] == "<f4"
    assert record["shape"] == [1, 2]
    assert record["size_bytes"] == values.nbytes
    assert len(record["sha256"]) == 64
    try:
        smoke._array_record(np.asarray([np.nan], dtype=np.float64))
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite constructed array was accepted")


def test_resource_limits_use_strict_memory_and_inclusive_four_hour_boundary() -> None:
    smoke._validate_resource_limits(
        peak_rss_bytes=smoke.MEMORY_LIMIT_BYTES - 1,
        elapsed_seconds=float(smoke.WALLTIME_LIMIT_SECONDS),
    )
    invalid = (
        (smoke.MEMORY_LIMIT_BYTES, 1.0),
        (1, float(smoke.WALLTIME_LIMIT_SECONDS) + 1e-6),
        (-1, 1.0),
        (1, -1.0),
    )
    for peak, elapsed in invalid:
        try:
            smoke._validate_resource_limits(
                peak_rss_bytes=peak, elapsed_seconds=elapsed
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"invalid resource observation was accepted: {peak}, {elapsed}"
            )


def test_runtime_payload_rejects_partition_account_constraint_and_memory_drift() -> None:
    valid = {
        "platform": "Linux-test",
        "python": "3.11.0",
        "slurm_job_id": "123456",
        "slurm_cpus_per_task": 32,
        "slurm_job_partition": "cpu",
        "slurm_job_account": "pi-hadwigm",
        "slurm_scontrol_features": "rome",
        "slurm_memory_per_node": str(smoke.SLURM_MEMORY_PER_NODE_MIB),
        "slurm_num_nodes": 1,
        "slurm_time_limit": smoke.RESOURCE_SMOKE_TIME_LIMIT,
        "slurm_gpu_allocation": "none",
        "slurm_scontrol_source": "scontrol_show_job_one_line",
        "slurm_scontrol_record_sha256": "2" * 64,
        "slurm_scontrol_alloc_tres": "cpu=32,mem=128G,node=1,billing=32",
        "slurm_scontrol_req_tres": "cpu=32,mem=128G,node=1,billing=32",
        "slurm_scontrol_tres_per_node": "(not_set)",
        "slurm_scontrol_gres": "(not_set)",
        "cpu_model": "AMD EPYC 7702 64-Core Processor",
        "requested_device": "cpu",
        "gpu_requested": False,
    }
    smoke._validate_frozen_slurm_runtime_payload(valid)
    mutations = {
        "slurm_job_partition": "gpu",
        "slurm_job_account": "other",
        "slurm_scontrol_features": "notrome",
        "slurm_memory_per_node": str(smoke.SLURM_MEMORY_PER_NODE_MIB - 1),
        "slurm_num_nodes": 2,
        "slurm_time_limit": "05:00:00",
        "slurm_gpu_allocation": "gres/gpu=1",
    }
    for field, value in mutations.items():
        changed = dict(valid)
        changed[field] = value
        try:
            smoke._validate_frozen_slurm_runtime_payload(changed)
        except RuntimeError as error:
            assert "frozen Slurm allocation" in str(error)
        else:
            raise AssertionError(f"runtime drift was accepted: {field}")


def test_runtime_uses_scontrol_features_without_optional_constraint_environment() -> None:
    allocation = {
        "source": "scontrol_show_job_one_line",
        "job_id": "123456",
        "num_nodes": 1,
        "time_limit": smoke.RESOURCE_SMOKE_TIME_LIMIT,
        "features": "rome",
        "gpu_allocation": "none",
        "alloc_tres": "cpu=32,mem=128G,node=1,billing=32",
        "req_tres": "cpu=32,mem=128G,node=1,billing=32",
        "tres_per_node": "(not_set)",
        "gres": "(not_set)",
        "record_sha256": "2" * 64,
    }
    captured: dict[str, object] = {}
    environment = {
        "SLURM_JOB_ID": "123456",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_JOB_PARTITION": "cpu",
        "SLURM_JOB_ACCOUNT": "pi-hadwigm",
        "SLURM_MEM_PER_NODE": str(smoke.SLURM_MEMORY_PER_NODE_MIB),
    }
    with (
        patch.dict(os.environ, environment, clear=False),
        patch.object(smoke.sys, "platform", "linux"),
        patch.object(
            smoke,
            "_validated_scontrol_allocation",
            return_value=allocation,
        ),
        patch.object(
            smoke,
            "_validate_frozen_slurm_runtime_payload",
            side_effect=lambda value: captured.update(value),
        ),
    ):
        os.environ.pop("SLURM_JOB_CONSTRAINTS", None)
        result = smoke._runtime_environment_audit()
    assert result["slurm_scontrol_features"] == "rome"
    assert "slurm_job_constraints" not in result
    assert captured == result


def test_scontrol_parser_rejects_cli_node_gpu_and_time_overrides() -> None:
    base = (
        "JobId=123456 JobName=PTMClassSmoke Partition=cpu Account=pi-hadwigm "
        "NumNodes=1 NumCPUs=32 CPUs/Task=32 TimeLimit=04:00:00 Features=rome "
        "ReqTRES=cpu=32,mem=128G,node=1,billing=32 "
        "AllocTRES=cpu=32,mem=128G,node=1,billing=32 "
        "TresPerNode=(null) Gres=(null) JobState=RUNNING"
    )
    allocation = smoke._validated_scontrol_allocation(
        base,
        expected_job_id="123456",
        expected_time_limit=smoke.RESOURCE_SMOKE_TIME_LIMIT,
    )
    assert allocation["num_nodes"] == 1
    assert allocation["time_limit"] == "04:00:00"
    assert allocation["features"] == "rome"
    assert allocation["gpu_allocation"] == "none"
    mutations = (
        base.replace("NumNodes=1", "NumNodes=2"),
        base.replace("TimeLimit=04:00:00", "TimeLimit=05:00:00"),
        base.replace("Features=rome ", ""),
        base.replace("Features=rome", "Features=notrome"),
        base.replace("Features=rome", "Features=zen3|rome"),
        base.replace(
            "AllocTRES=cpu=32,mem=128G,node=1,billing=32",
            "AllocTRES=cpu=32,mem=128G,node=1,gres/gpu=1,billing=32",
        ),
    )
    for record in mutations:
        try:
            smoke._validated_scontrol_allocation(
                record,
                expected_job_id="123456",
                expected_time_limit=smoke.RESOURCE_SMOKE_TIME_LIMIT,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("scontrol CLI resource override was accepted")


def test_forbidden_quality_output_fields_are_rejected_recursively() -> None:
    smoke._assert_no_forbidden_output_fields(
        {"resource": {"elapsed_seconds": 1.0}, "support_count": 3}
    )
    for key in sorted(smoke.FORBIDDEN_OUTPUT_FIELD_NAMES):
        try:
            smoke._assert_no_forbidden_output_fields({"nested": [{key: 1}]})
        except RuntimeError as error:
            assert "forbidden" in str(error)
        else:
            raise AssertionError(f"forbidden output field was accepted: {key}")


def test_audit_is_self_hashed_and_marker_is_last_and_immutable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "smoke"
        output.mkdir()
        audit_payload = {
            "schema": smoke.AUDIT_SCHEMA,
            "experiment": smoke.EXPERIMENT,
            "stage": smoke.STAGE,
            "status": "passed",
            "resource": {"elapsed_seconds": 1.0, "linux_peak_rss_bytes": 1024},
        }
        marker_payload = {
            "schema": smoke.PASS_SCHEMA,
            "experiment": smoke.EXPERIMENT,
            "stage": smoke.STAGE,
            "status": "passed",
        }
        audit_path, marker_path, audit_sha, marker_sha = smoke._publish_smoke_evidence(
            output, audit_payload, marker_payload
        )
        audit = smoke._authenticate_self_hashed_file(
            audit_path, expected_file_sha256=audit_sha
        )
        marker = smoke._authenticate_self_hashed_file(
            marker_path, expected_file_sha256=marker_sha
        )
        assert set(path.name for path in output.iterdir()) == {
            smoke.AUDIT_NAME,
            smoke.PASS_NAME,
        }
        assert marker_path.stat().st_mtime_ns >= audit_path.stat().st_mtime_ns
        assert marker["audit"]["sha256"] == audit_sha
        assert marker["audit"]["content_sha256"] == audit["content_sha256"]
        assert marker["write_order"].startswith("last_after")
        try:
            smoke._publish_smoke_evidence(output, audit_payload, marker_payload)
        except RuntimeError as error:
            assert "not empty" in str(error)
        else:
            raise AssertionError("immutable resource-smoke output was overwritten")


def test_self_hash_authentication_rejects_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "evidence.json"
        value = smoke._with_self_hash({"status": "passed"})
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        smoke._authenticate_self_hashed_file(path)
        value["status"] = "changed"
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        try:
            smoke._authenticate_self_hashed_file(path)
        except RuntimeError as error:
            assert "self hash" in str(error)
        else:
            raise AssertionError("tampered resource-smoke evidence was accepted")


def test_public_release_authenticator_replays_complete_two_file_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "frozen_output"
        with _temporary_frozen_output_root(root):
            audit, marker, commit = _valid_release_payloads()
            output = _release_output(root, commit=commit)
            _audit_path, pass_path, _audit_sha, pass_sha = (
                smoke._publish_smoke_evidence(output, audit, marker)
            )
            authenticated = smoke.authenticate_resource_smoke_release(
                pass_path,
                pass_sha,
                commit,
                smoke.EXPECTED_CONFIG_SHA256,
            )
            assert authenticated["status"] == "authenticated"
            assert authenticated["resource_limits_passed"] is True
            assert authenticated["reserved_family_zero_member_open"] is True
            assert authenticated["directory_exact_two_files"] is True


def test_public_release_authenticator_rejects_audit_and_resource_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "frozen_output"
        with _temporary_frozen_output_root(root):
            audit, marker, commit = _valid_release_payloads()
            output = _release_output(root, commit=commit)
            audit_path, pass_path, _audit_sha, pass_sha = (
                smoke._publish_smoke_evidence(output, audit, marker)
            )
            changed = json.loads(audit_path.read_text(encoding="utf-8"))
            changed["status"] = "changed"
            audit_path.write_text(
                json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
            )
            try:
                smoke.authenticate_resource_smoke_release(
                    pass_path, pass_sha, commit, smoke.EXPECTED_CONFIG_SHA256
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("changed detailed audit was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "frozen_output"
        with _temporary_frozen_output_root(root):
            audit, marker, commit = _valid_release_payloads()
            output = _release_output(root, commit=commit)
            _audit_path, pass_path, _audit_sha, _pass_sha = (
                smoke._publish_smoke_evidence(output, audit, marker)
            )
            changed_marker = json.loads(pass_path.read_text(encoding="utf-8"))
            changed_marker.pop("content_sha256")
            changed_marker["linux_peak_rss_bytes"] = smoke.MEMORY_LIMIT_BYTES
            changed_marker = smoke._with_self_hash(changed_marker)
            pass_path.write_text(
                json.dumps(changed_marker, sort_keys=True) + "\n", encoding="utf-8"
            )
            changed_sha = _digest(pass_path)
            try:
                smoke.authenticate_resource_smoke_release(
                    pass_path, changed_sha, commit, smoke.EXPECTED_CONFIG_SHA256
                )
            except RuntimeError as error:
                assert "peak RSS" in str(error)
            else:
                raise AssertionError("resource-limit tampering was accepted")


def test_public_release_authenticator_rejects_non_numeric_resource_observations() -> None:
    mutations = (
        ("marker", "linux_peak_rss_bytes", "2048"),
        ("marker", "elapsed_seconds", "2.0"),
        ("audit", "linux_peak_rss_bytes_before_audit_publish", "1024"),
        ("audit", "elapsed_seconds_before_audit_publish", "1.0"),
        ("marker", "linux_peak_rss_bytes", True),
        ("audit", "elapsed_seconds_before_audit_publish", True),
    )
    for target, field, forged_value in mutations:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen_output"
            with _temporary_frozen_output_root(root):
                audit, marker, commit = _valid_release_payloads()
                if target == "marker":
                    marker[field] = forged_value
                else:
                    resource = audit["resource"]
                    assert isinstance(resource, dict)
                    resource[field] = forged_value
                output = _release_output(root, commit=commit)
                _audit_path, pass_path, _audit_sha, pass_sha = (
                    smoke._publish_smoke_evidence(output, audit, marker)
                )
                try:
                    smoke.authenticate_resource_smoke_release(
                        pass_path,
                        pass_sha,
                        commit,
                        smoke.EXPECTED_CONFIG_SHA256,
                    )
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        f"non-numeric resource observation was accepted: {target}/{field}"
                    )


def test_public_release_authenticator_rejects_type_coercion_and_resource_schema_drift() -> None:
    mutations = (
        (("data_access", "population_envelope_sidecar_count"), 32.0),
        (("data_access", "member_open_counts_by_family", "f22_raptor"), 4.0),
        (("data_access", "reserved_family_zero_member_open", "half_cylinder"), 1),
        (("data_access", "reserved_parent_cache_whole_file_open_count"), False),
        (("synthetic_arithmetic_gate", "feature_width"), 165.0),
        (("constructed_model", "feature_width"), 165.0),
        (("constructed_model", "k"), 31.0),
        (("constructed_model", "strict_majority_family_count"), 2.0),
        (("runtime", "slurm_cpus_per_task"), 32.0),
        (("runtime", "slurm_num_nodes"), True),
        (("resource", "memory_limit_bytes_exclusive"), float(smoke.MEMORY_LIMIT_BYTES)),
        (("resource", "walltime_limit_seconds_inclusive"), float(smoke.WALLTIME_LIMIT_SECONDS)),
        (("resource", "unexpected_resource_field"), 0),
        (("final_gates", "synthetic_queries_only"), 1),
    )

    for field_path, forged_value in mutations:
        def forged_type(
            audit: dict[str, object],
            *,
            path: tuple[str, ...] = field_path,
            value: object = forged_value,
        ) -> None:
            target: object = audit
            for name in path[:-1]:
                assert isinstance(target, dict)
                target = target[name]
            assert isinstance(target, dict)
            target[path[-1]] = value

        _expect_self_consistent_release_rejection(forged_type)


def test_public_release_authenticator_rejects_marker_binding_and_limit_type_coercion() -> None:
    mutations = (
        ("audit_size", "string"),
        ("audit_size", "float"),
        ("marker_limit", "memory_limit_bytes_exclusive"),
        ("marker_limit", "walltime_limit_seconds_inclusive"),
    )
    for target, variant in mutations:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen_output"
            with _temporary_frozen_output_root(root):
                audit, marker, commit = _valid_release_payloads()
                if target == "marker_limit":
                    marker[variant] = float(marker[variant])
                output = _release_output(root, commit=commit)
                _audit_path, pass_path, _audit_sha, pass_sha = (
                    smoke._publish_smoke_evidence(output, audit, marker)
                )
                if target == "audit_size":
                    changed = json.loads(pass_path.read_text(encoding="utf-8"))
                    changed.pop("content_sha256")
                    audit_binding = changed["audit"]
                    assert isinstance(audit_binding, dict)
                    size = int(audit_binding["size_bytes"])
                    audit_binding["size_bytes"] = (
                        str(size) if variant == "string" else float(size)
                    )
                    changed = smoke._with_self_hash(changed)
                    pass_path.write_text(
                        json.dumps(changed, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    pass_sha = _digest(pass_path)
                try:
                    smoke.authenticate_resource_smoke_release(
                        pass_path,
                        pass_sha,
                        commit,
                        smoke.EXPECTED_CONFIG_SHA256,
                    )
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(
                        f"marker type coercion was accepted: {target}/{variant}"
                    )


def test_public_release_authenticator_rejects_self_consistent_forged_source_hash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "frozen_output"
        with _temporary_frozen_output_root(root):
            audit, marker, commit = _valid_release_payloads()
            source = audit["production_source_identity"]
            assert isinstance(source, dict)
            files = source["files"]
            assert isinstance(files, dict)
            target = smoke.CONFIG_PATH.relative_to(smoke.ROOT).as_posix()
            replacement = "f" * 64
            if files[target] == replacement:
                replacement = "e" * 64
            files[target] = replacement
            source["files_content_sha256"] = smoke.canonical_json_sha256(files)
            output = _release_output(root, commit=commit)
            _audit_path, pass_path, _audit_sha, pass_sha = (
                smoke._publish_smoke_evidence(output, audit, marker)
            )
            try:
                smoke.authenticate_resource_smoke_release(
                    pass_path, pass_sha, commit, smoke.EXPECTED_CONFIG_SHA256
                )
            except RuntimeError as error:
                assert "current frozen checkout" in str(error)
            else:
                raise AssertionError("self-consistent forged source hash was accepted")


def test_public_release_authenticator_cross_binds_constructed_and_query_scale_sets() -> None:
    def forged_query_sha(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["natural_raw_present_exact_scale_ids_sha256"] = "0" * 64

    def forged_query_counts(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["query_row_count"] = 3
        query["natural_raw_present_exact_scale_count"] = 3
        query["effective_retained_exact_scale_count"] = 2
        query["natural_raw_only_no_scaler_exact_scale_count"] = 1
        query["joint_family_count_histogram"] = {
            "0": 2,
            "1": 0,
            "2": 0,
            "3": 1,
        }

    def forged_query_field_set(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["unexpected_scale_set_claim"] = True

    def forged_constructed_field_set(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        constructed["unexpected_scale_set_claim"] = True

    for mutation in (
        forged_query_sha,
        forged_query_counts,
        forged_query_field_set,
        forged_constructed_field_set,
    ):
        _expect_self_consistent_release_rejection(mutation)


def test_public_release_authenticator_cross_binds_fit_population_and_query_support_arithmetic() -> None:
    def forged_natural_raw_total(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        constructed["natural_raw_family_class_row_count"] = (
            int(constructed["natural_raw_family_class_row_count"]) + 1
        )

    def forged_fit_negative_total(audit: dict[str, object]) -> None:
        fit_rows = audit["fit_input_rows"]
        assert isinstance(fit_rows, list)
        row = fit_rows[0]
        assert isinstance(row, dict)
        row["negative_row_count"] = int(row["negative_row_count"]) + 1
        row["positive_row_count"] = int(row["positive_row_count"]) - 1

    def forged_joint_histogram(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        histogram = query["joint_family_count_histogram"]
        assert isinstance(histogram, dict)
        histogram["0"] = int(histogram["0"]) - 1
        histogram["2"] = int(histogram["2"]) + 1

    def forged_retrieval_below_joint(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["strict_majority_retrieval_supported_row_count"] = 0

    def forged_retrieval_above_effective(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["strict_majority_retrieval_supported_row_count"] = 2

    def forged_joint_above_effective(audit: dict[str, object]) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["strict_majority_joint_supported_row_count"] = 2
        query["strict_majority_retrieval_supported_row_count"] = 2
        query["joint_family_count_histogram"] = {
            "0": 0,
            "1": 0,
            "2": 1,
            "3": 1,
        }

    def forged_raw_only_query_with_nonzero_family_support(
        audit: dict[str, object],
    ) -> None:
        query = audit["synthetic_query_path"]
        assert isinstance(query, dict)
        query["joint_family_count_histogram"] = {
            "0": 0,
            "1": 1,
            "2": 0,
            "3": 1,
        }

    def forged_scale_domain_above_2000(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        query = audit["synthetic_query_path"]
        assert isinstance(constructed, dict) and isinstance(query, dict)
        for value in (constructed, query):
            value["natural_raw_present_exact_scale_count"] = smoke.SCALE_COUNT + 1
            value["effective_retained_exact_scale_count"] = smoke.SCALE_COUNT
            value["natural_raw_only_no_scaler_exact_scale_count"] = 1
        query["query_row_count"] = smoke.SCALE_COUNT + 1
        query["joint_family_count_histogram"] = {
            "0": smoke.SCALE_COUNT,
            "1": 0,
            "2": 0,
            "3": 1,
        }

    def forged_raw_only_scale_count_above_discarded_rows(
        audit: dict[str, object],
    ) -> None:
        constructed = audit["constructed_model"]
        query = audit["synthetic_query_path"]
        assert isinstance(constructed, dict) and isinstance(query, dict)
        for value in (constructed, query):
            value["natural_raw_present_exact_scale_count"] = 10
            value["effective_retained_exact_scale_count"] = 1
            value["natural_raw_only_no_scaler_exact_scale_count"] = 9
        query["query_row_count"] = 10
        query["joint_family_count_histogram"] = {
            "0": 9,
            "1": 0,
            "2": 0,
            "3": 1,
        }

    def forged_effective_scale_count_above_negative_rows(
        audit: dict[str, object],
    ) -> None:
        constructed = audit["constructed_model"]
        query = audit["synthetic_query_path"]
        assert isinstance(constructed, dict) and isinstance(query, dict)
        shared_negative = int(constructed["shared_negative_row_count"])
        for value in (constructed, query):
            value["natural_raw_present_exact_scale_count"] = shared_negative + 2
            value["effective_retained_exact_scale_count"] = shared_negative + 1
            value["natural_raw_only_no_scaler_exact_scale_count"] = 1
        query["query_row_count"] = shared_negative + 2
        query["joint_family_count_histogram"] = {
            "0": shared_negative + 1,
            "1": 0,
            "2": 0,
            "3": 1,
        }

    for mutation in (
        forged_natural_raw_total,
        forged_fit_negative_total,
        forged_joint_histogram,
        forged_retrieval_below_joint,
        forged_retrieval_above_effective,
        forged_joint_above_effective,
        forged_raw_only_query_with_nonzero_family_support,
        forged_scale_domain_above_2000,
        forged_raw_only_scale_count_above_discarded_rows,
        forged_effective_scale_count_above_negative_rows,
    ):
        _expect_self_consistent_release_rejection(mutation)


def test_public_release_authenticator_cross_binds_per_family_fit_class_counts() -> None:
    def forged_cross_family_class_exchange(audit: dict[str, object]) -> None:
        fit_rows = audit["fit_input_rows"]
        assert isinstance(fit_rows, list)
        first_family_row = fit_rows[0]
        second_family_row = fit_rows[4]
        assert isinstance(first_family_row, dict)
        assert isinstance(second_family_row, dict)
        assert first_family_row["physical_family"] != second_family_row["physical_family"]
        first_family_row["negative_row_count"] = (
            int(first_family_row["negative_row_count"]) + 1
        )
        first_family_row["positive_row_count"] = (
            int(first_family_row["positive_row_count"]) - 1
        )
        second_family_row["negative_row_count"] = (
            int(second_family_row["negative_row_count"]) - 1
        )
        second_family_row["positive_row_count"] = (
            int(second_family_row["positive_row_count"]) + 1
        )

    _expect_self_consistent_release_rejection(forged_cross_family_class_exchange)


def test_constructed_class_supported_scale_counts_require_minimum_rows() -> None:
    unsupported_nonempty_loo = _constructed_model_audit_fixture()
    family_audits = unsupported_nonempty_loo[
        "class_library_and_reference_audits"
    ]
    assert isinstance(family_audits, dict)
    class_record = family_audits[smoke.FIT_FAMILIES[0]]["positive"]
    class_record["loo_supported_scale_count_k31"] = 0

    impossible_library = _constructed_model_audit_fixture()
    family_audits = impossible_library["class_library_and_reference_audits"]
    assert isinstance(family_audits, dict)
    class_record = family_audits[smoke.FIT_FAMILIES[0]]["positive"]
    class_record["library_supported_scale_count_k31"] = 2
    impossible_library["natural_raw_present_exact_scale_count"] = 2
    impossible_library["effective_retained_exact_scale_count"] = 2
    impossible_library["natural_raw_only_no_scaler_exact_scale_count"] = 0

    impossible_loo = _constructed_model_audit_fixture()
    family_audits = impossible_loo["class_library_and_reference_audits"]
    assert isinstance(family_audits, dict)
    class_record = family_audits[smoke.FIT_FAMILIES[0]]["positive"]
    class_record["library_row_count"] = 63
    class_record["library_supported_scale_count_k31"] = 2
    class_record["loo_supported_scale_count_k31"] = 2
    impossible_loo["effective_retained_family_class_library_row_count"] = (
        int(impossible_loo["effective_retained_family_class_library_row_count"])
        + 31
    )
    impossible_loo["full_family_class_library_row_count"] = int(
        impossible_loo["full_family_class_library_row_count"]
    ) + 31
    impossible_loo["natural_raw_present_exact_scale_count"] = 2
    impossible_loo["effective_retained_exact_scale_count"] = 2
    impossible_loo["natural_raw_only_no_scaler_exact_scale_count"] = 0
    arrays = impossible_loo["family_class_library_and_calibration_arrays"]
    assert isinstance(arrays, dict)
    member_name = "calibrator_f0_c1__negative_features"
    arrays[member_name] = _array_audit_record(
        f"calibrator/{member_name}",
        "<f4",
        (63, smoke.FEATURE_WIDTH),
    )

    for constructed in (
        unsupported_nonempty_loo,
        impossible_library,
        impossible_loo,
    ):
        try:
            smoke._validate_constructed_model_artifact_audit(constructed)
        except RuntimeError as error:
            assert "class audit counts/hash" in str(error)
        else:
            raise AssertionError("impossible supported-scale cardinality was accepted")


def test_public_release_authenticator_rejects_incomplete_constructed_artifact_audits() -> None:
    def empty_scaler(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        constructed["scaler_arrays"] = {}

    def deleted_calibrator_member(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        arrays = constructed["family_class_library_and_calibration_arrays"]
        assert isinstance(arrays, dict)
        arrays.pop("class_scale_counts_int64")

    def invalid_array_byte_arithmetic(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        arrays = constructed["scaler_arrays"]
        assert isinstance(arrays, dict)
        record = arrays["local_mean_float64"]
        assert isinstance(record, dict)
        record["size_bytes"] = int(record["size_bytes"]) + 1

    def empty_class_family(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        classes = constructed["class_library_and_reference_audits"]
        assert isinstance(classes, dict)
        classes[smoke.FIT_FAMILIES[0]] = {}

    def forged_all_serialization_versions(audit: dict[str, object]) -> None:
        constructed = audit["constructed_model"]
        assert isinstance(constructed, dict)
        arrays = constructed["family_class_library_and_calibration_arrays"]
        assert isinstance(arrays, dict)
        forged = smoke.canonical_array_sha256(np.asarray(2, dtype=np.int16))
        arrays["serialization_version_int16"]["sha256"] = forged
        for family_index in range(len(smoke.FIT_FAMILIES)):
            for class_index in range(len(smoke.CLASS_NAMES)):
                arrays[
                    f"calibrator_f{family_index}_c{class_index}__serialization_version"
                ]["sha256"] = forged

    for mutation in (
        empty_scaler,
        deleted_calibrator_member,
        invalid_array_byte_arithmetic,
        empty_class_family,
        forged_all_serialization_versions,
    ):
        _expect_self_consistent_release_rejection(mutation)


def test_public_release_authenticator_binds_all_frozen_evidence_identities() -> None:
    def forged_path(audit: dict[str, object]) -> None:
        evidence = audit["authenticated_evidence"]
        assert isinstance(evidence, dict)
        value = evidence["kinematic_input_manifest"]
        assert isinstance(value, dict)
        value["path"] = str(value["path"]) + ".forged"

    def forged_sha(audit: dict[str, object]) -> None:
        evidence = audit["authenticated_evidence"]
        assert isinstance(evidence, dict)
        value = evidence["parent_synthetic_pass"]
        assert isinstance(value, dict)
        value["sha256"] = "0" * 64

    def forged_frozen_size(audit: dict[str, object]) -> None:
        evidence = audit["authenticated_evidence"]
        assert isinstance(evidence, dict)
        value = evidence["train_cache_input_manifest"]
        assert isinstance(value, dict)
        value["size_bytes"] = int(value["size_bytes"]) + 1

    for mutation in (forged_path, forged_sha, forged_frozen_size):
        _expect_self_consistent_release_rejection(mutation)


def test_public_release_authenticator_reauthenticates_all_four_persisted_evidence_files() -> None:
    evidence_names = (
        "train_cache_input_manifest",
        "kinematic_input_manifest",
        "parent_synthetic_pass",
        "sealed_sidecar_population",
    )
    for evidence_name in evidence_names:
        for field in ("path", "sha256", "size_bytes"):
            def forged_identity(
                audit: dict[str, object],
                *,
                name: str = evidence_name,
                identity_field: str = field,
            ) -> None:
                evidence = audit["authenticated_evidence"]
                assert isinstance(evidence, dict)
                value = evidence[name]
                assert isinstance(value, dict)
                if identity_field == "path":
                    value[identity_field] = str(value[identity_field]) + ".forged"
                elif identity_field == "sha256":
                    value[identity_field] = "0" * 64
                else:
                    value[identity_field] = int(value[identity_field]) + 1

            _expect_self_consistent_release_rejection(forged_identity)


def test_public_release_authenticator_requires_frozen_output_parent_and_basename() -> None:
    def unchanged(_audit: dict[str, object]) -> None:
        return None

    _expect_self_consistent_release_rejection(unchanged, path_mode="wrong_basename")
    _expect_self_consistent_release_rejection(unchanged, path_mode="outside_root")


def test_public_release_authenticator_requires_exact_config_path() -> None:
    def forged_config_path(audit: dict[str, object]) -> None:
        config = audit["config"]
        assert isinstance(config, dict)
        config["path"] = str(smoke.CONFIG_PATH) + ".forged"

    _expect_self_consistent_release_rejection(forged_config_path)


def test_public_release_authenticator_cross_binds_all_fit_access_identities() -> None:
    def forged_fit_sidecar_sha(audit: dict[str, object]) -> None:
        fit_rows = audit["fit_input_rows"]
        assert isinstance(fit_rows, list)
        sidecar = fit_rows[0]["sidecar"]
        assert isinstance(sidecar, dict)
        sidecar["sha256"] = "0" * 64

    def forged_fit_archive_members(audit: dict[str, object]) -> None:
        fit_rows = audit["fit_input_rows"]
        assert isinstance(fit_rows, list)
        fit_rows[0]["parent_archive_members"] = ["fmt_features"]

    def forged_but_cross_matched_duplicate_source_ordinal(
        audit: dict[str, object],
    ) -> None:
        access = audit["data_access"]
        fit_rows = audit["fit_input_rows"]
        assert isinstance(access, dict) and isinstance(fit_rows, list)
        opened_rows = access["opened_fit_rows"]
        assert isinstance(opened_rows, list)
        opened_rows[1]["source_ordinal"] = 0
        fit_rows[1]["source_ordinal"] = 0

    def forged_parent_identity_in_both_audit_copies(
        audit: dict[str, object],
    ) -> None:
        access = audit["data_access"]
        fit_rows = audit["fit_input_rows"]
        assert isinstance(access, dict) and isinstance(fit_rows, list)
        opened_rows = access["opened_fit_rows"]
        assert isinstance(opened_rows, list)
        for row in (opened_rows[0], fit_rows[0]):
            parent = row["parent_cache"]
            assert isinstance(parent, dict)
            parent["sha256"] = "0" * 64

    def forged_sidecar_identity_in_both_audit_copies(
        audit: dict[str, object],
    ) -> None:
        access = audit["data_access"]
        fit_rows = audit["fit_input_rows"]
        assert isinstance(access, dict) and isinstance(fit_rows, list)
        opened_rows = access["opened_fit_rows"]
        assert isinstance(opened_rows, list)
        for row in (opened_rows[0], fit_rows[0]):
            sidecar = row["sidecar"]
            assert isinstance(sidecar, dict)
            sidecar["size_bytes"] = int(sidecar["size_bytes"]) + 1

    for mutation in (
        forged_fit_sidecar_sha,
        forged_fit_archive_members,
        forged_but_cross_matched_duplicate_source_ordinal,
        forged_parent_identity_in_both_audit_copies,
        forged_sidecar_identity_in_both_audit_copies,
    ):
        _expect_self_consistent_release_rejection(mutation)
