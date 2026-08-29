#!/usr/bin/env python3
"""Build parallel cache shards or evaluate mainExp_TemplateMatching_2.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.phase21_pipeline import (
    build_phase21_cache_slice,
    authorize_phase31_portable_population_marker_path,
    cache_summary_payload,
    configure_deterministic_execution,
    discover_phase21_cache_sidecars,
    evaluate_phase21_caches,
    audit_phase31_train_coverage,
    _atomic_bytes,
    _atomic_json,
    load_cache_summary_sidecar,
    load_phase21_plan,
    load_phase31_plan,
    recover_phase21_cache_summary,
    validate_phase31_synthetic_pass,
    validate_phase31_train_coverage_pass,
    validate_phase31_cache_portable_population_evidence,
    write_cache_summary_sidecar,
)
from pathline_template_matching.portable_flow import (
    canonical_json_sha256,
    load_portable_flow_window,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "config/mainExp_TemplateMatching_2.1.yaml"
DATASET_MANIFEST_SCHEMA = (
    "pathline_template_matching.portable_flow_dataset_manifest.v1"
)


def _is_lower_hex(value: Any, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


class PortableManifestIndex:
    """Strict index over the four staged windows for each frozen dataset."""

    def __init__(
        self,
        path: Path | None,
        *,
        portable_root: Path | None = None,
        expected_experiment: str,
        expected_config_sha256: str,
        expected_dataset_registry_sha256: str,
        expected_datasets: tuple[str, ...],
        expected_family_by_dataset: Mapping[str, str],
        expected_split_by_dataset: Mapping[str, str],
        expected_builder_git_commit: str,
        expected_window_frame_count: int = 13,
        expected_source_kind_by_dataset: Mapping[str, str] | None = None,
    ) -> None:
        self.portable_root = portable_root.resolve() if portable_root else None
        self.expected_window_frame_count = int(expected_window_frame_count)
        if self.expected_window_frame_count < 2:
            raise ValueError("portable window frame count must be at least two")
        if path is None:
            if self.portable_root is None:
                raise ValueError("either portable manifest or portable root is required")
            # Resolve only the caller-authorized datasets.  In the 3.1
            # train-only phase this prevents even opening test manifests.
            manifest_paths = [
                self.portable_root / dataset / "manifest.json"
                for dataset in expected_datasets
            ]
            missing_manifests = [
                value for value in manifest_paths if not value.is_file()
            ]
            if missing_manifests:
                raise FileNotFoundError(
                    "portable dataset manifests are missing: "
                    f"{[str(value) for value in missing_manifests]}"
                )
        else:
            resolved = path.resolve()
            manifest_paths = (
                sorted(resolved.glob("*/manifest.json"))
                if resolved.is_dir()
                else [resolved]
            )
        if not manifest_paths:
            raise FileNotFoundError("no portable dataset manifest was found")
        self.rows: list[dict[str, Any]] = []
        seen_datasets: set[str] = set()
        for manifest_path in manifest_paths:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema") != DATASET_MANIFEST_SCHEMA:
                raise ValueError(
                    f"production accepts only {DATASET_MANIFEST_SCHEMA}: {manifest_path}"
                )
            claimed = payload.get("manifest_content_sha256")
            digest_payload = dict(payload)
            digest_payload.pop("manifest_content_sha256", None)
            from pathline_template_matching.portable_flow import canonical_json_sha256

            if not _is_lower_hex(claimed, 64) or claimed != canonical_json_sha256(
                digest_payload
            ):
                raise ValueError(f"dataset manifest content SHA-256 mismatch: {manifest_path}")
            required_top_level = {
                "experiment",
                "config_sha256",
                "dataset_registry_sha256",
                "builder_git_commit",
                "dataset",
                "physical_family",
                "split",
                "source_kind",
                "source_file",
                "source_file_size",
                "source_file_sha256",
                "source_total_frames",
                "selected_source_indices",
                "window_count",
                "windows",
            }
            missing = required_top_level.difference(payload)
            if missing:
                raise ValueError(
                    f"dataset manifest misses frozen fields {sorted(missing)}: {manifest_path}"
                )
            dataset = str(payload["dataset"])
            expected_top = {
                "experiment": expected_experiment,
                "config_sha256": expected_config_sha256,
                "dataset_registry_sha256": expected_dataset_registry_sha256,
                "builder_git_commit": expected_builder_git_commit,
                "physical_family": expected_family_by_dataset.get(dataset),
                "split": expected_split_by_dataset.get(dataset),
            }
            if expected_source_kind_by_dataset is not None:
                expected_top["source_kind"] = expected_source_kind_by_dataset.get(dataset)
            top_drift = {
                key: (payload.get(key), expected)
                for key, expected in expected_top.items()
                if payload.get(key) != expected
            }
            if top_drift:
                raise ValueError(
                    f"dataset manifest differs from frozen provenance in {manifest_path}: "
                    f"{top_drift}"
                )
            if dataset not in expected_datasets:
                raise ValueError(f"unexpected dataset manifest: {dataset}")
            if dataset in seen_datasets:
                raise ValueError(f"duplicate dataset manifest: {dataset}")
            seen_datasets.add(dataset)
            if (
                not isinstance(payload["source_file"], str)
                or not payload["source_file"]
                or int(payload["source_file_size"]) < 1
                or not _is_lower_hex(payload["source_file_sha256"], 64)
                or int(payload["source_total_frames"]) < 1
            ):
                raise ValueError(f"dataset manifest source identity is invalid: {manifest_path}")
            windows = payload["windows"]
            if not isinstance(windows, list) or not all(
                isinstance(row, Mapping) for row in windows
            ):
                raise ValueError(f"dataset manifest windows are invalid: {manifest_path}")
            if int(payload["window_count"]) != len(windows):
                raise ValueError(f"dataset manifest window count mismatch: {manifest_path}")
            selected = [int(value) for value in payload["selected_source_indices"]]
            row_indices = [int(row["source_start_index"]) for row in windows]
            if selected != row_indices:
                raise ValueError(f"dataset manifest source-index list mismatch: {manifest_path}")
            frozen_fields = {
                "dataset": dataset,
                "experiment": expected_experiment,
                "config_sha256": expected_config_sha256,
                "physical_family": str(payload["physical_family"]),
                "split": str(payload["split"]),
                "dataset_registry_sha256": str(payload["dataset_registry_sha256"]),
                "builder_git_commit": str(payload["builder_git_commit"]),
                "source_total_frames": int(payload["source_total_frames"]),
                "source_file": str(payload["source_file"]),
                "source_file_size": int(payload["source_file_size"]),
                "source_file_sha256": str(payload["source_file_sha256"]),
                "source_kind": str(payload["source_kind"]),
            }
            for ordinal, raw_row in enumerate(windows):
                row = dict(raw_row)
                drift = {
                    key: (row.get(key), expected)
                    for key, expected in frozen_fields.items()
                    if row.get(key) != expected
                }
                if drift:
                    raise ValueError(
                        f"portable row/top-level provenance mismatch in {manifest_path}: {drift}"
                    )
                if int(row.get("source_ordinal", -1)) != ordinal:
                    raise ValueError(f"portable source ordinal mismatch in {manifest_path}")
                if int(row.get("frame_count", -1)) != self.expected_window_frame_count:
                    raise ValueError(f"portable window frame count mismatch in {manifest_path}")
                if not _is_lower_hex(row.get("file_sha256"), 64):
                    raise ValueError(f"portable window has no valid file SHA-256: {manifest_path}")
                if int(row.get("file_size", -1)) < 1:
                    raise ValueError(f"portable window file size is invalid: {manifest_path}")
                relative = row.get("relative_path")
                relative_path = Path(str(relative)) if relative is not None else None
                if (
                    relative_path is None
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or relative_path.name != str(relative_path)
                ):
                    raise ValueError(f"portable relative_path is unsafe: {manifest_path}")
                row["__manifest_dir"] = str(manifest_path.parent.resolve())
                row["__manifest_path"] = str(manifest_path.resolve())
                row["__manifest_file_sha256"] = sha256_file(manifest_path)
                self.rows.append(row)
        found_datasets = {str(row.get("dataset")) for row in self.rows}
        if found_datasets != set(expected_datasets):
            raise ValueError(
                "portable manifest population does not exactly cover the frozen ten flows: "
                f"missing={sorted(set(expected_datasets)-found_datasets)}, "
                f"extra={sorted(found_datasets-set(expected_datasets))}"
            )
        for dataset in expected_datasets:
            dataset_rows = self.dataset_rows(dataset)
            for row in dataset_rows:
                if row.get("physical_family") != expected_family_by_dataset[dataset]:
                    raise ValueError(f"portable physical family differs from config: {dataset}")
                if row.get("split") != expected_split_by_dataset[dataset]:
                    raise ValueError(f"portable split differs from config: {dataset}")
        registry_hashes = {str(row.get("dataset_registry_sha256")) for row in self.rows}
        builder_commits = {str(row.get("builder_git_commit")) for row in self.rows}
        if registry_hashes != {expected_dataset_registry_sha256}:
            raise ValueError(
                "portable dataset-registry SHA-256 differs from the committed config registry: "
                f"portable={sorted(registry_hashes)}, expected={expected_dataset_registry_sha256}"
            )
        if builder_commits != {expected_builder_git_commit}:
            raise ValueError(
                "portable manifests were not staged from the current clean Git commit: "
                f"portable={sorted(builder_commits)}, current={expected_builder_git_commit}"
            )

    def dataset_rows(self, dataset: str) -> list[dict[str, Any]]:
        rows = [row for row in self.rows if str(row.get("dataset")) == dataset]
        rows.sort(key=lambda row: int(row["source_start_index"]))
        if len(rows) != 4:
            raise ValueError(f"portable manifest has {len(rows)} windows for {dataset}, expected 4")
        indices = [int(row["source_start_index"]) for row in rows]
        if len(set(indices)) != 4:
            raise ValueError(f"portable manifest source indices are duplicated for {dataset}")
        return rows

    def native_frame_count(self, dataset: str) -> int:
        rows = self.dataset_rows(dataset)
        explicit = {
            int(row[key])
            for row in rows
            for key in (
                "source_total_frames",
                "native_frame_count",
                "raw_frame_count",
                "source_frame_count",
            )
            if row.get(key) is not None
        }
        if len(explicit) > 1:
            raise ValueError(f"portable manifest gives contradictory frame counts for {dataset}")
        # The frozen fourth source index is exactly T-window, so it supplies a
        # deterministic fallback when the staging manifest omits native T.
        return (
            next(iter(explicit))
            if explicit
            else int(rows[-1]["source_start_index"])
            + self.expected_window_frame_count
        )

    def row(self, dataset: str, source_index: int) -> dict[str, Any]:
        matches = [
            row
            for row in self.dataset_rows(dataset)
            if int(row["source_start_index"]) == int(source_index)
        ]
        if len(matches) != 1:
            raise ValueError(f"portable manifest does not uniquely resolve {dataset}/{source_index}")
        return matches[0]

    def resolve_path(self, row: Mapping[str, Any]) -> Path:
        raw = row.get("relative_path", row.get("path", row.get("portable_path")))
        if raw is None:
            raise ValueError("portable manifest row has no relative_path or path")
        source = Path(str(raw))
        candidates: list[Path] = []
        if row.get("relative_path") is not None and row.get("__manifest_dir") is not None:
            candidates.append(Path(str(row["__manifest_dir"])) / str(row["relative_path"]))
        elif source.is_absolute():
            candidates.append(source)
        else:
            manifest_dir = row.get("__manifest_dir")
            if manifest_dir is not None:
                candidates.append(Path(str(manifest_dir)) / source)
        if self.portable_root is not None:
            relative = row.get("relative_path")
            if relative is not None:
                candidates.append(self.portable_root / str(relative))
            candidates.append(self.portable_root / str(row["dataset"]) / source.name)
            candidates.append(self.portable_root / source.name)
        existing = list(
            dict.fromkeys(candidate.resolve() for candidate in candidates if candidate.is_file())
        )
        if len(existing) != 1:
            raise FileNotFoundError(
                f"portable path resolution for {row.get('dataset')}/{row.get('source_start_index')} "
                f"found {existing}; candidates were {[str(value) for value in candidates]}"
            )
        return existing[0]


def _git_commit_and_clean(
    expected_experiment: str = "mainExp_TemplateMatching_2.1",
) -> str:
    if expected_experiment == "mainExp_TemplateMatching_3.1":
        entry_script = "scripts/run_mainexp_template_matching_3_1.py"
        config_path = "config/mainExp_TemplateMatching_3.1.yaml"
    elif expected_experiment == "mainExp_TemplateMatching_2.1":
        entry_script = "scripts/run_mainexp_template_matching_2_1.py"
        config_path = "config/mainExp_TemplateMatching_2.1.yaml"
    else:
        raise ValueError(f"unsupported production experiment {expected_experiment}")
    critical_paths = (
        entry_script,
        "scripts/run_mainexp_template_matching_2_1.py",
        "src/pathline_template_matching/phase21_pipeline.py",
        "src/pathline_template_matching/arc_length_primitives.py",
        "src/pathline_template_matching/portable_flow.py",
        "scripts/verify_long_arc_horizon_1_1.py",
        "config/Verify_LongArcHorizon_1.1.yaml",
        config_path,
        "config/datasets.yaml",
    )
    for relative in critical_paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "working tree contains tracked or untracked changes; "
            "Ibex runs require a clean committed revision"
        )
    return commit


def _load_plan(path: Path, expected_experiment: str):
    if expected_experiment == "mainExp_TemplateMatching_3.1":
        return load_phase31_plan(path)
    if expected_experiment == "mainExp_TemplateMatching_2.1":
        return load_phase21_plan(path)
    raise ValueError(f"unsupported production experiment {expected_experiment}")


def _portable_population_marker_name(access_scope: str) -> str:
    if access_scope == "train-only":
        return "TRAIN_PORTABLES_PASS.json"
    if access_scope == "all":
        return "ALL_PORTABLES_PASS.json"
    raise ValueError("portable population scope must be train-only or all")


def _validate_portable_population_pass(
    plan,
    marker_path: Path,
    *,
    access_scope: str,
    portable_root: Path,
    current_git_commit: str,
    synthetic_evidence: Mapping[str, Any],
    coverage_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Authenticate a completed portable population without opening any NPZ."""

    source = marker_path.resolve()
    expected_name = _portable_population_marker_name(access_scope)
    if source.name != expected_name:
        raise ValueError(f"portable population marker must be named {expected_name}")
    entries = list(source.parent.iterdir())
    if any(not entry.is_file() for entry in entries) or {
        entry.name for entry in entries
    } != {expected_name}:
        raise ValueError("portable population marker directory has an unexpected file set")
    marker = json.loads(source.read_text(encoding="utf-8"))
    expected_datasets = (
        plan.train_datasets if access_scope == "train-only" else plan.datasets
    )
    expected = {
        "schema": "pathline_template_matching.phase31_portable_population_pass.v1",
        "experiment": plan.experiment,
        "status": "passed",
        "access_scope": access_scope,
        "git_commit": current_git_commit,
        "worktree_clean": True,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "portable_root": str(portable_root.resolve()),
        "dataset_count": len(expected_datasets),
        "window_count": len(expected_datasets) * plan.source_count,
        "synthetic_pass_file_sha256": synthetic_evidence["file_sha256"],
        "train_coverage_pass_file_sha256": (
            None if coverage_evidence is None else coverage_evidence["file_sha256"]
        ),
    }
    drift = {
        name: (marker.get(name), value)
        for name, value in expected.items()
        if marker.get(name) != value
    }
    if drift:
        raise ValueError(f"portable population marker changed: {drift}")
    rows = marker.get("rows")
    if not isinstance(rows, list) or len(rows) != expected["window_count"]:
        raise ValueError("portable population marker row count is incomplete")
    if marker.get("rows_content_sha256") != canonical_json_sha256(rows):
        raise ValueError("portable population marker row SHA-256 changed")
    expected_keys = {
        (dataset, ordinal)
        for dataset in expected_datasets
        for ordinal in range(plan.source_count)
    }
    found_keys = {
        (str(row.get("dataset")), int(row.get("source_ordinal", -1)))
        for row in rows
    }
    if found_keys != expected_keys or len(found_keys) != len(rows):
        raise ValueError("portable population marker dataset/ordinal coverage changed")
    for row in rows:
        if (
            str(row.get("split")) != plan.split_for(str(row.get("dataset")))
            or int(row.get("file_size", -1)) < 1
            or not _is_lower_hex(row.get("file_sha256"), 64)
            or not _is_lower_hex(row.get("manifest_file_sha256"), 64)
            or not _is_lower_hex(row.get("portable_metadata_sha256"), 64)
        ):
            raise ValueError("portable population marker row identity is invalid")
    return {
        "path": str(source),
        "file_size": int(source.stat().st_size),
        "file_sha256": sha256_file(source),
        "access_scope": access_scope,
        "rows_content_sha256": str(marker["rows_content_sha256"]),
        "synthetic_pass_file_sha256": synthetic_evidence["file_sha256"],
        "train_coverage_pass_file_sha256": expected[
            "train_coverage_pass_file_sha256"
        ],
        "rows": [dict(row) for row in rows],
    }


def _portable_manifest_index(plan, args, commit: str, authorized_datasets):
    registry = yaml.safe_load(plan.dataset_registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping) or not isinstance(registry.get("datasets"), list):
        raise ValueError("committed dataset registry has an invalid structure")
    registry_rows = {str(row["id"]): str(row["kind"]) for row in registry["datasets"]}
    if set(registry_rows) != set(plan.datasets):
        raise ValueError("committed dataset registry does not exactly cover the frozen split")
    index = PortableManifestIndex(
        args.portable_manifest,
        portable_root=args.portable_root,
        expected_experiment=plan.experiment,
        expected_config_sha256=plan.config_sha256,
        expected_dataset_registry_sha256=plan.dataset_registry_sha256,
        expected_datasets=authorized_datasets,
        expected_family_by_dataset=plan.family_by_dataset,
        expected_split_by_dataset={dataset: plan.split_for(dataset) for dataset in plan.datasets},
        expected_builder_git_commit=commit,
        expected_window_frame_count=plan.window_frame_count,
        expected_source_kind_by_dataset=registry_rows,
    )
    for dataset in authorized_datasets:
        expected_indices = plan.source_indices(index.native_frame_count(dataset))
        found_indices = tuple(
            int(row["source_start_index"]) for row in index.dataset_rows(dataset)
        )
        if expected_indices != found_indices:
            raise ValueError(
                f"portable windows do not follow the frozen source formula for "
                f"{dataset}: expected {expected_indices}, found {found_indices}"
            )
    return index


def _preflight_portables(args: argparse.Namespace) -> None:
    plan = _load_plan(args.config, args.expected_experiment)
    if plan.experiment != "mainExp_TemplateMatching_3.1":
        raise ValueError("portable population preflight is available only for mainExp 3.1")
    commit = _git_commit_and_clean(args.expected_experiment)
    if args.synthetic_pass is None:
        raise ValueError("3.1 portable preflight requires --synthetic-pass")
    synthetic = validate_phase31_synthetic_pass(
        plan,
        args.synthetic_pass,
        verify_config_path=args.verify_config,
        current_git_commit=commit,
    )
    if args.access_scope not in {"train-only", "all"}:
        raise ValueError("portable preflight requires --access-scope train-only or all")
    coverage = None
    if args.access_scope == "all":
        if args.train_coverage_pass is None:
            raise ValueError("all-portable preflight requires --train-coverage-pass")
        coverage = validate_phase31_train_coverage_pass(
            plan,
            args.train_coverage_pass,
            synthetic_pass_path=args.synthetic_pass,
            verify_config_path=args.verify_config,
            current_git_commit=commit,
        )
    if args.portable_root is None or args.portable_manifest is not None:
        raise ValueError("3.1 portable preflight requires only --portable-root")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"immutable portable preflight directory exists: {run_dir}")
    authorized = plan.train_datasets if args.access_scope == "train-only" else plan.datasets
    index = _portable_manifest_index(plan, args, commit, authorized)
    rows = []
    for dataset in authorized:
        for row in index.dataset_rows(dataset):
            path = index.resolve_path(row)
            if (
                path.stat().st_size != int(row["file_size"])
                or sha256_file(path) != str(row["file_sha256"])
            ):
                raise ValueError(f"portable population file size/hash changed: {path}")
            loaded = load_portable_flow_window(
                path,
                expected_dataset=dataset,
                expected_experiment=plan.experiment,
                expected_config_sha256=plan.config_sha256,
                expected_source_start_index=int(row["source_start_index"]),
                expected_file_sha256=str(row["file_sha256"]),
                expected_dataset_registry_sha256=plan.dataset_registry_sha256,
                expected_builder_git_commit=commit,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "split": plan.split_for(dataset),
                    "source_ordinal": int(row["source_ordinal"]),
                    "source_start_index": int(row["source_start_index"]),
                    "relative_path": str(path.relative_to(args.portable_root.resolve())),
                    "file_size": int(path.stat().st_size),
                    "file_sha256": str(row["file_sha256"]),
                    "manifest_relative_path": str(
                        Path(str(row["__manifest_path"])).relative_to(
                            args.portable_root.resolve()
                        )
                    ),
                    "manifest_file_sha256": str(row["__manifest_file_sha256"]),
                    "portable_metadata_sha256": canonical_json_sha256(loaded.metadata),
                }
            )
    run_dir.mkdir(parents=True, exist_ok=False)
    marker = {
        "schema": "pathline_template_matching.phase31_portable_population_pass.v1",
        "experiment": plan.experiment,
        "status": "passed",
        "access_scope": args.access_scope,
        "git_commit": commit,
        "worktree_clean": True,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "portable_root": str(args.portable_root.resolve()),
        "dataset_count": len(authorized),
        "window_count": len(rows),
        "synthetic_pass_file_sha256": synthetic["file_sha256"],
        "train_coverage_pass_file_sha256": (
            None if coverage is None else coverage["file_sha256"]
        ),
        "rows": rows,
        "rows_content_sha256": canonical_json_sha256(rows),
        "marker_write_order": "after_every_authorized_portable_window_was_loaded_and_hashed",
    }
    marker_path = run_dir / _portable_population_marker_name(args.access_scope)
    _atomic_json(marker_path, marker)
    print(
        json.dumps(
            {
                "status": "passed",
                "access_scope": args.access_scope,
                "dataset_count": len(authorized),
                "window_count": len(rows),
                "rows_content_sha256": marker["rows_content_sha256"],
                "path": str(marker_path),
                "file_sha256": sha256_file(marker_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _build_slice(args: argparse.Namespace) -> None:
    plan = _load_plan(args.config, args.expected_experiment)
    cache_builder_commit = _git_commit_and_clean(args.expected_experiment)
    synthetic_evidence = None
    coverage_evidence = None
    portable_population_evidence = None
    if plan.experiment == "mainExp_TemplateMatching_3.1":
        if args.synthetic_pass is None:
            raise ValueError("3.1 build-slice requires --synthetic-pass")
        synthetic_evidence = validate_phase31_synthetic_pass(
            plan,
            args.synthetic_pass,
            verify_config_path=args.verify_config,
            current_git_commit=cache_builder_commit,
        )
    if plan.experiment == "mainExp_TemplateMatching_3.1":
        if args.access_scope not in {"train-only", "all"}:
            raise ValueError("3.1 build-slice requires an explicit --access-scope")
        if args.portable_root is None or args.portable_manifest is not None:
            raise ValueError(
                "3.1 build-slice accepts --portable-root only so its population "
                "PASS marker has one unambiguous root"
            )
        if args.access_scope == "train-only":
            authorized_datasets = plan.train_datasets
        else:
            if args.train_coverage_pass is None:
                raise ValueError(
                    "3.1 access-scope=all requires --train-coverage-pass, "
                    "including when the requested target is a train dataset"
                )
            coverage_evidence = validate_phase31_train_coverage_pass(
                plan,
                args.train_coverage_pass,
                synthetic_pass_path=args.synthetic_pass,
                verify_config_path=args.verify_config,
                current_git_commit=cache_builder_commit,
            )
            authorized_datasets = plan.datasets
    else:
        authorized_datasets = plan.datasets
    if args.dataset not in authorized_datasets:
        raise ValueError(f"dataset is outside the frozen split: {args.dataset}")
    if (
        plan.experiment == "mainExp_TemplateMatching_3.1"
        and args.dataset in plan.test_datasets
    ):
        if args.access_scope != "all":
            raise ValueError("3.1 test cache build requires --access-scope all")
    if plan.experiment == "mainExp_TemplateMatching_3.1":
        if args.portable_population_pass is None:
            raise ValueError(
                "3.1 build-slice requires the matching --portable-population-pass"
            )
        portable_population_evidence = _validate_portable_population_pass(
            plan,
            args.portable_population_pass,
            access_scope=args.access_scope,
            portable_root=args.portable_root,
            current_git_commit=cache_builder_commit,
            synthetic_evidence=synthetic_evidence,
            coverage_evidence=coverage_evidence,
        )
    if not 0 <= args.ordinal < plan.source_count:
        raise ValueError(f"ordinal must be in [0,{plan.source_count - 1}]")
    index = _portable_manifest_index(
        plan, args, cache_builder_commit, authorized_datasets
    )
    native_frames = index.native_frame_count(args.dataset)
    frozen_indices = plan.source_indices(native_frames)
    manifest_indices = tuple(
        int(row["source_start_index"]) for row in index.dataset_rows(args.dataset)
    )
    if frozen_indices != manifest_indices:
        raise AssertionError("all-dataset source formula validation was not stable")
    source_index = frozen_indices[args.ordinal]
    row = index.row(args.dataset, source_index)
    if portable_population_evidence is not None:
        population_rows = [
            value
            for value in portable_population_evidence["rows"]
            if value["dataset"] == args.dataset
            and int(value["source_ordinal"]) == args.ordinal
        ]
        if len(population_rows) != 1:
            raise ValueError("portable population marker does not contain this shard")
        population_row = population_rows[0]
        expected_population_identity = {
            "source_start_index": int(row["source_start_index"]),
            "file_size": int(row["file_size"]),
            "file_sha256": str(row["file_sha256"]),
            "manifest_file_sha256": str(row["__manifest_file_sha256"]),
        }
        drift = {
            name: (population_row.get(name), value)
            for name, value in expected_population_identity.items()
            if population_row.get(name) != value
        }
        if drift:
            raise ValueError(f"portable population marker/shard identity changed: {drift}")
    portable_path = index.resolve_path(row)
    file_sha = row.get("file_sha256", row.get("portable_file_sha256"))
    portable = load_portable_flow_window(
        portable_path,
        expected_dataset=args.dataset,
        expected_experiment=plan.experiment,
        expected_config_sha256=plan.config_sha256,
        expected_source_start_index=source_index,
        expected_file_sha256=str(file_sha) if file_sha else None,
        expected_dataset_registry_sha256=str(row["dataset_registry_sha256"]),
        expected_builder_git_commit=str(row["builder_git_commit"]),
    )
    split = plan.split_for(args.dataset)
    output_dir = args.cache_root.resolve() / split / args.dataset
    cache_path = output_dir / f"source_{source_index:06d}.npz"
    sidecar_path = output_dir / f"source_{source_index:06d}.summary.json"
    if sidecar_path.exists() and not cache_path.exists():
        raise FileNotFoundError(
            f"cache sidecar exists without its immutable cache: {sidecar_path}"
        )
    if cache_path.exists():
        current_window_provenance = {
            **dict(portable.metadata),
            "portable_path": str(portable.path.resolve()),
            "portable_file_sha256": portable.file_sha256,
            "portable_file_size": int(portable.path.stat().st_size),
        }
        recovered = recover_phase21_cache_summary(
            plan,
            cache_path=cache_path,
            dataset=args.dataset,
            source_ordinal=args.ordinal,
            source_index=source_index,
            cache_builder_git_commit=cache_builder_commit,
            strict_evidence=True,
            expected_window_provenance=current_window_provenance,
        )
        if portable_population_evidence is not None:
            recovered.cache_row.update(
                {
                    "portable_population_pass_path": portable_population_evidence["path"],
                    "portable_population_pass_file_size": portable_population_evidence["file_size"],
                    "portable_population_pass_file_sha256": portable_population_evidence["file_sha256"],
                    "portable_population_scope": portable_population_evidence["access_scope"],
                    "portable_population_rows_content_sha256": portable_population_evidence["rows_content_sha256"],
                }
            )
        if sidecar_path.exists():
            published = load_cache_summary_sidecar(sidecar_path)
            if cache_summary_payload(published) != cache_summary_payload(recovered):
                raise ValueError(
                    f"published cache sidecar differs from the validated cache: {sidecar_path}"
                )
            print(
                f"cache already complete and verified dataset={args.dataset} "
                f"ordinal={args.ordinal} source={source_index} "
                f"sha256={recovered.cache_row['file_sha256']}",
                flush=True,
            )
            return
        write_cache_summary_sidecar(recovered, sidecar_path)
        print(
            f"recovered missing sidecar dataset={args.dataset} ordinal={args.ordinal} "
            f"source={source_index} sha256={recovered.cache_row['file_sha256']}",
            flush=True,
        )
        return
    summary = build_phase21_cache_slice(
        plan,
        dataset=args.dataset,
        source_ordinal=args.ordinal,
        source_index=source_index,
        resolved_input=portable,
        cache_path=cache_path,
        integration_chunk_size=args.integration_chunk_size,
        encoding_chunk_size=args.encoding_chunk_size,
        strict_evidence=True,
        cache_builder_git_commit=cache_builder_commit,
    )
    if portable_population_evidence is not None:
        summary.cache_row.update(
            {
                "portable_population_pass_path": portable_population_evidence["path"],
                "portable_population_pass_file_size": portable_population_evidence["file_size"],
                "portable_population_pass_file_sha256": portable_population_evidence["file_sha256"],
                "portable_population_scope": portable_population_evidence["access_scope"],
                "portable_population_rows_content_sha256": portable_population_evidence["rows_content_sha256"],
            }
        )
    write_cache_summary_sidecar(summary, sidecar_path)
    print(
        f"cache complete dataset={args.dataset} ordinal={args.ordinal} "
        f"source={source_index} valid={summary.cache_row['valid_count']}/"
        f"{summary.cache_row['assigned_count']} sha256={summary.cache_row['file_sha256']}",
        flush=True,
    )


def _evaluate(args: argparse.Namespace) -> None:
    plan = _load_plan(args.config, args.expected_experiment)
    commit = _git_commit_and_clean(args.expected_experiment)
    verification_evidence = None
    if plan.experiment == "mainExp_TemplateMatching_3.1":
        if args.synthetic_pass is None or args.train_coverage_pass is None:
            raise ValueError(
                "3.1 evaluate requires --synthetic-pass and --train-coverage-pass"
            )
        verification_evidence = validate_phase31_train_coverage_pass(
            plan,
            args.train_coverage_pass,
            synthetic_pass_path=args.synthetic_pass,
            verify_config_path=args.verify_config,
            current_git_commit=commit,
        )
    # Test-cache discovery occurs only after the complete Verify pass above.
    summaries = discover_phase21_cache_sidecars(plan, args.cache_root)
    result = evaluate_phase21_caches(
        plan,
        cache_summaries=summaries,
        run_dir=args.run_dir,
        git_commit=commit,
        device=args.device,
        strict_protocol=True,
        verify_cache_hashes=True,
        query_chunk_size=args.query_chunk_size,
        library_chunk_size=args.library_chunk_size,
        phase31_verification_evidence=verification_evidence,
    )
    print(
        f"evaluation complete run_dir={result['run_dir']} "
        f"result_sha256={result['result_manifest_file_sha256']}",
        flush=True,
    )


def _audit_train_coverage(args: argparse.Namespace) -> None:
    plan = _load_plan(args.config, args.expected_experiment)
    commit = _git_commit_and_clean(args.expected_experiment)
    if args.synthetic_pass is None:
        raise ValueError("train coverage requires --synthetic-pass")
    synthetic_evidence = validate_phase31_synthetic_pass(
        plan,
        args.synthetic_pass,
        verify_config_path=args.verify_config,
        current_git_commit=commit,
    )
    if args.portable_population_pass is None:
        raise ValueError(
            "train coverage requires --portable-population-pass "
            "TRAIN_PORTABLES_PASS.json"
        )
    authorized_train_portable_marker = (
        authorize_phase31_portable_population_marker_path(
            plan,
            args.portable_population_pass,
            access_scope="train-only",
        )
    )
    marker_preview = json.loads(
        authorized_train_portable_marker.read_text(encoding="utf-8")
    )
    marker_portable_root = marker_preview.get("portable_root")
    if not isinstance(marker_portable_root, str) or not marker_portable_root:
        raise ValueError("TRAIN_PORTABLES_PASS has no portable_root identity")
    validated_train_portable_marker = _validate_portable_population_pass(
        plan,
        authorized_train_portable_marker,
        access_scope="train-only",
        portable_root=Path(marker_portable_root),
        current_git_commit=commit,
        synthetic_evidence=synthetic_evidence,
        coverage_evidence=None,
    )
    execution_contract = configure_deterministic_execution()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"immutable Phase B run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(run_dir / "frozen_verify_config.yaml", args.verify_config.read_bytes())
    _atomic_bytes(run_dir / "frozen_main_config.yaml", plan.config_path.read_bytes())
    sidecar_records = []
    cache_root = args.cache_root.resolve()
    for dataset in plan.train_datasets:
        sources = sorted((cache_root / "train" / dataset).glob("*.summary.json"))
        if len(sources) != plan.source_count:
            raise ValueError(
                f"train coverage requires {plan.source_count} sidecars for "
                f"{dataset}, found {len(sources)}"
            )
        for path in sources:
            resolved_sidecar = path.resolve()
            if resolved_sidecar.parent != (cache_root / "train" / dataset).resolve():
                raise ValueError("train sidecar resolves outside its authorized directory")
            sidecar_records.append(
                (dataset, resolved_sidecar, load_cache_summary_sidecar(resolved_sidecar))
            )
    if len(sidecar_records) != len(plan.train_datasets) * plan.source_count:
        raise ValueError("train coverage requires exactly 32 train sidecars")
    summaries = [record[2] for record in sidecar_records]
    train_rows = [summary.cache_row for summary in summaries]
    train_portable_population_evidence = (
        validate_phase31_cache_portable_population_evidence(
            plan,
            train_rows,
            usage="train-coverage",
            expected_git_commit=commit,
            synthetic_pass_file_sha256=synthetic_evidence["file_sha256"],
            authorized_marker_paths_by_scope={
                "train-only": authorized_train_portable_marker
            },
        )[0]
    )
    if (
        train_portable_population_evidence["file_sha256"]
        != validated_train_portable_marker["file_sha256"]
    ):
        raise ValueError(
            "train cache sidecars and explicit portable marker identity differ"
        )

    # Validate every row and resolved target scope before stat/hash opens any
    # cache.  A forged train sidecar must not be able to probe a test cache.
    ordinals_by_dataset: dict[str, set[int]] = {
        dataset: set() for dataset in plan.train_datasets
    }
    validated_cache_paths = []
    for expected_dataset, _sidecar_path, summary in sidecar_records:
        row = summary.cache_row
        dataset = str(row.get("dataset", ""))
        split = str(row.get("split", ""))
        ordinal = int(row.get("source_ordinal", -1))
        if dataset != expected_dataset or dataset not in plan.train_datasets:
            raise ValueError("train sidecar dataset differs from its authorized directory")
        if split != "train":
            raise ValueError("train coverage sidecar must declare split=train")
        if ordinal < 0 or ordinal >= plan.source_count:
            raise ValueError("train coverage sidecar source ordinal is outside 0..3")
        if ordinal in ordinals_by_dataset[dataset]:
            raise ValueError("train coverage has a duplicate dataset/source ordinal")
        ordinals_by_dataset[dataset].add(ordinal)
        raw_cache_path = Path(str(row.get("path", "")))
        cache_path_lexical = Path(os.path.abspath(raw_cache_path))
        authorized_parent = (cache_root / "train" / dataset).resolve()
        if cache_path_lexical.parent != authorized_parent:
            raise ValueError("train sidecar cache path escapes its authorized train directory")
        relative_parts = cache_path_lexical.relative_to(cache_root).parts
        if any(
            test_dataset.casefold() in part.casefold()
            for test_dataset in plan.test_datasets
            for part in relative_parts
        ):
            raise ValueError("train sidecar cache path contains a test dataset name")
        cache_path = cache_path_lexical.resolve()
        if cache_path.parent != authorized_parent:
            raise ValueError("train sidecar cache path resolves outside its train directory")
        validated_cache_paths.append(cache_path)
    if any(
        ordinals != set(range(plan.source_count))
        for ordinals in ordinals_by_dataset.values()
    ):
        raise ValueError("train coverage requires source ordinals 0..3 for every dataset")

    cache_input_rows = []
    for (_dataset, sidecar_path, summary), cache_path in zip(
        sidecar_records, validated_cache_paths, strict=True
    ):
        cache_input_rows.append(
            {
                "dataset": str(summary.cache_row["dataset"]),
                "source_ordinal": int(summary.cache_row["source_ordinal"]),
                "source_index": int(summary.cache_row["source_index"]),
                "cache_path": str(cache_path),
                "cache_size_bytes": int(cache_path.stat().st_size),
                "cache_file_sha256": sha256_file(cache_path),
                "sidecar_path": str(sidecar_path),
                "sidecar_size_bytes": int(sidecar_path.stat().st_size),
                "sidecar_file_sha256": sha256_file(sidecar_path),
            }
        )
    cache_input_manifest = {
        "schema": "pathline_template_matching.long_arc_train_cache_input.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "parent_experiment": plan.experiment,
        "git_commit": commit,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": synthetic_evidence["verify_config_sha256"],
        "synthetic_pass_file_sha256": synthetic_evidence["file_sha256"],
        "train_portable_population_pass": train_portable_population_evidence,
        "input_scope": "exactly_32_train_cache_shards_and_sidecars",
        "test_dataset_access": False,
        "row_count": len(cache_input_rows),
        "rows": cache_input_rows,
        "rows_content_sha256": canonical_json_sha256(cache_input_rows),
    }
    _atomic_json(run_dir / "train_cache_input_manifest.json", cache_input_manifest)
    result = audit_phase31_train_coverage(
        plan,
        train_rows,
        run_dir,
        verify_cache_hashes=True,
        expected_git_commit=commit,
        synthetic_pass_file_sha256=synthetic_evidence["file_sha256"],
        authorized_portable_population_marker_path=(
            authorized_train_portable_marker
        ),
    )
    environment = {
        "schema": "pathline_template_matching.long_arc_train_coverage_environment.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "phase": "train_coverage",
        "git_commit": commit,
        "python": sys.version,
        "platform": sys.platform,
        "torch_deterministic_algorithms": bool(
            __import__("torch").are_deterministic_algorithms_enabled()
        ),
        "deterministic_execution": execution_contract,
        "real_data_scope": "train_cache_only",
        "test_dataset_access": False,
    }
    _atomic_json(run_dir / "environment_versions.json", environment)
    verification = {
        "schema": "pathline_template_matching.long_arc_verification.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "phase": "train_coverage",
        "status": "passed" if result["status"] == "pass" else "failed",
        "final_verify_pass": result["status"] == "pass",
        "git_commit": commit,
        "worktree_clean": True,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": synthetic_evidence["verify_config_sha256"],
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "synthetic_pass_path": synthetic_evidence["path"],
        "synthetic_pass_file_sha256": synthetic_evidence["file_sha256"],
        "train_portable_population_pass": train_portable_population_evidence,
        "train_portable_population_pass_file_sha256": (
            train_portable_population_evidence["file_sha256"]
        ),
        "train_cache_input_manifest_sha256": sha256_file(
            run_dir / "train_cache_input_manifest.json"
        ),
        "train_only_coverage_diagnostics_sha256": sha256_file(
            run_dir / "train_only_coverage_diagnostics.csv"
        ),
        "train_only_coverage_summary_sha256": sha256_file(
            run_dir / "train_only_coverage_summary.json"
        ),
        "train_cache_count": 32,
        "train_only": True,
        "no_test_dataset_access": True,
        "coverage_pass_conditions": result["pass_conditions"],
    }
    verification["verification_content_sha256"] = canonical_json_sha256(
        verification
    )
    _atomic_json(run_dir / "verification.json", verification)
    if not verification["final_verify_pass"]:
        raise RuntimeError(
            "train coverage failed; evidence was preserved without a PASS marker"
        )
    evidence_names = (
        "frozen_verify_config.yaml",
        "frozen_main_config.yaml",
        "train_cache_input_manifest.json",
        "train_only_coverage_diagnostics.csv",
        "train_only_coverage_summary.json",
        "environment_versions.json",
        "verification.json",
    )
    output_rows = [
        {
            "path": name,
            "size_bytes": int((run_dir / name).stat().st_size),
            "sha256": sha256_file(run_dir / name),
        }
        for name in evidence_names
    ]
    marker = {
        "schema": "pathline_template_matching.long_arc_train_coverage_pass.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "phase": "train_coverage",
        "status": "passed",
        "git_commit": commit,
        "worktree_clean": True,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": synthetic_evidence["verify_config_sha256"],
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "synthetic_pass_file_sha256": synthetic_evidence["file_sha256"],
        "train_portable_population_pass_file_sha256": (
            train_portable_population_evidence["file_sha256"]
        ),
        "verification_file_sha256": sha256_file(run_dir / "verification.json"),
        "final_verify_pass": True,
        "outputs": output_rows,
        "outputs_content_sha256": canonical_json_sha256(output_rows),
        "marker_write_order": "last_after_all_seven_phase_b_outputs_are_fsynced",
    }
    _atomic_json(run_dir / "TRAIN_COVERAGE_PASS.json", marker)
    print(json.dumps(marker, sort_keys=True), flush=True)


def main(
    *,
    default_config: Path = DEFAULT_CONFIG,
    expected_experiment: str = "mainExp_TemplateMatching_2.1",
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Build parallel cache shards or evaluate {expected_experiment}."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "preflight-portables",
            "build-slice",
            "audit-train-coverage",
            "evaluate",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--portable-manifest", type=Path)
    parser.add_argument("--portable-root", type=Path)
    parser.add_argument("--access-scope", choices=("train-only", "all"))
    parser.add_argument("--dataset")
    parser.add_argument("--ordinal", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--synthetic-pass", type=Path)
    parser.add_argument("--train-coverage-pass", type=Path)
    parser.add_argument("--portable-population-pass", type=Path)
    parser.add_argument(
        "--verify-config",
        type=Path,
        default=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--integration-chunk-size", type=int, default=2048)
    parser.add_argument("--encoding-chunk-size", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--library-chunk-size", type=int, default=8192)
    args = parser.parse_args()
    args.expected_experiment = expected_experiment
    if expected_experiment == "mainExp_TemplateMatching_2.1" and args.access_scope is None:
        args.access_scope = "all"
    if args.mode == "build-slice":
        if (
            args.cache_root is None
            or
            (args.portable_manifest is None and args.portable_root is None)
            or args.dataset is None
            or args.ordinal is None
        ):
            parser.error(
                "build-slice requires --cache-root, --portable-root "
                "(or --portable-manifest), --dataset, and --ordinal"
            )
        _build_slice(args)
    elif args.mode == "preflight-portables":
        if args.run_dir is None or args.portable_root is None:
            parser.error(
                "preflight-portables requires --run-dir and --portable-root"
            )
        _preflight_portables(args)
    elif args.mode == "evaluate":
        if args.run_dir is None or args.cache_root is None:
            parser.error("evaluate requires --run-dir and --cache-root")
        _evaluate(args)
    else:
        if args.run_dir is None or args.cache_root is None:
            parser.error("audit-train-coverage requires --run-dir and --cache-root")
        if expected_experiment != "mainExp_TemplateMatching_3.1":
            parser.error("audit-train-coverage is available only for mainExp 3.1")
        _audit_train_coverage(args)


if __name__ == "__main__":
    main()
