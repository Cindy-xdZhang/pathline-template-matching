#!/usr/bin/env python3
"""Build parallel cache shards or evaluate mainExp_TemplateMatching_2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.phase21_pipeline import (
    build_phase21_cache_slice,
    cache_summary_payload,
    discover_phase21_cache_sidecars,
    evaluate_phase21_caches,
    load_cache_summary_sidecar,
    load_phase21_plan,
    recover_phase21_cache_summary,
    write_cache_summary_sidecar,
)
from pathline_template_matching.portable_flow import load_portable_flow_window


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
        expected_source_kind_by_dataset: Mapping[str, str] | None = None,
    ) -> None:
        self.portable_root = portable_root.resolve() if portable_root else None
        if path is None:
            if self.portable_root is None:
                raise ValueError("either portable manifest or portable root is required")
            manifest_paths = sorted(self.portable_root.glob("*/manifest.json"))
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
                if int(row.get("frame_count", -1)) != 13:
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
        # The frozen fourth source index is exactly T-13, so it supplies a
        # deterministic fallback when the staging manifest omits native T.
        return next(iter(explicit)) if explicit else int(rows[-1]["source_start_index"]) + 13

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


def _git_commit_and_clean() -> str:
    critical_paths = (
        "scripts/run_mainexp_template_matching_2_1.py",
        "src/pathline_template_matching/phase21_pipeline.py",
        "src/pathline_template_matching/arc_length_primitives.py",
        "src/pathline_template_matching/portable_flow.py",
        "config/mainExp_TemplateMatching_2.1.yaml",
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


def _build_slice(args: argparse.Namespace) -> None:
    plan = load_phase21_plan(args.config)
    cache_builder_commit = _git_commit_and_clean()
    if args.dataset not in plan.datasets:
        raise ValueError(f"dataset is outside the frozen split: {args.dataset}")
    if not 0 <= args.ordinal < plan.source_count:
        raise ValueError(f"ordinal must be in [0,{plan.source_count - 1}]")
    registry = yaml.safe_load(plan.dataset_registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping) or not isinstance(registry.get("datasets"), list):
        raise ValueError("committed dataset registry has an invalid structure")
    registry_rows = {
        str(row["id"]): str(row["kind"]) for row in registry["datasets"]
    }
    if set(registry_rows) != set(plan.datasets):
        raise ValueError("committed dataset registry does not exactly cover the frozen split")
    index = PortableManifestIndex(
        args.portable_manifest,
        portable_root=args.portable_root,
        expected_experiment=plan.experiment,
        expected_config_sha256=plan.config_sha256,
        expected_dataset_registry_sha256=plan.dataset_registry_sha256,
        expected_datasets=plan.datasets,
        expected_family_by_dataset=plan.family_by_dataset,
        expected_split_by_dataset={
            dataset: plan.split_for(dataset) for dataset in plan.datasets
        },
        expected_builder_git_commit=cache_builder_commit,
        expected_source_kind_by_dataset=registry_rows,
    )
    for frozen_dataset in plan.datasets:
        native_count = index.native_frame_count(frozen_dataset)
        expected_indices = plan.source_indices(native_count)
        found_indices = tuple(
            int(row["source_start_index"])
            for row in index.dataset_rows(frozen_dataset)
        )
        if expected_indices != found_indices:
            raise ValueError(
                f"portable windows do not follow the frozen source formula for "
                f"{frozen_dataset}: expected {expected_indices}, found {found_indices}"
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
    write_cache_summary_sidecar(summary, sidecar_path)
    print(
        f"cache complete dataset={args.dataset} ordinal={args.ordinal} "
        f"source={source_index} valid={summary.cache_row['valid_count']}/"
        f"{summary.cache_row['assigned_count']} sha256={summary.cache_row['file_sha256']}",
        flush=True,
    )


def _evaluate(args: argparse.Namespace) -> None:
    plan = load_phase21_plan(args.config)
    commit = _git_commit_and_clean()
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
    )
    print(
        f"evaluation complete run_dir={result['run_dir']} "
        f"result_sha256={result['result_manifest_file_sha256']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("build-slice", "evaluate"), required=True
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path)
    parser.add_argument("--portable-root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--ordinal", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--integration-chunk-size", type=int, default=2048)
    parser.add_argument("--encoding-chunk-size", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--library-chunk-size", type=int, default=8192)
    args = parser.parse_args()
    if args.mode == "build-slice":
        if (
            (args.portable_manifest is None and args.portable_root is None)
            or args.dataset is None
            or args.ordinal is None
        ):
            parser.error(
                "build-slice requires --portable-root (or --portable-manifest), "
                "--dataset, and --ordinal"
            )
        _build_slice(args)
    else:
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir")
        _evaluate(args)


if __name__ == "__main__":
    main()
