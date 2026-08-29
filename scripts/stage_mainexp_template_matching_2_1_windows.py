"""Build immutable portable windows for a frozen template-matching experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.channel_flow import (  # noqa: E402
    build_channel_flow_window,
    load_steady_channel_vtk,
)
from pathline_template_matching.netcdf_io import (  # noqa: E402
    inspect_netcdf_3d,
    load_netcdf_window_3d,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    load_portable_flow_window,
    sha256_file,
    write_portable_flow_window,
)
from pathline_template_matching.phase21_pipeline import (  # noqa: E402
    load_phase31_plan,
    validate_phase31_synthetic_pass,
    validate_phase31_train_coverage_pass,
)


DATASET_MANIFEST_SCHEMA = (
    "pathline_template_matching.portable_flow_dataset_manifest.v1"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _git_commit_and_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("portable staging requires a clean committed worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("could not resolve a full Git commit")
    return commit


def _require_file_matches_commit(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"frozen input is outside the Git repository: {path}") from error
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != sha256_file(path):
        raise RuntimeError(
            f"working-file bytes differ from committed bytes (including line endings): {path}"
        )


def _source_indices(total_frames: int, future_intervals: int = 12) -> list[int]:
    total = int(total_frames)
    intervals = int(future_intervals)
    maximum_start = total - intervals - 1
    if intervals < 1 or maximum_start < 3:
        raise ValueError(
            f"dataset cannot provide four unique {intervals}-frame future windows"
        )
    indices = [int(np.floor(k * maximum_start / 3.0)) for k in range(4)]
    if len(set(indices)) != 4 or indices[-1] + intervals >= total:
        raise AssertionError("source-index selection violated the frozen window contract")
    return indices


def _dataset_maps(config: dict[str, Any], registry: dict[str, Any]):
    rows = {str(row["id"]): row for row in registry["datasets"]}
    train = [str(value) for value in config["split"]["train_datasets"]]
    test = [str(value) for value in config["split"]["test_datasets"]]
    expected = train + test
    if len(expected) != 10 or len(set(expected)) != 10 or set(expected) != set(rows):
        raise ValueError("config split and dataset registry disagree")
    split = {dataset: "train" for dataset in train}
    split.update({dataset: "test" for dataset in test})
    return rows, split


def _registered_source(row: dict[str, Any], environment: str) -> Path:
    candidates = [Path(value) for value in row["raw_paths"].get(environment, [])]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"{row['id']}: expected exactly one registered {environment} source, "
            f"found {len(existing)}"
        )
    return existing[0]


def _stable_source_identity(path: Path) -> tuple[int, str]:
    """Return size/hash only when the source stayed unchanged while hashing."""

    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
        int(getattr(before, "st_ino", 0)),
    )
    after_identity = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
        int(getattr(after, "st_ino", 0)),
    )
    if before_identity != after_identity:
        raise RuntimeError(f"source changed while its SHA-256 was being computed: {path}")
    return int(after.st_size), digest


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _load_published_manifest(
    path: Path,
    *,
    expected_top_level: dict[str, Any],
    selected_indices: list[int],
    frame_count: int,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate an already-published manifest before any resume operation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise ValueError(f"existing dataset manifest schema is unsupported: {path}")
    claimed = payload.get("manifest_content_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("manifest_content_sha256", None)
    if claimed != canonical_json_sha256(digest_payload):
        raise ValueError(f"existing dataset manifest content SHA-256 mismatch: {path}")
    drift = {
        key: (payload.get(key), expected)
        for key, expected in expected_top_level.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"existing dataset manifest provenance drift: {drift}")
    if payload.get("selected_source_indices") != selected_indices:
        raise ValueError("existing dataset manifest source indices drifted")
    windows = payload.get("windows")
    if (
        not isinstance(windows, list)
        or len(windows) != len(selected_indices)
        or int(payload.get("window_count", -1)) != len(windows)
        or not all(isinstance(row, dict) for row in windows)
    ):
        raise ValueError("existing dataset manifest window population is invalid")
    by_index: dict[int, dict[str, Any]] = {}
    for ordinal, row in enumerate(windows):
        index = int(row.get("source_start_index", -1))
        expected_name = f"window_{ordinal:02d}_index_{index:04d}.npz"
        if (
            index != selected_indices[ordinal]
            or int(row.get("source_ordinal", -1)) != ordinal
            or index in by_index
        ):
            raise ValueError("existing dataset manifest window ordering drifted")
        expected_row = {
            **expected_top_level,
            "source_ordinal": ordinal,
            "source_start_index": index,
            "frame_count": frame_count,
            "relative_path": expected_name,
        }
        row_drift = {
            key: (row.get(key), expected)
            for key, expected in expected_row.items()
            if row.get(key) != expected
        }
        if row_drift:
            raise ValueError(
                f"existing dataset manifest window provenance drift: {row_drift}"
            )
        window_path = path.parent / expected_name
        if not window_path.is_file():
            raise FileNotFoundError(
                f"published manifest window is missing; immutable resume refused: {window_path}"
            )
        if int(row.get("file_size", -1)) != int(window_path.stat().st_size):
            raise ValueError(f"published manifest window size drifted: {window_path}")
        file_sha = str(row.get("file_sha256", ""))
        if len(file_sha) != 64 or any(
            character not in "0123456789abcdef" for character in file_sha
        ):
            raise ValueError(f"published manifest window SHA-256 is invalid: {window_path}")
        by_index[index] = row
    return payload, by_index


def stage_dataset(
    *,
    config_path: Path,
    registry_path: Path,
    environment: str,
    dataset: str,
    output_root: Path,
    resume: bool,
    synthetic_pass: Path | None = None,
    train_coverage_pass: Path | None = None,
    verify_config_path: Path | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    registry = _load_yaml(registry_path)
    experiment = str(config.get("experiment", ""))
    expected_frames = {
        "mainExp_TemplateMatching_2.1": 13,
        "mainExp_TemplateMatching_3.1": 49,
    }
    if experiment not in expected_frames:
        raise ValueError("staging script accepts only frozen mainExp 2.1 or 3.1")
    rows, split_by_dataset = _dataset_maps(config, registry)
    if dataset not in rows:
        raise ValueError(f"unknown dataset {dataset!r}")
    row = rows[dataset]
    builder_git_commit = _git_commit_and_clean()
    _require_file_matches_commit(config_path)
    _require_file_matches_commit(registry_path)
    if experiment == "mainExp_TemplateMatching_3.1":
        if synthetic_pass is None or verify_config_path is None:
            raise ValueError("3.1 portable staging requires --synthetic-pass")
        plan = load_phase31_plan(config_path)
        validate_phase31_synthetic_pass(
            plan,
            synthetic_pass,
            verify_config_path=verify_config_path,
            current_git_commit=builder_git_commit,
        )
        if dataset in plan.test_datasets:
            if train_coverage_pass is None:
                raise ValueError(
                    "3.1 test portable staging requires --train-coverage-pass"
                )
            validate_phase31_train_coverage_pass(
                plan,
                train_coverage_pass,
                synthetic_pass_path=synthetic_pass,
                verify_config_path=verify_config_path,
                current_git_commit=builder_git_commit,
            )
    # Raw-path existence, hashing, inspection, and loading all occur only after
    # the relevant 3.1 verification markers have passed above.
    source = _registered_source(row, environment)
    source_size, source_sha256 = _stable_source_identity(source)
    config_sha256 = sha256_file(config_path)
    registry_sha256 = sha256_file(registry_path)
    loading = config["source_loading"]
    frame_count = int(loading["derived_window_frame_count"])
    if frame_count != expected_frames[experiment]:
        raise ValueError(
            f"{experiment} requires exactly {expected_frames[experiment]} loaded frames"
        )
    max_spatial_dim = int(loading["max_spatial_dim"])
    fallback = loading["netcdf_coordinate_policy"]["dataset_overrides"].get(
        dataset, {}
    ).get("axes", [])

    kind = str(row["kind"])
    steady_channel = None
    channel_contract = loading["channel_vtk_contract"]
    if kind == "netcdf":
        info = inspect_netcdf_3d(source, index_coordinate_axes=fallback)
        total_frames = int(info["shape"]["t"])
        native_shape_xyz = [int(info["shape"][axis]) for axis in "xyz"]
        coordinate_units = info["coordinate_units"]
    elif kind == "vtk" and dataset == "channel":
        total_frames = int(channel_contract["total_frames"])
        steady_channel = load_steady_channel_vtk(
            source,
            max_spatial_dim=max_spatial_dim,
            crop_fraction=float(channel_contract["crop_fraction"]),
        )
        native_shape_xyz = list(steady_channel.metadata["source_dimensions_xyz"])
        coordinate_units = {
            axis: {
                "units_attribute_present": False,
                "units_attribute_value": None,
                "effective_units": (
                    "deterministic_observer_time_dimensionless"
                    if axis == "t"
                    else "vtk_coordinate_units_attribute_absent"
                ),
            }
            for axis in "xyzt"
        }
    else:
        raise ValueError(f"unsupported source kind for {dataset}: {kind}")
    selected_indices = _source_indices(
        total_frames, future_intervals=frame_count - 1
    )

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"dataset manifest already exists: {manifest_path}")
    expected_top_level = {
        "experiment": str(config["experiment"]),
        "config_sha256": config_sha256,
        "dataset_registry_sha256": registry_sha256,
        "builder_git_commit": builder_git_commit,
        "dataset": dataset,
        "physical_family": str(row["physical_family"]),
        "split": split_by_dataset[dataset],
        "source_kind": kind,
        "source_file": str(source.resolve()),
        "source_file_size": source_size,
        "source_file_sha256": source_sha256,
        "source_total_frames": total_frames,
    }
    published_manifest: dict[str, Any] | None = None
    published_rows: dict[int, dict[str, Any]] = {}
    if manifest_path.exists():
        published_manifest, published_rows = _load_published_manifest(
            manifest_path,
            expected_top_level=expected_top_level,
            selected_indices=selected_indices,
            frame_count=frame_count,
        )
    window_rows: list[dict[str, Any]] = []
    for ordinal, source_index in enumerate(selected_indices):
        output = dataset_dir / f"window_{ordinal:02d}_index_{source_index:04d}.npz"
        if output.exists():
            if not resume:
                raise FileExistsError(f"portable window already exists: {output}")
            published_row = published_rows.get(source_index)
            loaded = load_portable_flow_window(
                output,
                expected_dataset=dataset,
                expected_experiment=str(config["experiment"]),
                expected_config_sha256=config_sha256,
                expected_dataset_registry_sha256=registry_sha256,
                expected_builder_git_commit=builder_git_commit,
                expected_source_start_index=source_index,
                expected_file_sha256=(
                    str(published_row["file_sha256"])
                    if published_row is not None
                    else None
                ),
            )
            metadata = loaded.metadata
            expected_metadata = {
                **expected_top_level,
                "source_ordinal": ordinal,
                "source_start_index": source_index,
                "frame_count": frame_count,
            }
            metadata_drift = {
                key: (metadata.get(key), expected)
                for key, expected in expected_metadata.items()
                if metadata.get(key) != expected
            }
            if metadata_drift:
                raise ValueError(
                    f"{output}: existing portable provenance drift: {metadata_drift}"
                )
            manifest_row = {
                "relative_path": output.name,
                "file_size": int(output.stat().st_size),
                "file_sha256": loaded.file_sha256,
                **metadata,
            }
            print(f"[{dataset}] verified existing {output.name}", flush=True)
        else:
            if kind == "netcdf":
                window = load_netcdf_window_3d(
                    source,
                    source_index,
                    frame_count,
                    max_spatial_dim=max_spatial_dim,
                    index_coordinate_axes=fallback,
                )
                extra = {
                    "native_shape_xyz": native_shape_xyz,
                    "source_total_frames": total_frames,
                    "source_ordinal": ordinal,
                    "source_kind": kind,
                }
            else:
                window, channel_metadata = build_channel_flow_window(
                    source,
                    source_index,
                    frame_count,
                    max_spatial_dim=max_spatial_dim,
                    crop_fraction=float(channel_contract["crop_fraction"]),
                    total_frames=total_frames,
                    duration=float(channel_contract["duration"]),
                    steady_grid=steady_channel,
                )
                extra = {
                    "native_shape_xyz": native_shape_xyz,
                    "source_total_frames": total_frames,
                    "source_ordinal": ordinal,
                    "source_kind": kind,
                    "channel_contract": channel_metadata,
                }
            manifest_row = write_portable_flow_window(
                output,
                dataset=dataset,
                physical_family=str(row["physical_family"]),
                split=split_by_dataset[dataset],
                experiment=str(config["experiment"]),
                config_sha256=config_sha256,
                dataset_registry_sha256=registry_sha256,
                builder_git_commit=builder_git_commit,
                coordinate_units=coordinate_units,
                source_file=source,
                source_file_sha256=source_sha256,
                source_file_size=source_size,
                window=window,
                extra_metadata=extra,
            )
            manifest_row["relative_path"] = output.name
            manifest_row.pop("path", None)
            print(
                f"[{dataset}] wrote ordinal={ordinal} index={source_index} "
                f"shape={tuple(window.velocity.shape)} {output.name}",
                flush=True,
            )
        window_rows.append(manifest_row)
    # Re-read the complete raw source immediately before publication.  This
    # closes the staging time-of-check/time-of-use interval: no manifest is
    # published for windows built while their source changed.
    publish_source_size, publish_source_sha256 = _stable_source_identity(source)
    if (publish_source_size, publish_source_sha256) != (source_size, source_sha256):
        raise RuntimeError(
            f"source identity changed during portable staging; refusing publication: {source}"
        )
    manifest: dict[str, Any] = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "experiment": str(config["experiment"]),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "dataset_registry_path": str(registry_path.resolve()),
        "dataset_registry_sha256": registry_sha256,
        "builder_git_commit": builder_git_commit,
        "dataset": dataset,
        "physical_family": str(row["physical_family"]),
        "split": split_by_dataset[dataset],
        "source_kind": kind,
        "source_file": str(source.resolve()),
        "source_file_size": source_size,
        "source_file_sha256": source_sha256,
        "source_total_frames": total_frames,
        "selected_source_indices": selected_indices,
        "window_count": len(window_rows),
        "windows": window_rows,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    if published_manifest is not None:
        if published_manifest != manifest:
            raise ValueError(f"existing dataset manifest differs: {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)
    print(
        f"[{dataset}] complete manifest_sha256={manifest['manifest_content_sha256']}",
        flush=True,
    )
    return manifest


def main(
    *,
    default_config: Path | None = None,
    expected_experiment: str = "mainExp_TemplateMatching_2.1",
) -> None:
    if default_config is None:
        default_config = ROOT / "config/mainExp_TemplateMatching_2.1.yaml"
    parser = argparse.ArgumentParser(
        description=f"Build immutable portable windows for {expected_experiment}."
    )
    parser.add_argument(
        "--config", type=Path, default=default_config
    )
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "config/datasets.yaml"
    )
    parser.add_argument("--environment", choices=("local", "ibex"), required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--access-scope", choices=("train-only", "test-only", "all")
    )
    parser.add_argument("--synthetic-pass", type=Path)
    parser.add_argument("--train-coverage-pass", type=Path)
    parser.add_argument(
        "--verify-config",
        type=Path,
        default=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
    )
    args = parser.parse_args()
    config = _load_yaml(args.config)
    if config.get("experiment") != expected_experiment:
        raise ValueError(
            f"entry point requires {expected_experiment}, got {config.get('experiment')}"
        )
    registry = _load_yaml(args.registry)
    rows, _ = _dataset_maps(config, registry)
    train = [str(value) for value in config["split"]["train_datasets"]]
    test = [str(value) for value in config["split"]["test_datasets"]]
    if expected_experiment == "mainExp_TemplateMatching_3.1":
        if args.access_scope is None:
            raise ValueError("3.1 staging requires explicit --access-scope")
        authorized = {
            "train-only": train,
            "test-only": test,
            "all": train + test,
        }[args.access_scope]
    else:
        authorized = train + test
        if args.access_scope not in (None, "all"):
            raise ValueError("2.1 staging supports only the complete dataset scope")
    datasets = list(authorized) if args.dataset == ["all"] else args.dataset
    if len(set(datasets)) != len(datasets):
        raise ValueError("--dataset values must be unique")
    unauthorized = sorted(set(datasets) - set(authorized))
    if unauthorized:
        raise ValueError(
            f"datasets are outside --access-scope {args.access_scope}: {unauthorized}"
        )
    for dataset in datasets:
        stage_dataset(
            config_path=args.config,
            registry_path=args.registry,
            environment=args.environment,
            dataset=dataset,
            output_root=args.output_root,
            resume=args.resume,
            synthetic_pass=args.synthetic_pass,
            train_coverage_pass=args.train_coverage_pass,
            verify_config_path=args.verify_config,
        )


if __name__ == "__main__":
    main()
