#!/usr/bin/env python3
"""Immutable assigned-row sidecar stages for SourceCenteredPairedScaleTemplate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from pathline_template_matching.source_centered_sidecar import (  # noqa: E402
    EXPERIMENT,
    PRODUCTION_CONTRACT,
    authenticate_source_centered_input_manifest,
    authenticate_source_centered_population_manifest,
    authenticate_source_centered_row_completion,
    build_one_source_centered_sidecar_and_completion,
    build_source_centered_input_manifest,
    capture_clean_source_identity,
    sidecar_row_relative_directory,
    write_source_centered_population_manifest,
)


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_forbidden_dataset_tokens(value: object, *, name: str) -> None:
    normalized = "".join(
        character for character in str(value).casefold() if character.isalnum()
    )
    if "tangaroa" in normalized or "smokebuoyancy" in normalized:
        raise ValueError(f"{name} contains a forbidden confirmation dataset token")


def _sha256(value: str, *, name: str) -> str:
    _require(_HEX64.fullmatch(value) is not None, f"{name} must be lowercase SHA-256")
    return value


def _capture_identity(project_root: Path, expected_git_commit: str):
    _require(
        _HEX40.fullmatch(expected_git_commit) is not None,
        "expected Git commit must be lowercase 40-hex",
    )
    identity = capture_clean_source_identity(project_root)
    _require(
        identity.git_commit == expected_git_commit,
        "clean checkout commit differs from --expected-git-commit",
    )
    return identity


def _row_key(row_index: int) -> tuple[str, int]:
    contract = PRODUCTION_CONTRACT
    _require(
        isinstance(row_index, int)
        and not isinstance(row_index, bool)
        and 0 <= row_index < contract.row_count,
        f"row index must be in 0..{contract.row_count - 1}",
    )
    dataset_index, source_ordinal = divmod(row_index, contract.source_count)
    return str(contract.datasets[dataset_index]), int(source_ordinal)


def _manifest_row(
    manifest: Mapping[str, Any], row_index: int
) -> Mapping[str, Any]:
    dataset, source_ordinal = _row_key(row_index)
    rows = manifest.get("rows")
    _require(isinstance(rows, tuple), "authenticated input rows are not immutable")
    matches = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and int(row.get("source_ordinal", -1)) == source_ordinal
    ]
    _require(len(matches) == 1, "row index does not resolve uniquely")
    return matches[0]


def _summary(
    value: Mapping[str, Any], *, path: Path, identity: Any, stage: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "stage": stage,
        "status": str(value.get("status", "authenticated")),
        "schema": str(value.get("schema", "")),
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "file_sha256": sha256_file(path),
        "content_sha256": str(value.get("content_sha256", "")),
        "git_commit": identity.git_commit,
        "source_content_sha256": identity.source_content_sha256,
    }
    for name in (
        "row_count",
        "sidecar_count",
        "assigned_row_count",
        "valid_projection_row_count",
        "assigned_row_count_total",
        "valid_projection_row_count_total",
    ):
        if name in value:
            result[name] = int(value[name])
    for name in ("dataset", "dataset_index", "source_ordinal", "source_index"):
        if name in value:
            result[name] = value[name]
    return result


def _recapture(project_root: Path, identity: Any) -> None:
    observed = capture_clean_source_identity(project_root)
    _require(observed == identity, "clean commit or source hashes changed during stage")


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(arguments.project_root).resolve()
    for name, value in vars(arguments).items():
        if isinstance(value, (str, Path)):
            _reject_forbidden_dataset_tokens(value, name=name)
    identity = _capture_identity(project_root, arguments.expected_git_commit)
    contract = PRODUCTION_CONTRACT

    if arguments.command == "freeze-input":
        output = Path(arguments.output_path).resolve()
        value = build_source_centered_input_manifest(
            output,
            early_input_manifest_path=arguments.early_input_manifest,
            identity=identity,
            contract=contract,
        )
        stage = "freeze_exact_32_assigned_row_inputs"
    elif arguments.command == "authenticate-input":
        output = Path(arguments.input_manifest).resolve()
        value = authenticate_source_centered_input_manifest(
            output,
            expected_file_sha256=_sha256(
                arguments.input_manifest_sha256, name="input manifest SHA-256"
            ),
            identity=identity,
            contract=contract,
            authenticate_all_referenced_rows=arguments.authenticate_all_rows,
        )
        stage = "fresh_authenticate_input_manifest"
    elif arguments.command == "build-sidecar":
        dataset, source_ordinal = _row_key(arguments.row_index)
        value = build_one_source_centered_sidecar_and_completion(
            arguments.sidecar_root,
            dataset=dataset,
            source_ordinal=source_ordinal,
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256, name="input manifest SHA-256"
            ),
            identity=identity,
            contract=contract,
        )
        input_manifest = authenticate_source_centered_input_manifest(
            arguments.input_manifest,
            expected_file_sha256=arguments.input_manifest_sha256,
            identity=identity,
            contract=contract,
            authenticate_all_referenced_rows=False,
        )
        row = _manifest_row(input_manifest, arguments.row_index)
        output = (
            Path(arguments.sidecar_root).resolve()
            / sidecar_row_relative_directory(
                dataset, source_ordinal, int(row["source_index"])
            )
            / "SIDECAR_COMPLETE.json"
        )
        stage = "build_one_assigned_row_sidecar"
    elif arguments.command == "authenticate-row":
        input_manifest = authenticate_source_centered_input_manifest(
            arguments.input_manifest,
            expected_file_sha256=_sha256(
                arguments.input_manifest_sha256, name="input manifest SHA-256"
            ),
            identity=identity,
            contract=contract,
            authenticate_all_referenced_rows=False,
        )
        row = _manifest_row(input_manifest, arguments.row_index)
        dataset, source_ordinal = _row_key(arguments.row_index)
        output = (
            Path(arguments.sidecar_root).resolve()
            / sidecar_row_relative_directory(
                dataset, source_ordinal, int(row["source_index"])
            )
            / "SIDECAR_COMPLETE.json"
        )
        value = authenticate_source_centered_row_completion(
            output,
            sidecar_root=arguments.sidecar_root,
            expected_file_sha256=_sha256(
                arguments.completion_sha256, name="completion SHA-256"
            ),
            input_manifest=input_manifest,
            input_manifest_file_sha256=arguments.input_manifest_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        stage = "fresh_authenticate_one_assigned_row_sidecar"
    elif arguments.command == "seal-population":
        value = write_source_centered_population_manifest(
            arguments.sidecar_root,
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256, name="input manifest SHA-256"
            ),
            identity=identity,
            contract=contract,
        )
        output = Path(arguments.sidecar_root).resolve() / "SIDECAR_POPULATION.json"
        stage = "seal_exact_32_assigned_row_sidecars"
    elif arguments.command == "authenticate-population":
        output = Path(arguments.population_manifest).resolve()
        value = authenticate_source_centered_population_manifest(
            output,
            sidecar_root=arguments.sidecar_root,
            expected_file_sha256=_sha256(
                arguments.population_manifest_sha256,
                name="population manifest SHA-256",
            ),
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256, name="input manifest SHA-256"
            ),
            identity=identity,
            contract=contract,
        )
        stage = "fresh_authenticate_exact_32_sidecar_population"
    else:  # pragma: no cover
        raise RuntimeError(f"unsupported command: {arguments.command}")

    result = _summary(value, path=output, identity=identity, stage=stage)
    _recapture(project_root, identity)
    return result


def _input_evidence(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--expected-git-commit", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-input")
    freeze.add_argument("--output-path", required=True)
    freeze.add_argument("--early-input-manifest", required=True)

    authenticate_input = subparsers.add_parser("authenticate-input")
    _input_evidence(authenticate_input)
    authenticate_input.add_argument("--authenticate-all-rows", action="store_true")

    sidecar = subparsers.add_parser("build-sidecar")
    sidecar.add_argument("--sidecar-root", required=True)
    sidecar.add_argument("--row-index", type=int, required=True)
    _input_evidence(sidecar)

    row = subparsers.add_parser("authenticate-row")
    row.add_argument("--sidecar-root", required=True)
    row.add_argument("--row-index", type=int, required=True)
    row.add_argument("--completion-sha256", required=True)
    _input_evidence(row)

    seal = subparsers.add_parser("seal-population")
    seal.add_argument("--sidecar-root", required=True)
    _input_evidence(seal)

    population = subparsers.add_parser("authenticate-population")
    population.add_argument("--sidecar-root", required=True)
    population.add_argument("--population-manifest", required=True)
    population.add_argument("--population-manifest-sha256", required=True)
    _input_evidence(population)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(_run(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
