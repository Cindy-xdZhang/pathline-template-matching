#!/usr/bin/env python3
"""Complete local rendered QA for the RankLikelihood four-flow report.

This post-render auditor authenticates the immutable machine bundle, source
checkout, editable-text exports, panel geometry, collision evidence, and an
independent final-size human review.  It deliberately contains no ``np.load``
call and never opens prediction, label, parent-scene, or combined-scene array
members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_rank_likelihood_visualization import (  # noqa: E402
    publish_bytes_without_overwrite,
    publish_file_without_overwrite,
)
from scripts.run_verify_early_opposite_pair_kinematics_1_1 import (  # noqa: E402
    _atomic_json,
)
from scripts import run_verify_source_centered_rank_likelihood_template_1_1 as runner  # noqa: E402
from scripts.render_source_centered_rank_likelihood_template_visualizations import (  # noqa: E402
    COMPLETE_SCHEMA,
    CONFIG_SHA256,
    DATASETS,
    EXPERIMENT,
    METHOD_INTERPRETATION_RELATIVE_PATHS,
    PANEL_TITLES,
    REPORTING_DEPENDENCY_RELATIVE_PATHS,
    RESULT_SCHEMA,
    VISUALIZATION_SCHEMA,
    _authenticate_config,
    _read_self_hashed_json,
)


DELIVERY_QA_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization_delivery_qa.v1"
)
VISUAL_REVIEW_SCHEMA = (
    "pathline_template_matching.source_centered_rank_likelihood_visualization_visual_review.v1"
)
VISUAL_CHECKS = (
    "no_empty_or_clipped_panel",
    "camera_and_bounds_identical_across_panels",
    "both_pathline_block_encodings_visible",
    "no_legend_or_title_collision",
    "panel_b_and_panel_c_use_identical_center_population_and_order",
)
TOOL_NAMES = (
    "validate_figure.py",
    "audit_panel_alignment.py",
    "audit_pdf_text.py",
    "audit_figure_collisions.py",
)
EXPECTED_PNG_SIZE = (7560, 1800)
EXPECTED_SOURCE_WARNINGS = ("EXPORT-RASTER", "RASTER-DPI", "FINAL-WIDTH")
AUDITOR_RELATIVE_PATH = (
    "scripts/audit_source_centered_rank_likelihood_template_visualizations.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_output_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    _require(
        bool(relative)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and ":" not in pure.parts[0],
        f"unsafe report-relative path: {relative}",
    )
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"report path escapes output root: {relative}") from exc
    return candidate


def _authenticate_machine_bundle(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    _require(root.is_dir(), f"machine-render root does not exist: {root}")
    config = _authenticate_config()
    frozen_config = root / "frozen_config.yaml"
    _require(
        frozen_config.is_file()
        and sha256_file(frozen_config) == CONFIG_SHA256
        and frozen_config.read_bytes()
        == (REPOSITORY_ROOT / "config" / f"{EXPERIMENT}.yaml").read_bytes(),
        "machine-render frozen config changed",
    )
    result_path = root / "result_manifest.json"
    complete_path = root / "RUN_COMPLETE.json"
    visualization_path = root / "visualization_manifest.json"
    result = _read_self_hashed_json(result_path)
    complete = _read_self_hashed_json(complete_path)
    visualization = _read_self_hashed_json(visualization_path)
    _require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("experiment") == EXPERIMENT
        and result.get("status") == "completed_pending_local_rendered_qa"
        and result.get("formal_confirmation") is False
        and result.get("plotted_arm") == "dual_histogram_llr"
        and tuple(result.get("table_only_control_arms", ()))
        == ("negative_ecdf", "direct_rank_mean_top5")
        and result.get("table_only_control_metrics_reported") is True
        and int(result.get("figure_count", -1)) == 4,
        "machine-render result identity/status changed",
    )
    _require(
        complete.get("schema") == COMPLETE_SCHEMA
        and complete.get("experiment") == EXPERIMENT
        and complete.get("status") == "complete_pending_local_rendered_qa"
        and complete.get("plotted_arm") == "dual_histogram_llr"
        and int(complete.get("figure_count", -1)) == 4
        and complete.get("result_manifest_file_sha256") == sha256_file(result_path)
        and complete.get("result_manifest_content_sha256") == result.get("content_sha256"),
        "machine completion does not bind the pending-QA result",
    )
    _require(
        visualization.get("schema") == VISUALIZATION_SCHEMA
        and visualization.get("experiment") == EXPERIMENT
        and visualization.get("formal_confirmation") is False
        and visualization.get("plotted_arm") == "dual_histogram_llr"
        and visualization.get("controls_not_plotted") is True
        and tuple(visualization.get("table_only_control_arms", ()))
        == ("negative_ecdf", "direct_rank_mean_top5")
        and visualization.get("table_only_control_metrics_reported") is True
        and visualization.get("panel_b_is_not_FMT") is True
        and int(visualization.get("figure_count", -1)) == 4
        and visualization.get("classification_unit") == "combined-valid unique center"
        and visualization.get("scale_blocks_are_context_not_separate_classifiers") is True
        and visualization.get("primary_valid_projection_metrics_reported_not_plotted") is True,
        "visualization contract changed",
    )
    _require(
        result.get("visualization_manifest_file_sha256") == sha256_file(visualization_path)
        and result.get("input_manifest_file_sha256") == sha256_file(root / "input_manifest.json")
        and result.get("per_figure_metrics_file_sha256") == sha256_file(root / "per_figure_metrics.csv"),
        "result does not bind its global files",
    )
    input_manifest = _read_self_hashed_json(root / "input_manifest.json")
    _require(
        input_manifest.get("npz_array_access_before_manifest_write") is False
        and input_manifest.get("fold_sidecar_or_label_member_access") is False
        and input_manifest.get("all_18_files_authenticated_per_required_fold") is True,
        "machine input gate changed",
    )
    artifacts = result.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and int(result.get("artifact_count", -1)) == 33 == len(artifacts)
        and result.get("artifacts_content_sha256") == canonical_json_sha256(artifacts),
        "machine artifact transaction changed",
    )
    seen: set[str] = set()
    for row in artifacts:
        _require(isinstance(row, Mapping), "result artifact row is invalid")
        relative = str(row.get("relative_path", ""))
        _require(relative not in seen, f"duplicate artifact path: {relative}")
        seen.add(relative)
        path = _safe_output_path(root, relative)
        _require(
            path.is_file()
            and int(row.get("size_bytes", -1)) == path.stat().st_size
            and row.get("sha256") == sha256_file(path),
            f"machine artifact changed: {relative}",
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(actual == seen | {"result_manifest.json", "RUN_COMPLETE.json"}, "machine file set is not exactly 35")
    entries = visualization.get("entries")
    _require(
        isinstance(entries, list)
        and tuple(str(row.get("dataset")) for row in entries if isinstance(row, Mapping)) == DATASETS
        and all(
            row.get("plotted_arm") == "dual_histogram_llr"
            and tuple(row.get("table_only_control_arms", ()))
            == ("negative_ecdf", "direct_rank_mean_top5")
            for row in entries
        ),
        "four-figure dataset/arm order changed",
    )
    dependency_sha = result.get("reporting_dependency_sha256")
    _require(
        isinstance(dependency_sha, Mapping)
        and set(dependency_sha) == set(REPORTING_DEPENDENCY_RELATIVE_PATHS),
        "reporting dependency identity is incomplete",
    )
    for relative, digest in dependency_sha.items():
        _require(
            sha256_file(REPOSITORY_ROOT / str(relative)) == digest,
            f"local dependency differs from rendered commit: {relative}",
        )
    _require(
        config["qa_contract"]["delivery_status_before_local_qa"] == "not_deliverable",
        "frozen pre-QA state changed",
    )
    return result, complete, visualization


def _authenticate_local_qa_checkout(machine_result: Mapping[str, Any]) -> dict[str, Any]:
    expected_commit = str(machine_result.get("reporting_git_commit", ""))
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(status == "", "local QA checkout must be clean")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(commit == expected_commit, "local QA checkout differs from reporting commit")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", AUDITOR_RELATIVE_PATH],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().replace("\\", "/")
    _require(tracked == AUDITOR_RELATIVE_PATH, "local QA auditor is not tracked")
    dependencies = machine_result.get("reporting_dependency_sha256")
    _require(
        isinstance(dependencies, Mapping)
        and dependencies.get(AUDITOR_RELATIVE_PATH)
        == sha256_file(REPOSITORY_ROOT / AUDITOR_RELATIVE_PATH),
        "local QA auditor differs from the machine dependency",
    )
    method_commit = str(machine_result.get("method_interpretation_git_commit", ""))
    claimed_blobs = machine_result.get("method_interpretation_git_blob_sha1")
    _require(
        isinstance(claimed_blobs, Mapping)
        and set(claimed_blobs) == set(METHOD_INTERPRETATION_RELATIVE_PATHS)
        and machine_result.get(
            "method_interpretation_git_blob_sha1_content_sha256"
        )
        == canonical_json_sha256(claimed_blobs),
        "method interpretation blob evidence is incomplete",
    )
    for relative in METHOD_INTERPRETATION_RELATIVE_PATHS:
        method_blob = subprocess.run(
            ["git", "rev-parse", f"{method_commit}:{relative}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        reporting_blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{relative}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _require(
            method_blob == reporting_blob == claimed_blobs[relative],
            f"method interpretation blob changed: {relative}",
        )
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "auditor_relative_path": AUDITOR_RELATIVE_PATH,
        "auditor_sha256": dependencies[AUDITOR_RELATIVE_PATH],
    }


def _tool_paths(tool_root: Path) -> dict[str, Path]:
    paths = {name: tool_root.resolve() / name for name in TOOL_NAMES}
    _require(all(path.is_file() for path in paths.values()), "Nature-figure QA tools are incomplete")
    return paths


def _run_json_command(arguments: Sequence[str], *, label: str) -> dict[str, Any]:
    completed = subprocess.run(list(arguments), check=False, capture_output=True, text=True)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{label} did not emit JSON: {detail}") from exc
    _require(isinstance(value, dict), f"{label} JSON must contain one mapping")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or json.dumps(value, ensure_ascii=False)
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}: {detail}")
    return value


def _qa_temporary_path(destination: Path) -> Path:
    """Reserve a unique same-directory path for an external QA tool."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _png_header(path: Path) -> dict[str, Any]:
    prefix = path.read_bytes()[:33]
    _require(
        len(prefix) == 33
        and prefix[:8] == b"\x89PNG\r\n\x1a\n"
        and prefix[12:16] == b"IHDR",
        f"PNG header is invalid: {path}",
    )
    width = int.from_bytes(prefix[16:20], "big")
    height = int.from_bytes(prefix[20:24], "big")
    _require((width, height) == EXPECTED_PNG_SIZE, f"PNG dimensions changed: {path}")
    return {"width_pixels": width, "height_pixels": height, "bit_depth": int(prefix[24]), "color_type": int(prefix[25])}


def _source_warning_dispositions(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings = audit.get("findings")
    _require(isinstance(findings, list), "source-preflight findings are missing")
    warnings = [row for row in findings if isinstance(row, Mapping) and row.get("level") == "WARN"]
    _require(tuple(str(row.get("check_id")) for row in warnings) == EXPECTED_SOURCE_WARNINGS, "source warning set changed")
    rationale = {
        "EXPORT-RASTER": "Editable SVG/PDF are primary; TIFF is outside the frozen report contract.",
        "RASTER-DPI": "The required 7560x1800 PNG is exactly 360 dpi for the 21x5-inch canvas.",
        "FINAL-WIDTH": "The 21-inch reporting canvas preserves three readable 3D panels and is not a journal-column width.",
    }
    return [
        {
            "check_id": str(row["check_id"]),
            "source_level": "WARN",
            "disposition": "ACCEPTED_FROZEN_REPORT_REQUIREMENT",
            "rationale": rationale[str(row["check_id"])],
            "config_sha256": CONFIG_SHA256,
        }
        for row in warnings
    ]


def _audit_svg_editable_text(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"SVG is not well-formed XML: {path}") from exc
    elements = [item for item in root.iter() if str(item.tag).rsplit("}", 1)[-1] == "text"]
    text_values = ["".join(item.itertext()).strip() for item in elements if "".join(item.itertext()).strip()]
    joined = "\n".join(text_values)
    required = (*PANEL_TITLES, "legacy_2_1 first 120", "expanded_3_1 first 120")
    missing = [fragment for fragment in required if fragment not in joined]
    _require(elements and not missing and "FMT source-centered" not in joined, f"SVG editable text is incomplete or mislabeled: {missing}")
    return {
        "schema": "pathline_template_matching.source_centered_rank_likelihood_svg_text_audit.v1",
        "source_svg": path.name,
        "source_svg_sha256": sha256_file(path),
        "result": "PASS",
        "text_element_count": len(elements),
        "nonempty_text_element_count": len(text_values),
        "required_fragments": list(required),
        "missing_required_fragments": [],
        "text_content_sha256": canonical_json_sha256(text_values),
    }


def _load_visual_review(
    path: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    collision_warn_counts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], str]:
    payload = path.resolve().read_bytes()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    value = json.loads(payload.decode("utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == VISUAL_REVIEW_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("review_scope") == "every PNG at final 21x5-inch physical size"
        and value.get("result") == "PASS"
        and bool(str(value.get("reviewer", "")).strip())
        and bool(str(value.get("reviewed_at_utc", "")).strip()),
        "visual-review header is incomplete",
    )
    rows = value.get("entries")
    _require(
        isinstance(rows, list)
        and tuple(str(row.get("dataset")) for row in rows if isinstance(row, Mapping)) == DATASETS,
        "visual-review dataset order changed",
    )
    expected_png = {str(row["dataset"]): str(row["png_sha256"]) for row in entries}
    by_dataset: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "visual-review entry is invalid")
        dataset = str(row.get("dataset", ""))
        checks = row.get("checks")
        _require(
            dataset not in by_dataset
            and row.get("png_sha256") == expected_png.get(dataset)
            and row.get("result") == "PASS"
            and isinstance(checks, Mapping)
            and set(checks) == set(VISUAL_CHECKS)
            and all(checks[name] is True for name in VISUAL_CHECKS),
            f"visual-review checks are incomplete: {dataset}",
        )
        if collision_warn_counts[dataset] > 0:
            _require(
                row.get("collision_warning_review") == "accepted_after_final_size_review"
                and bool(str(row.get("notes", "")).strip()),
                f"collision warnings lack review: {dataset}",
            )
        else:
            _require(row.get("collision_warning_review") == "not_applicable_no_warnings", f"zero-warning state changed: {dataset}")
        by_dataset[dataset] = dict(row)
    return value, by_dataset, payload_sha256


def _reauthenticate_machine_artifacts(
    root: Path,
    *,
    result: Mapping[str, Any],
    complete: Mapping[str, Any],
    visualization: Mapping[str, Any],
) -> set[str]:
    """Recheck every immutable machine byte without rejecting new QA files."""

    _require(
        _read_self_hashed_json(root / "result_manifest.json") == result
        and _read_self_hashed_json(root / "RUN_COMPLETE.json") == complete
        and _read_self_hashed_json(root / "visualization_manifest.json")
        == visualization,
        "machine manifests changed during local QA",
    )
    expected = {"result_manifest.json", "RUN_COMPLETE.json"}
    for row in result["artifacts"]:
        relative = str(row["relative_path"])
        path = _safe_output_path(root, relative)
        _require(
            path.is_file()
            and int(row["size_bytes"]) == path.stat().st_size
            and row["sha256"] == sha256_file(path),
            f"machine artifact changed during local QA: {relative}",
        )
        expected.add(relative)
    return expected


def audit_delivery(
    *,
    output_root: Path,
    visual_review_path: Path,
    nature_figure_tool_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    result, complete, visualization = _authenticate_machine_bundle(output_root)
    local_checkout = _authenticate_local_qa_checkout(result)
    tools = _tool_paths(nature_figure_tool_root)
    tool_sha256 = {name: sha256_file(path) for name, path in tools.items()}
    source_path = REPOSITORY_ROOT / "src/pathline_template_matching/source_centered_rank_likelihood_visualization.py"
    source_audit_path = output_root / "source-preflight.json"
    summary_path = output_root / "delivery_qa_summary.json"
    entries = [dict(row) for row in visualization["entries"]]
    destinations = [source_audit_path, summary_path]
    for entry in entries:
        stem = _safe_output_path(output_root, str(entry["pdf"])).with_suffix("")
        destinations.extend(
            stem.with_suffix(suffix)
            for suffix in (
                ".alignment-audit.json",
                ".alignment-overlay.svg",
                ".svg-text-audit.json",
                ".pdf-text-audit.json",
                ".collision-audit.json",
                ".collision-overlay.pdf",
            )
        )
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite local QA evidence: {existing}")
    source_audit = _run_json_command(
        (sys.executable, str(tools["validate_figure.py"]), str(source_path), "--backend", "python", "--json"),
        label="source preflight",
    )
    _require(source_audit.get("summary", {}).get("ready") is True, "source preflight is not ready")
    warning_dispositions = _source_warning_dispositions(source_audit)
    _atomic_json(source_audit_path, source_audit)
    figure_audits: list[dict[str, Any]] = []
    collision_warn_counts: dict[str, int] = {}
    for entry in entries:
        dataset = str(entry["dataset"])
        _require(
            entry.get("population") == "combined_valid_unique_centers"
            and entry.get("plotted_arm") == "dual_histogram_llr"
            and int(entry.get("source_ordinal", -1)) == 2,
            f"figure population/arm/source changed: {dataset}",
        )
        png_path = _safe_output_path(output_root, str(entry["png"]))
        pdf_path = _safe_output_path(output_root, str(entry["pdf"]))
        svg_path = _safe_output_path(output_root, str(entry["svg"]))
        alignment_path = _safe_output_path(output_root, str(entry["alignment"]))
        for path, field in ((png_path, "png_sha256"), (pdf_path, "pdf_sha256"), (svg_path, "svg_sha256"), (alignment_path, "alignment_sha256")):
            _require(path.is_file() and sha256_file(path) == entry[field], f"figure export changed: {dataset}/{path.name}")
        png_audit = _png_header(png_path)
        stem = pdf_path.with_suffix("")
        svg_text = _audit_svg_editable_text(svg_path)
        svg_text_path = stem.with_suffix(".svg-text-audit.json")
        _atomic_json(svg_text_path, svg_text)
        alignment_audit_path = stem.with_suffix(".alignment-audit.json")
        alignment_overlay_path = stem.with_suffix(".alignment-overlay.svg")
        alignment_overlay_temporary = _qa_temporary_path(
            alignment_overlay_path
        )
        try:
            alignment = _run_json_command(
                (
                    sys.executable,
                    str(tools["audit_panel_alignment.py"]),
                    str(alignment_path),
                    "--json",
                    "--overlay-svg",
                    str(alignment_overlay_temporary),
                    "--tolerance-pt",
                    "1.5",
                    "--gutter-tolerance-pt",
                    "1.5",
                    "--require-panel-labels",
                    "--strict",
                ),
                label=f"panel alignment audit for {dataset}",
            )
            _atomic_json(alignment_audit_path, alignment)
            publish_file_without_overwrite(
                alignment_overlay_temporary, alignment_overlay_path
            )
        finally:
            alignment_overlay_temporary.unlink(missing_ok=True)
        _require(
            alignment.get("auditable") is True
            and alignment.get("verdict") == "PASS"
            and int(alignment.get("summary", {}).get("fail", -1)) == 0
            and int(alignment.get("summary", {}).get("warn", -1)) == 0,
            f"panel alignment failed: {dataset}",
        )
        pdf_text = _run_json_command(
            (sys.executable, str(tools["audit_pdf_text.py"]), str(pdf_path), "--min-pt", "5", "--json"),
            label=f"PDF text audit for {dataset}",
        )
        _require(
            pdf_text.get("auditable") is True
            and int(pdf_text.get("below_minimum_count", -1)) == 0
            and float(pdf_text.get("minimum_found_pt", 0.0)) >= 5.0,
            f"PDF text failed: {dataset}",
        )
        pdf_text_path = stem.with_suffix(".pdf-text-audit.json")
        _atomic_json(pdf_text_path, pdf_text)
        collision_path = stem.with_suffix(".collision-audit.json")
        collision_overlay_path = stem.with_suffix(".collision-overlay.pdf")
        collision_overlay_temporary = _qa_temporary_path(
            collision_overlay_path
        )
        try:
            collision = _run_json_command(
                (
                    sys.executable,
                    str(tools["audit_figure_collisions.py"]),
                    str(pdf_path),
                    "--json",
                    "--overlay-pdf",
                    str(collision_overlay_temporary),
                ),
                label=f"collision audit for {dataset}",
            )
            _atomic_json(collision_path, collision)
            if collision_overlay_temporary.stat().st_size > 0:
                publish_file_without_overwrite(
                    collision_overlay_temporary, collision_overlay_path
                )
            else:
                _require(
                    not collision.get("findings"),
                    f"collision overlay missing: {dataset}",
                )
                publish_bytes_without_overwrite(
                    collision_overlay_path, pdf_path.read_bytes()
                )
        finally:
            collision_overlay_temporary.unlink(missing_ok=True)
        _require(collision.get("auditable") is True and int(collision.get("summary", {}).get("fail", -1)) == 0, f"collision hard failure: {dataset}")
        warning_count = int(collision.get("summary", {}).get("warn", -1))
        _require(warning_count >= 0, f"collision warning count invalid: {dataset}")
        collision_warn_counts[dataset] = warning_count
        figure_audits.append(
            {
                "dataset": dataset,
                "png": {"path": entry["png"], "sha256": sha256_file(png_path), **png_audit},
                "alignment": {
                    "verdict": alignment["verdict"],
                    "fail": int(alignment["summary"]["fail"]),
                    "warn": int(alignment["summary"]["warn"]),
                    "audit_path": alignment_audit_path.relative_to(output_root).as_posix(),
                    "audit_sha256": sha256_file(alignment_audit_path),
                    "overlay_path": alignment_overlay_path.relative_to(output_root).as_posix(),
                    "overlay_sha256": sha256_file(alignment_overlay_path),
                },
                "pdf_text": {
                    "minimum_required_pt": float(pdf_text["minimum_required_pt"]),
                    "minimum_found_pt": float(pdf_text["minimum_found_pt"]),
                    "below_minimum_count": int(pdf_text["below_minimum_count"]),
                    "audit_path": pdf_text_path.relative_to(output_root).as_posix(),
                    "audit_sha256": sha256_file(pdf_text_path),
                },
                "svg_text": {
                    "result": svg_text["result"],
                    "text_element_count": int(svg_text["text_element_count"]),
                    "audit_path": svg_text_path.relative_to(output_root).as_posix(),
                    "audit_sha256": sha256_file(svg_text_path),
                },
                "collision": {
                    "verdict": collision["verdict"],
                    "fail": int(collision["summary"]["fail"]),
                    "warn": warning_count,
                    "audit_path": collision_path.relative_to(output_root).as_posix(),
                    "audit_sha256": sha256_file(collision_path),
                    "overlay_path": collision_overlay_path.relative_to(output_root).as_posix(),
                    "overlay_sha256": sha256_file(collision_overlay_path),
                },
            }
        )
    visual_review, reviews, visual_review_sha256 = _load_visual_review(
        visual_review_path,
        entries=entries,
        collision_warn_counts=collision_warn_counts,
    )
    for row in figure_audits:
        row["visual_review"] = reviews[row["dataset"]]
    machine_paths = _reauthenticate_machine_artifacts(
        output_root,
        result=result,
        complete=complete,
        visualization=visualization,
    )
    _require(
        _authenticate_local_qa_checkout(result) == local_checkout,
        "reporting checkout changed during local QA",
    )
    _require(
        all(sha256_file(path) == tool_sha256[name] for name, path in tools.items()),
        "Nature-figure QA tool changed during execution",
    )
    _require(
        sha256_file(visual_review_path.resolve()) == visual_review_sha256,
        "visual-review file changed during local QA",
    )
    summary = runner._manifest(
        {
            "schema": DELIVERY_QA_SCHEMA,
            "experiment": EXPERIMENT,
            "delivery_status": "PASS",
            "formal_confirmation": False,
            "machine_render_status": result["status"],
            "machine_completion_status": complete["status"],
            "reporting_git_commit": result["reporting_git_commit"],
            "local_qa_checkout": local_checkout,
            "frozen_report_config_sha256": CONFIG_SHA256,
            "result_manifest_sha256": sha256_file(output_root / "result_manifest.json"),
            "RUN_COMPLETE_sha256": sha256_file(output_root / "RUN_COMPLETE.json"),
            "visualization_manifest_sha256": sha256_file(output_root / "visualization_manifest.json"),
            "source_preflight": {
                "source": source_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "source_sha256": sha256_file(source_path),
                "audit_path": source_audit_path.relative_to(output_root).as_posix(),
                "audit_sha256": sha256_file(source_audit_path),
                "ready": True,
                "counts": source_audit["summary"]["counts"],
                "warning_dispositions": warning_dispositions,
            },
            "qa_auditor": {
                "relative_path": AUDITOR_RELATIVE_PATH,
                "sha256": local_checkout["auditor_sha256"],
                "git_commit": local_checkout["git_commit"],
            },
            "qa_tools": {
                name: {"path": str(path.resolve()), "sha256": tool_sha256[name]}
                for name, path in tools.items()
            },
            "visual_review": {
                "path": str(visual_review_path.resolve()),
                "sha256": visual_review_sha256,
                "schema": visual_review["schema"],
                "reviewer": visual_review["reviewer"],
                "reviewed_at_utc": visual_review["reviewed_at_utc"],
                "result": visual_review["result"],
            },
            "figure_count": len(figure_audits),
            "panel_alignment_strict_pass_count": sum(row["alignment"]["verdict"] == "PASS" for row in figure_audits),
            "pdf_text_pass_count": sum(row["pdf_text"]["below_minimum_count"] == 0 for row in figure_audits),
            "svg_editable_text_pass_count": sum(row["svg_text"]["result"] == "PASS" for row in figure_audits),
            "collision_hard_fail_count": sum(row["collision"]["fail"] for row in figure_audits),
            "collision_warning_count": sum(row["collision"]["warn"] for row in figure_audits),
            "visual_review_pass_count": len(reviews),
            "figures": figure_audits,
            "evidence_scope": "family-held-out exposed-development fixed-source visualization only",
        }
    )
    _require(
        len(figure_audits) == 4
        and summary["panel_alignment_strict_pass_count"] == 4
        and summary["pdf_text_pass_count"] == 4
        and summary["svg_editable_text_pass_count"] == 4
        and summary["collision_hard_fail_count"] == 0
        and summary["visual_review_pass_count"] == 4,
        "delivery QA population is incomplete",
    )
    _atomic_json(summary_path, summary)
    expected_qa_paths = {
        path.relative_to(output_root).as_posix() for path in destinations
    }
    actual_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    _require(
        actual_paths == machine_paths | expected_qa_paths
        and len(machine_paths) == 35
        and len(expected_qa_paths) == 26
        and len(actual_paths) == 61,
        "final machine-plus-QA file set changed",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-review-json", type=Path, required=True)
    parser.add_argument("--nature-figure-tool-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    audit_delivery(
        output_root=arguments.output_root,
        visual_review_path=arguments.visual_review_json,
        nature_figure_tool_root=arguments.nature_figure_tool_root,
    )


if __name__ == "__main__":
    main()
