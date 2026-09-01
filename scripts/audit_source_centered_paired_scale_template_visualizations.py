#!/usr/bin/env python3
"""Finish immutable local QA for the four source-centered triptychs.

The machine render is deliberately not deliverable until this script verifies
its authenticated artifact transaction, runs the external Nature-figure
audits, and binds a human final-size review of every PNG.  It does not open
prediction, feature, label, parent-scene, or combined-scene NPZ members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.phase21_pipeline import (  # noqa: E402
    _atomic_bytes,
    _atomic_json,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from scripts import run_verify_source_centered_paired_scale_template_1_1 as runner  # noqa: E402
from scripts.render_source_centered_paired_scale_template_visualizations import (  # noqa: E402
    CONFIG_SHA256,
    DATASETS,
    EXPERIMENT,
    PANEL_TITLES,
    REPORTING_DEPENDENCY_RELATIVE_PATHS,
    _authenticate_config,
    _read_self_hashed_json,
)


DELIVERY_QA_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_visualization_delivery_qa.v1"
)
VISUAL_REVIEW_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_visualization_visual_review.v1"
)
RESULT_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_visualization_result.v1"
)
COMPLETE_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_visualization_run_complete.v1"
)
VISUALIZATION_SCHEMA = (
    "pathline_template_matching.source_centered_paired_scale_visualization.v1"
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
EXPECTED_SOURCE_WARNINGS = (
    "EXPORT-RASTER",
    "RASTER-DPI",
    "FINAL-WIDTH",
)
AUDITOR_RELATIVE_PATH = (
    "scripts/audit_source_centered_paired_scale_template_visualizations.py"
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
        and int(result.get("figure_count", -1)) == 4,
        "machine-render result identity/status changed",
    )
    _require(
        complete.get("schema") == COMPLETE_SCHEMA
        and complete.get("experiment") == EXPERIMENT
        and complete.get("status") == "complete_pending_local_rendered_qa"
        and int(complete.get("figure_count", -1)) == 4
        and complete.get("result_manifest_file_sha256") == sha256_file(result_path)
        and complete.get("result_manifest_content_sha256") == result.get("content_sha256"),
        "machine-render completion does not bind the pending-QA result",
    )
    _require(
        visualization.get("schema") == VISUALIZATION_SCHEMA
        and visualization.get("experiment") == EXPERIMENT
        and visualization.get("formal_confirmation") is False
        and int(visualization.get("figure_count", -1)) == 4
        and visualization.get("classification_unit") == "combined-valid unique center"
        and visualization.get("scale_blocks_are_context_not_separate_classifiers") is True
        and visualization.get("primary_valid_projection_metrics_reported_not_plotted") is True,
        "visualization contract changed",
    )
    _require(
        result.get("visualization_manifest_file_sha256") == sha256_file(visualization_path)
        and result.get("input_manifest_file_sha256")
        == sha256_file(root / "input_manifest.json")
        and result.get("per_figure_metrics_file_sha256")
        == sha256_file(root / "per_figure_metrics.csv"),
        "result does not bind its global report files",
    )
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, list), "result artifact list is missing")
    _require(
        int(result.get("artifact_count", -1)) == len(artifacts)
        and result.get("artifacts_content_sha256") == canonical_json_sha256(artifacts),
        "result artifact list digest changed",
    )
    seen: set[str] = set()
    for row in artifacts:
        _require(isinstance(row, Mapping), "result artifact row is invalid")
        relative = str(row.get("relative_path", ""))
        _require(relative not in seen, f"duplicate result artifact path: {relative}")
        seen.add(relative)
        path = _safe_output_path(root, relative)
        _require(
            path.is_file()
            and int(row.get("size_bytes", -1)) == path.stat().st_size
            and row.get("sha256") == sha256_file(path),
            f"machine-render artifact changed: {relative}",
        )
    entries = visualization.get("entries")
    _require(
        isinstance(entries, list)
        and tuple(str(row.get("dataset")) for row in entries if isinstance(row, Mapping))
        == DATASETS,
        "four-figure dataset order changed",
    )
    reporting_dependency_sha256 = result.get("reporting_dependency_sha256")
    _require(
        isinstance(reporting_dependency_sha256, Mapping)
        and set(reporting_dependency_sha256) == set(REPORTING_DEPENDENCY_RELATIVE_PATHS),
        "reporting dependency identity is incomplete",
    )
    for relative, digest in reporting_dependency_sha256.items():
        _require(
            sha256_file(REPOSITORY_ROOT / str(relative)) == digest,
            f"local reporting dependency differs from the rendered commit: {relative}",
        )
    _require(
        config["qa_contract"]["delivery_status_before_local_qa"] == "not_deliverable",
        "frozen pre-QA delivery state changed",
    )
    return result, complete, visualization


def _authenticate_local_qa_checkout(
    machine_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind local QA to the exact clean checkout used for machine rendering."""

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
    _require(tracked == AUDITOR_RELATIVE_PATH, "local QA auditor is not tracked exactly")
    dependency_sha256 = machine_result.get("reporting_dependency_sha256")
    _require(
        isinstance(dependency_sha256, Mapping)
        and dependency_sha256.get(AUDITOR_RELATIVE_PATH)
        == sha256_file(REPOSITORY_ROOT / AUDITOR_RELATIVE_PATH),
        "local QA auditor differs from the machine reporting dependency",
    )
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "auditor_relative_path": AUDITOR_RELATIVE_PATH,
        "auditor_sha256": dependency_sha256[AUDITOR_RELATIVE_PATH],
    }


def _tool_paths(tool_root: Path) -> dict[str, Path]:
    tool_root = tool_root.resolve()
    paths = {name: tool_root / name for name in TOOL_NAMES}
    _require(all(path.is_file() for path in paths.values()), "Nature-figure QA tool set is incomplete")
    return paths


def _run_json_command(arguments: Sequence[str], *, label: str) -> dict[str, Any]:
    completed = subprocess.run(
        list(arguments), check=False, capture_output=True, text=True
    )
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
    return {
        "width_pixels": width,
        "height_pixels": height,
        "bit_depth": int(prefix[24]),
        "color_type": int(prefix[25]),
    }


def _source_preflight_warning_dispositions(
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    findings = audit.get("findings")
    _require(isinstance(findings, list), "source-preflight findings are missing")
    warnings = [
        row
        for row in findings
        if isinstance(row, Mapping) and row.get("level") == "WARN"
    ]
    _require(
        tuple(str(row.get("check_id")) for row in warnings)
        == EXPECTED_SOURCE_WARNINGS,
        "source-preflight warning set changed and lacks a frozen disposition",
    )
    rationale = {
        "EXPORT-RASTER": (
            "Accepted by the frozen export contract: editable SVG/PDF are the "
            "primary report artifacts and the required raster is a 360 dpi PNG preview; "
            "TIFF is not a required output."
        ),
        "RASTER-DPI": (
            "Accepted by the frozen export contract: every PNG must be exactly "
            "7560x1800 pixels for a 21x5 inch canvas, which is 360 dpi and exceeds "
            "the 300 dpi floor."
        ),
        "FINAL-WIDTH": (
            "Accepted by the frozen four-flow reporting contract: 21 inches "
            "supports three readable 3D panels and is not presented as a journal-column width."
        ),
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
    """Require real SVG text elements for titles, panel labels, and block key."""

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"SVG is not well-formed XML: {path}") from exc
    text_elements = [
        element
        for element in root.iter()
        if str(element.tag).rsplit("}", 1)[-1] == "text"
    ]
    text_values = [
        "".join(element.itertext()).strip()
        for element in text_elements
        if "".join(element.itertext()).strip()
    ]
    joined = "\n".join(text_values)
    required_fragments = (
        *PANEL_TITLES,
        "legacy_2_1 first 120",
        "expanded_3_1 first 120",
    )
    missing = [fragment for fragment in required_fragments if fragment not in joined]
    _require(text_elements and not missing, f"SVG editable text is incomplete: {missing}")
    return {
        "schema": "pathline_template_matching.source_centered_svg_editable_text_audit.v1",
        "source_svg": path.name,
        "source_svg_sha256": sha256_file(path),
        "result": "PASS",
        "text_element_count": len(text_elements),
        "nonempty_text_element_count": len(text_values),
        "required_fragments": list(required_fragments),
        "missing_required_fragments": [],
        "text_content_sha256": canonical_json_sha256(text_values),
    }


def _load_visual_review(
    path: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    collision_warn_counts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    path = path.resolve()
    _require(path.is_file(), f"visual-review JSON does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "visual-review JSON must contain one mapping")
    _require(
        value.get("schema") == VISUAL_REVIEW_SCHEMA
        and value.get("experiment") == EXPERIMENT
        and value.get("review_scope") == "every PNG at final 21x5-inch physical size"
        and value.get("result") == "PASS"
        and isinstance(value.get("reviewer"), str)
        and bool(value["reviewer"].strip())
        and isinstance(value.get("reviewed_at_utc"), str)
        and bool(value["reviewed_at_utc"].strip()),
        "visual-review header is incomplete or not PASS",
    )
    rows = value.get("entries")
    _require(
        isinstance(rows, list)
        and tuple(str(row.get("dataset")) for row in rows if isinstance(row, Mapping))
        == DATASETS,
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
        warning_review = row.get("collision_warning_review")
        if collision_warn_counts[dataset] > 0:
            _require(
                warning_review == "accepted_after_final_size_review"
                and isinstance(row.get("notes"), str)
                and bool(row["notes"].strip()),
                f"collision warnings lack documented final-size review: {dataset}",
            )
        else:
            _require(
                warning_review == "not_applicable_no_warnings",
                f"zero-warning collision review state changed: {dataset}",
            )
        by_dataset[dataset] = dict(row)
    return value, by_dataset


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
    source_path = REPOSITORY_ROOT / "src/pathline_template_matching/source_centered_visualization.py"
    source_audit_path = output_root / "source-preflight.json"
    summary_path = output_root / "delivery_qa_summary.json"
    destinations = [source_audit_path, summary_path]
    entries = [dict(row) for row in visualization["entries"]]
    for entry in entries:
        stem = _safe_output_path(output_root, str(entry["pdf"])).with_suffix("")
        destinations.extend(
            (
                stem.with_suffix(".alignment-audit.json"),
                stem.with_suffix(".alignment-overlay.svg"),
                stem.with_suffix(".svg-text-audit.json"),
                stem.with_suffix(".pdf-text-audit.json"),
                stem.with_suffix(".collision-audit.json"),
                stem.with_suffix(".collision-overlay.pdf"),
            )
        )
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite local QA evidence: {existing}")

    source_audit = _run_json_command(
        (
            sys.executable,
            str(tools["validate_figure.py"]),
            str(source_path),
            "--backend",
            "python",
            "--json",
        ),
        label="source preflight",
    )
    _require(source_audit.get("summary", {}).get("ready") is True, "source preflight is not ready")
    warning_dispositions = _source_preflight_warning_dispositions(source_audit)
    _atomic_json(source_audit_path, source_audit)

    figure_audits: list[dict[str, Any]] = []
    collision_warn_counts: dict[str, int] = {}
    for entry in entries:
        dataset = str(entry["dataset"])
        _require(
            entry.get("population") == "combined_valid_unique_centers"
            and int(entry.get("source_ordinal", -1)) == 2,
            f"figure population/source changed: {dataset}",
        )
        png_path = _safe_output_path(output_root, str(entry["png"]))
        pdf_path = _safe_output_path(output_root, str(entry["pdf"]))
        svg_path = _safe_output_path(output_root, str(entry["svg"]))
        alignment_path = _safe_output_path(output_root, str(entry["alignment"]))
        for path, field in (
            (png_path, "png_sha256"),
            (pdf_path, "pdf_sha256"),
            (svg_path, "svg_sha256"),
            (alignment_path, "alignment_sha256"),
        ):
            _require(
                path.is_file() and sha256_file(path) == entry[field],
                f"figure export changed: {dataset}/{path.name}",
            )
        png_audit = _png_header(png_path)
        stem = pdf_path.with_suffix("")
        svg_text = _audit_svg_editable_text(svg_path)
        svg_text_path = stem.with_suffix(".svg-text-audit.json")
        _atomic_json(svg_text_path, svg_text)
        alignment_audit_path = stem.with_suffix(".alignment-audit.json")
        alignment_overlay_path = stem.with_suffix(".alignment-overlay.svg")
        alignment_audit = _run_json_command(
            (
                sys.executable,
                str(tools["audit_panel_alignment.py"]),
                str(alignment_path),
                "--json",
                "--json-out",
                str(alignment_audit_path),
                "--overlay-svg",
                str(alignment_overlay_path),
                "--tolerance-pt",
                "1.5",
                "--gutter-tolerance-pt",
                "1.5",
                "--require-panel-labels",
                "--strict",
            ),
            label=f"panel alignment audit for {dataset}",
        )
        _require(
            alignment_audit.get("auditable") is True
            and alignment_audit.get("verdict") == "PASS"
            and int(alignment_audit.get("summary", {}).get("fail", -1)) == 0
            and int(alignment_audit.get("summary", {}).get("warn", -1)) == 0,
            f"panel alignment did not pass strictly: {dataset}",
        )
        pdf_text = _run_json_command(
            (
                sys.executable,
                str(tools["audit_pdf_text.py"]),
                str(pdf_path),
                "--min-pt",
                "5",
                "--json",
            ),
            label=f"PDF text audit for {dataset}",
        )
        _require(
            pdf_text.get("auditable") is True
            and int(pdf_text.get("below_minimum_count", -1)) == 0
            and float(pdf_text.get("minimum_found_pt", 0.0)) >= 5.0,
            f"PDF text did not pass the 5 pt floor: {dataset}",
        )
        pdf_text_path = stem.with_suffix(".pdf-text-audit.json")
        _atomic_json(pdf_text_path, pdf_text)
        collision_path = stem.with_suffix(".collision-audit.json")
        collision_overlay_path = stem.with_suffix(".collision-overlay.pdf")
        collision = _run_json_command(
            (
                sys.executable,
                str(tools["audit_figure_collisions.py"]),
                str(pdf_path),
                "--json",
                "--json-out",
                str(collision_path),
                "--overlay-pdf",
                str(collision_overlay_path),
            ),
            label=f"rendered collision audit for {dataset}",
        )
        _require(
            collision.get("auditable") is True
            and int(collision.get("summary", {}).get("fail", -1)) == 0,
            f"rendered collision hard failure: {dataset}",
        )
        warning_count = int(collision.get("summary", {}).get("warn", -1))
        _require(warning_count >= 0, f"collision warning count is invalid: {dataset}")
        collision_warn_counts[dataset] = warning_count
        if not collision_overlay_path.exists():
            _require(not collision.get("findings"), f"collision overlay is missing: {dataset}")
            _atomic_bytes(collision_overlay_path, pdf_path.read_bytes())
        figure_audits.append(
            {
                "dataset": dataset,
                "png": {
                    "path": entry["png"],
                    "sha256": sha256_file(png_path),
                    **png_audit,
                },
                "alignment": {
                    "verdict": alignment_audit["verdict"],
                    "fail": int(alignment_audit["summary"]["fail"]),
                    "warn": int(alignment_audit["summary"]["warn"]),
                    "comparisons": int(alignment_audit["summary"]["comparisons"]),
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
                    "empty_overlay_is_authenticated_pdf_copy": not bool(collision.get("findings")),
                },
            }
        )

    visual_review, reviews = _load_visual_review(
        visual_review_path,
        entries=entries,
        collision_warn_counts=collision_warn_counts,
    )
    for row in figure_audits:
        row["visual_review"] = reviews[row["dataset"]]
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
            "visualization_manifest_sha256": sha256_file(
                output_root / "visualization_manifest.json"
            ),
            "source_preflight": {
                "source": source_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "source_sha256": sha256_file(source_path),
                "audit_path": source_audit_path.relative_to(output_root).as_posix(),
                "audit_sha256": sha256_file(source_audit_path),
                "ready": True,
                "counts": source_audit["summary"]["counts"],
                "warning_dispositions": warning_dispositions,
                "all_warnings_have_explicit_dispositions": True,
            },
            "qa_auditor": {
                "relative_path": AUDITOR_RELATIVE_PATH,
                "sha256": local_checkout["auditor_sha256"],
                "git_commit": local_checkout["git_commit"],
                "worktree_clean": local_checkout["worktree_clean"],
            },
            "qa_tools": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in tools.items()
            },
            "visual_review": {
                "path": str(visual_review_path.resolve()),
                "sha256": sha256_file(visual_review_path.resolve()),
                "schema": visual_review["schema"],
                "reviewer": visual_review["reviewer"],
                "reviewed_at_utc": visual_review["reviewed_at_utc"],
                "result": visual_review["result"],
            },
            "figure_count": len(figure_audits),
            "panel_alignment_strict_pass_count": sum(
                row["alignment"]["verdict"] == "PASS" for row in figure_audits
            ),
            "pdf_text_pass_count": sum(
                row["pdf_text"]["below_minimum_count"] == 0 for row in figure_audits
            ),
            "svg_editable_text_pass_count": sum(
                row["svg_text"]["result"] == "PASS" for row in figure_audits
            ),
            "collision_hard_fail_count": sum(
                row["collision"]["fail"] for row in figure_audits
            ),
            "collision_warning_count": sum(
                row["collision"]["warn"] for row in figure_audits
            ),
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
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-review-json", type=Path, required=True)
    parser.add_argument(
        "--nature-figure-tool-root",
        type=Path,
        required=True,
        help="directory containing the four frozen Nature-figure QA scripts",
    )
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
