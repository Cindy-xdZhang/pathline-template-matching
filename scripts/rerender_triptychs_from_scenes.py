#!/usr/bin/env python3
"""Rerender completed triptychs from immutable scene artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.development_data import (
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.development_report import DATASET_VIEWS
from pathline_template_matching.visualization import (
    DEFAULT_DPI,
    render_template_matching_triptych,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("renderer worktree is dirty; commit before rerendering")


def _load_scene(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["ivd_mesh_vertices"], dtype=np.float32)
        faces = np.asarray(data["ivd_mesh_faces"], dtype=np.int64)
        level = float(data["ivd_mesh_level"])
        mesh = None
        if len(vertices):
            mesh = {"vertices": vertices, "faces": faces, "level": level}
        pathlines = np.asarray(data["display_pathlines"], dtype=np.float32)
        return {
            "dataset": str(data["dataset"]),
            "title": str(data["title"]),
            "regime": str(data["regime"]),
            "source_ordinal": int(data["source_ordinal"]),
            "bounds": np.asarray(data["bounds"], dtype=np.float64),
            "seeds": np.asarray(data["seeds"], dtype=np.float32),
            "reference": np.asarray(data["reference"], dtype=bool),
            "prediction": np.asarray(data["prediction"], dtype=bool),
            "display_pathlines": [path for path in pathlines],
            "ivd_points": None,
            "ivd_mesh": mesh,
        }


def rerender(run_dir: Path, output_name: str, dpi: int) -> Path:
    run_dir = run_dir.resolve()
    source_visualization_path = run_dir / "figures" / "visualization_manifest.json"
    result_manifest_path = run_dir / "result_manifest.json"
    run_state_path = run_dir / "run_state.json"
    source_visualization = json.loads(source_visualization_path.read_text(encoding="utf-8"))
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    if result_manifest.get("status") != "development_completed_confirmation_not_run":
        raise RuntimeError("source run is not a completed development result")
    result_content_sha256 = canonical_json_sha256(
        {
            key: value
            for key, value in result_manifest.items()
            if key != "manifest_content_sha256"
        }
    )
    if result_content_sha256 != result_manifest.get("manifest_content_sha256"):
        raise RuntimeError("source result manifest content digest is invalid")
    result_manifest_file_sha256 = sha256_file(result_manifest_path)
    if (
        result_manifest_file_sha256
        != run_state.get("result_manifest_file_sha256")
        or result_content_sha256
        != run_state.get("result_manifest_content_sha256")
    ):
        raise RuntimeError("run state does not anchor the source result manifest")
    source_visualization_sha256 = sha256_file(source_visualization_path)
    if source_visualization_sha256 != result_manifest["outputs"][
        "figures/visualization_manifest.json"
    ]["sha256"]:
        raise RuntimeError("result manifest does not anchor visualization manifest")
    source_figures = source_visualization["figures"]
    keys = [(str(item["dataset"]), str(item["regime"])) for item in source_figures]
    expected_keys = {
        (dataset, regime)
        for dataset in DATASET_VIEWS
        for regime in ("seen_scale", "unseen_scale")
    }
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise RuntimeError("source visualization manifest is not the frozen 10×2 set")
    result_artifacts = {
        (str(item["dataset"]), str(item["regime"])): item
        for item in result_manifest["visualization_artifacts"]
    }
    if set(result_artifacts) != expected_keys:
        raise RuntimeError("result manifest visualization keys are incomplete")
    for item in source_figures:
        key = (str(item["dataset"]), str(item["regime"]))
        anchored = result_artifacts[key]
        if (
            str(item["image_sha256"]) != str(anchored["image_sha256"])
            or str(item["scene_artifact_sha256"])
            != str(anchored["scene_artifact_sha256"])
        ):
            raise RuntimeError(f"result/visualization artifact mismatch: {key}")

    _require_clean_worktree()
    output_dir = run_dir / output_name
    output_dir.mkdir(parents=False, exist_ok=False)
    renderer_commit = _git_commit()
    rendered: list[dict[str, Any]] = []
    for item in sorted(source_figures, key=lambda row: (row["dataset"], row["regime"])):
        dataset = str(item["dataset"])
        regime = str(item["regime"])
        source_scene = run_dir / "figures" / "scenes" / f"{dataset}_{regime}_scene.npz"
        scene_sha256 = sha256_file(source_scene)
        if scene_sha256 != str(item["scene_artifact_sha256"]):
            raise RuntimeError(f"scene artifact digest mismatch: {source_scene}")
        source_image = run_dir / "figures" / f"{dataset}_{regime}_triptych.png"
        if sha256_file(source_image) != str(item["image_sha256"]):
            raise RuntimeError(f"source image digest mismatch: {source_image}")

        output_image = output_dir / source_image.name
        _, metadata = render_template_matching_triptych(
            _load_scene(source_scene),
            output_image,
            view=DATASET_VIEWS[dataset],
            dpi=int(dpi),
        )
        metadata.update(
            {
                "image_sha256": sha256_file(output_image),
                "layout_only_rerender": True,
                "metric_recomputation": False,
                "evidence_scope": item["evidence_scope"],
                "metric_based_selection": item["metric_based_selection"],
                "all_scale_tuples_included": item[
                    "all_scale_tuples_included"
                ],
                "canonical_scale_names": item["canonical_scale_names"],
                "source_scene_artifact": str(source_scene),
                "source_scene_artifact_sha256": scene_sha256,
                "source_image": str(source_image),
                "source_image_sha256": str(item["image_sha256"]),
                "raw_ivd_audit": item["raw_ivd_audit"],
                "display_pathline_selection": item["display_pathline_selection"],
                "selected_pathline_seed_indices_sha256": item[
                    "selected_pathline_seed_indices_sha256"
                ],
            }
        )
        rendered.append(metadata)
        print(f"[rerender] {dataset}/{regime}: {output_image}", flush=True)

    manifest = {
        "schema_version": 1,
        "artifact_id": "Other_MainExp12FigureLayout_1.1",
        "purpose": "layout-only rerender with uncropped titles",
        "source_experiment": result_manifest["experiment"],
        "source_numerical_git_commit": result_manifest["git_commit"],
        "source_result_manifest_file_sha256": result_manifest_file_sha256,
        "source_result_manifest_content_sha256": result_manifest[
            "manifest_content_sha256"
        ],
        "source_visualization_manifest_sha256": source_visualization_sha256,
        "renderer_git_commit": renderer_commit,
        "renderer_script_sha256": sha256_file(Path(__file__)),
        "renderer_module_sha256": sha256_file(
            ROOT / "src" / "pathline_template_matching" / "visualization.py"
        ),
        "dpi": int(dpi),
        "figure_count": len(rendered),
        "metric_recomputation": False,
        "figures": rendered,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "rerender manifest content SHA-256="
        f"{manifest['manifest_content_sha256']}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name", default="figures_layout_rerender_1.1"
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()
    rerender(args.run_dir, str(args.output_name), int(args.dpi))


if __name__ == "__main__":
    main()
