#!/usr/bin/env python3
"""Run or finalize the current cache-backed template-matching development evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.development_experiment import run_all_folds
from pathline_template_matching.development_report import finalize_development_run


DEFAULT_CONFIG = ROOT / "config/mainExp_TemplateMatching_1.2_development.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("run", "finalize", "run-and-finalize"), default="run-and-finalize"
    )
    parser.add_argument("--environment", default="ibex")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--figure-dpi", type=int, default=360)
    args = parser.parse_args()

    if args.mode in {"run", "run-and-finalize"}:
        run_all_folds(
            args.config,
            args.run_dir,
            environment=args.environment,
            device=args.device,
            resume=args.resume,
        )
    if args.mode in {"finalize", "run-and-finalize"}:
        result = finalize_development_run(
            args.config,
            args.run_dir,
            render_environment=args.environment,
            figure_dpi=args.figure_dpi,
        )
        print(
            "development run complete; formal confirmation not run; "
            f"result manifest: {args.run_dir / 'result_manifest.json'} "
            f"(content SHA-256 {result['manifest_content_sha256']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
