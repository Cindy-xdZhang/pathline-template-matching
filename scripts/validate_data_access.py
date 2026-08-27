"""Validate raw-flow and legacy-cache access from the project dataset registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.data_access import validate_dataset_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="config/datasets.yaml")
    parser.add_argument("--environment", choices=("local", "ibex"), required=True)
    parser.add_argument("--read-raw-sample", action="store_true")
    parser.add_argument("--sample-max-spatial-dim", type=int, default=8)
    parser.add_argument("--output")
    parser.add_argument(
        "--require-all-raw",
        action="store_true",
        help="Fail unless every registry entry has an accessible raw field.",
    )
    parser.add_argument(
        "--require-all-cache",
        action="store_true",
        help="Fail unless every registry entry has both fully validated cache phases.",
    )
    args = parser.parse_args()
    report = validate_dataset_registry(
        args.registry,
        args.environment,
        read_raw_sample=args.read_raw_sample,
        raw_sample_max_spatial_dim=args.sample_max_spatial_dim,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    summary = report["summary"]
    required = summary["dataset_count"] if args.require_all_raw else 0
    if args.require_all_raw and summary["raw_accessible_count"] != required:
        return 2
    if (
        args.require_all_cache
        and (
            summary["cache_accessible_count"] != summary["dataset_count"]
            or not summary["cache_config_consistent"]
        )
    ):
        return 4
    if summary["bootstrap_usable_count"] != summary["dataset_count"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
