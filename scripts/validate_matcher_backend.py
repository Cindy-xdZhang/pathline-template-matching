#!/usr/bin/env python3
"""Fail-fast CPU/CUDA equivalence audit for exhaustive 1NN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.matcher import ExhaustiveOneNearestNeighbor


def run(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(25068)
    library = rng.normal(size=(256, 672)).astype(np.float32)
    labels = np.repeat(np.asarray([False, True]), 128)
    # Force an exact duplicate across classes.  This audits zero distance and
    # the frozen non-vortex rule without relying on a numerically approximate
    # midpoint tie.
    library[128] = library[0]
    queries = rng.normal(size=(64, 672)).astype(np.float32)
    queries[0] = library[0]
    queries[1] = library[129]
    cpu = ExhaustiveOneNearestNeighbor(library, labels, device="cpu").query(
        queries, query_chunk_size=17, library_chunk_size=61
    )
    tested = ExhaustiveOneNearestNeighbor(library, labels, device=device).query(
        queries, query_chunk_size=17, library_chunk_size=61
    )
    if cpu.nearest_negative_distances[0] != 0.0 or cpu.nearest_positive_distances[0] != 0.0:
        raise RuntimeError("direct-distance backend failed exact duplicate zero-distance audit")
    if bool(cpu.labels[0]):
        raise RuntimeError("cross-class exact tie was not assigned to non-vortex")
    if not np.array_equal(cpu.labels, tested.labels):
        mismatch = np.flatnonzero(cpu.labels != tested.labels)
        raise RuntimeError(f"CPU/{device} prediction mismatch at rows {mismatch.tolist()}")
    if not np.array_equal(cpu.nearest_indices, tested.nearest_indices):
        mismatch = np.flatnonzero(cpu.nearest_indices != tested.nearest_indices)
        raise RuntimeError(f"CPU/{device} nearest-index mismatch at rows {mismatch.tolist()}")
    for name in (
        "scores",
        "nearest_distances",
        "nearest_positive_distances",
        "nearest_negative_distances",
    ):
        if not np.allclose(
            getattr(cpu, name), getattr(tested, name), rtol=2e-5, atol=2e-5
        ):
            difference = float(
                np.max(np.abs(getattr(cpu, name) - getattr(tested, name)))
            )
            raise RuntimeError(f"CPU/{device} {name} max difference {difference}")
    print(
        f"matcher backend audit OK: device={device}, "
        f"tf32={torch.backends.cuda.matmul.allow_tf32}, "
        f"precision={torch.get_float32_matmul_precision()}, "
        f"deterministic={torch.are_deterministic_algorithms_enabled()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run(parser.parse_args().device)
