"""Is a result an accident? Three quick robustness probes.

    python -m research.probes seeds      # run-to-run spread of the stochastic algorithms
    python -m research.probes restarts   # has the TSP heuristic settled?
    python -m research.probes noise      # does the ranking survive a messier tag space?
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from . import metrics, ordering
from .distances import combine
from .library import load_library, structural_tag_vectors
from .run import DEFAULT_WEIGHTS, build_blocks


def score(lib, D, order):
    return (metrics.mean_step(D, order), metrics.neighbour_recall(D, order),
            metrics.label_breaks(lib.families, order))


def setup(medium, w_tag=0.8):
    lib = load_library(medium)
    blocks = build_blocks(lib, DEFAULT_WEIGHTS, tag_vectors=structural_tag_vectors(lib, w_tag=w_tag))
    return lib, combine(blocks)


def seeds():
    print("five seeds each: mean step min..max | breaks min..max")
    for medium in ("lps", "books"):
        lib, D = setup(medium)
        for name, fn in [("umap1d", lambda s: ordering.umap1d(D, seed=s)),
                         ("tsp", lambda s: ordering.tsp(D, seed=s))]:
            rows = np.array([score(lib, D, fn(s)) for s in range(5)])
            print(f"  {medium:<6}{name:<8} step {rows[:, 0].min():.3f}..{rows[:, 0].max():.3f}"
                  f" | breaks {int(rows[:, 2].min())}..{int(rows[:, 2].max())}")


def restarts():
    print("TSP path length by number of random restarts (lower is better)")
    for medium in ("lps", "books"):
        lib, D = setup(medium)
        cells = [f"{r:>3}: {ordering.path_length(D, ordering.tsp(D, restarts=r)):.2f}" for r in (1, 5, 30, 100)]
        print(f"  {medium:<6}" + "   ".join(cells))


def noise():
    print("tag-space noise (w_tag): bigger = tags share less family structure")
    print(f"  {'lps':<6}{'w_tag':>6}   " + "   ".join(f"{a:<22}" for a in ("umap1d", "leaf", "tsp")))
    for medium in ("lps", "books"):
        for w_tag in (0.8, 1.6, 3.0):
            lib, D = setup(medium, w_tag)
            cells = []
            for fn in (ordering.umap1d, ordering.leaf, ordering.tsp):
                step, recall, breaks = score(lib, D, fn(D))
                cells.append(f"{step:.3f} r{recall:.2f} b{breaks:<3}")
            print(f"  {medium:<6}{w_tag:>6}   " + "   ".join(f"{c:<22}" for c in cells))
    print("  (step = mean step, r = recall@5, b = family breaks; ground-truth minimum is 11 for lps, 13 for books)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="umap")
    {"seeds": seeds, "restarts": restarts, "noise": noise}[sys.argv[1] if len(sys.argv) > 1 else "seeds"]()
