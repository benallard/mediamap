"""Does a frozen order survive new items? Compare ways of placing items added later.

Order a random base of the library, then add the held-out items one at a time:

    insert         cheapest insertion into the frozen order (existing items never move)
    insert+polish  insertion, then local search over the whole order (items may shift)
    umap+transform what mediamap does today: frozen UMAP on the base, `transform` the rest
    re-order all   start over with every item (the reference, but it moves everything)

    python -m research.placement --medium lps --held-out 20 --splits 8

"stability" = share of base-item pairs whose relative order is unchanged afterwards
(1.00 = nothing moved; 0.50 = as good as shuffled).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from . import metrics, ordering
from .distances import combine
from .library import load_library
from .run import DEFAULT_WEIGHTS, build_blocks


def stability(base_order, final_order) -> float:
    position = {item: k for k, item in enumerate(final_order)}
    seq = np.array([position[i] for i in base_order])
    i, j = np.triu_indices(len(seq), 1)
    agree = float((seq[i] < seq[j]).mean())
    return max(agree, 1 - agree)  # a reversed shelf is the same shelf


def _umap(D, seed=42):
    import umap

    return umap.UMAP(n_components=1, metric="precomputed", n_neighbors=10, min_dist=0.15,
                     random_state=seed)


def umap_transform_order(D, base, new):
    """Frozen UMAP on the base, `transform` for the rest. Returns (final order, base order)."""
    model = _umap(D).fit(D[np.ix_(base, base)])
    x_new = model.transform(D[np.ix_(new, base)])[:, 0]
    ids = np.concatenate([base, new])
    x = np.concatenate([model.embedding_[:, 0], x_new])
    return ids[np.argsort(x)], base[np.argsort(model.embedding_[:, 0])]


def run(medium: str, held_out: int, splits: int) -> None:
    lib = load_library(medium)
    D = combine(build_blocks(lib, DEFAULT_WEIGHTS))
    n = len(D)
    families = lib.families
    names = ["insert", "insert+polish", "re-order all", "umap+transform", "umap (all at once)"]
    scores = {name: [] for name in names}
    for split in range(splits):
        perm = np.random.default_rng(split).permutation(n)
        new, base = perm[:held_out], np.sort(perm[held_out:])
        base_order = base[ordering.tsp(D[np.ix_(base, base)], restarts=10)]

        frozen = base_order.copy()
        for item in new:
            frozen = ordering.insert(D, frozen, int(item))
        umap_final, umap_base = umap_transform_order(D, base, new)
        # (order, the base order that method started from)
        final = {
            "insert": (frozen, base_order),
            "insert+polish": (ordering.refine(D, frozen), base_order),
            "re-order all": (ordering.tsp(D, restarts=10), base_order),
            "umap+transform": (umap_final, umap_base),
            "umap (all at once)": (ordering.umap1d(D), umap_base),
        }
        for name, (order, started_from) in final.items():
            scores[name].append((metrics.mean_step(D, order), metrics.label_breaks(families, order),
                                 metrics.neighbour_recall(D, order), stability(started_from, order)))

    print(f"\n{medium}: {n - held_out} base items + {held_out} added one by one, {splits} random splits "
          f"(min family breaks {len(set(families)) - 1})")
    print(f"  {'method':<20}{'mean step':>10}{'  +-':>7}{'breaks':>8}{'recall@5':>10}{'stability':>11}")
    for name in names:
        a = np.array(scores[name])
        print(f"  {name:<20}{a[:, 0].mean():>10.3f}{a[:, 0].std():>7.3f}{a[:, 1].mean():>8.1f}"
              f"{a[:, 2].mean():>10.2f}{a[:, 3].mean():>11.2f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--medium", choices=["lps", "books"], default="lps")
    parser.add_argument("--held-out", type=int, default=20)
    parser.add_argument("--splits", type=int, default=8)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", module="umap")
    run(args.medium, args.held_out, args.splits)


if __name__ == "__main__":
    main()
