"""Run every ordering algorithm on a fake library and compare them.

    python -m research.run --medium lps
    python -m research.run --medium books --weights genre=0.6,era=0.3,creator=0.1
    python -m research.run --medium lps --explain tsp --show tsp

Writes, per medium, into --out (default research/out):
    <medium>_overview.png        one panel per algorithm: family strip + reordered distance matrix
    <medium>_<algorithm>.txt     the shelf, item by item
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from . import metrics, ordering
from .distances import Block, combine, creator_block, cosine_block, era_block, explain, weighted_tag_vectors
from .library import Library, load_library, minilm_tag_vectors, structural_tag_vectors

DEFAULT_WEIGHTS = {"genre": 0.7, "era": 0.2, "creator": 0.1}


def build_blocks(lib: Library, weights: dict[str, float], embedder: str = "fake",
                 tag_vectors: dict | None = None) -> list[Block]:
    if tag_vectors is None:
        tag_vectors = (minilm_tag_vectors if embedder == "minilm" else structural_tag_vectors)(lib)
    parts = {
        "genre": cosine_block(weighted_tag_vectors(lib.tags, tag_vectors)),
        "era": era_block(lib.years, lib.era_sigma),
        "creator": creator_block(lib.creators),
    }
    return [Block(name, parts[name], w) for name, w in weights.items() if w > 0]


def algorithms(lib: Library, restarts: int) -> dict:
    """name -> function(D) returning an order. The first two need item metadata, not just D."""
    alphabetical = np.array(sorted(range(len(lib)), key=lambda i: (lib.creators[i].split()[-1].lower(),
                                                                   lib.titles[i].lower())))
    by_year = np.array(sorted(range(len(lib)), key=lambda i: (lib.years[i], lib.titles[i])))
    return {
        "random": ordering.random_order,
        "alphabetical": lambda D: alphabetical,   # by creator's last word, then title
        "by_year": lambda D: by_year,
        "pcoa1d": ordering.pcoa1d,
        "spectral": ordering.spectral,
        "umap1d": ordering.umap1d,
        "leaf": ordering.leaf,
        "tsp": lambda D: ordering.tsp(D, restarts=restarts),
        "leaf+opt": lambda D: ordering.refine(D, ordering.leaf(D)),
    }


def write_listing(path: Path, lib: Library, order) -> None:
    lines = [f"{pos + 1:3}  {lib.families[i]:<17} {lib.label(i)}  [{', '.join(lib.tags[i])}]"
             for pos, i in enumerate(order)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, lib: Library, D, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    names = sorted(set(lib.families))
    cmap = ListedColormap(plt.get_cmap("tab20")(np.arange(len(names)) % 20))
    code = {name: k for k, name in enumerate(names)}
    labels = np.array([code[f] for f in lib.families])

    cols = 3
    rows = -(-len(results) // cols)
    fig = plt.figure(figsize=(5.2 * cols, 5.0 * rows))
    grid = fig.add_gridspec(rows * 2, cols, height_ratios=[1, 14] * rows, hspace=0.25, wspace=0.12)
    for k, (name, (order, scores)) in enumerate(results.items()):
        r, c = divmod(k, cols)
        strip = fig.add_subplot(grid[2 * r, c])
        strip.imshow(labels[order][None, :], cmap=cmap, vmin=0, vmax=len(names) - 1, aspect="auto",
                     interpolation="nearest")
        strip.set_axis_off()
        strip.set_title(f"{name}   step {scores['step']:.3f}   breaks {scores['breaks']}   "
                        f"recall {scores['recall']:.2f}", fontsize=9)
        heat = fig.add_subplot(grid[2 * r + 1, c])
        heat.imshow(D[np.ix_(order, order)], cmap="viridis_r", vmin=0, vmax=np.percentile(D, 95),
                    interpolation="nearest")
        heat.set_xticks([])
        heat.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(code[n])) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 7), fontsize=8, frameon=False)
    fig.suptitle(f"{lib.medium}: family strip (top of each panel) and distance matrix in shelf order "
                 f"(dark = close)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def parse_weights(text: str) -> dict[str, float]:
    return {k: float(v) for k, v in (part.split("=") for part in text.split(","))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--medium", choices=["lps", "books"], default="lps")
    parser.add_argument("--weights", type=parse_weights, default=DEFAULT_WEIGHTS)
    parser.add_argument("--embedder", choices=["fake", "minilm"], default="fake",
                        help="tag vectors: structural stand-in, or the real sentence model")
    parser.add_argument("--restarts", type=int, default=30, help="TSP random restarts")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--show", metavar="ALGO", help="print this algorithm's shelf")
    parser.add_argument("--around", metavar="TITLE", action="append",
                        help="show the neighbours of this title under every algorithm (repeatable)")
    parser.add_argument("--explain", metavar="ALGO", help="break down this algorithm's worst joins")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", module="umap")

    lib = load_library(args.medium)
    blocks = build_blocks(lib, args.weights, args.embedder)
    D = combine(blocks)
    n_families = len(set(lib.families))
    print(f"{args.medium}: {len(lib)} items, {n_families} families, "
          f"weights {', '.join(f'{b.name}={b.weight:g}' for b in blocks)}, tags: {args.embedder}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'algorithm':<13}{'mean step':>10}{'recall@5':>10}{'breaks':>8}{'path':>8}{'secs':>7}")
    for name, fn in algorithms(lib, args.restarts).items():
        started = time.perf_counter()
        order = np.asarray(fn(D))
        elapsed = time.perf_counter() - started
        assert sorted(order.tolist()) == list(range(len(lib))), f"{name} is not a permutation"
        scores = {"step": metrics.mean_step(D, order), "recall": metrics.neighbour_recall(D, order),
                  "breaks": metrics.label_breaks(lib.families, order)}
        results[name] = (order, scores)
        write_listing(args.out / f"{args.medium}_{name.replace('+', '_')}.txt", lib, order)
        print(f"{name:<13}{scores['step']:>10.3f}{scores['recall']:>10.2f}{scores['breaks']:>8}"
              f"{ordering.path_length(D, order):>8.2f}{elapsed:>7.2f}")
    print(f"\n'breaks' = neighbours from different families; the minimum possible is {n_families - 1}.")

    if not args.no_plot:
        plot(args.out / f"{args.medium}_overview.png", lib, D, results)
        print(f"wrote {args.out / (args.medium + '_overview.png')}")

    if args.show:
        print(f"\n--- {args.show} ---")
        print((args.out / f"{args.medium}_{args.show.replace('+', '_')}.txt").read_text(encoding="utf-8"))

    for title in args.around or []:
        i = lib.titles.index(title)
        print(f"\n--- around '{title}' [{', '.join(lib.tags[i])}] ---")
        for name in ("alphabetical", "umap1d", "leaf", "tsp"):
            order = list(results[name][0])
            p = order.index(i)
            near = [lib.titles[order[q]] if q != p else f"[{title}]"
                    for q in range(max(0, p - 2), min(len(order), p + 3))]
            print(f"  {name:<13}" + "  |  ".join(near))

    if args.explain:
        order = results[args.explain][0]
        print(f"\n--- worst joins in {args.explain} (distance split by block) ---")
        for a, b, d in metrics.worst_joins(D, order, top=6):
            shares = ", ".join(f"{k} {v:.2f}" for k, v in explain(blocks, a, b).items())
            print(f"{d:.2f}  {lib.titles[a]}  ->  {lib.titles[b]}\n      {shares}")


if __name__ == "__main__":
    main()
