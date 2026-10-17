# research/ — ordering a shelf by semantic similarity

Self-contained sandbox next to `mediamap/`. Needs numpy, scipy, matplotlib, umap-learn.

```
python -m research.run --medium lps                    # compare all algorithms, write out/
python -m research.run --medium books --weights genre=0.6,era=0.3,creator=0.1
python -m research.run --medium lps --explain tsp --around "Head Hunters"
python -m research.probes seeds|restarts|noise         # is a result an accident?
python -m unittest research.test_ordering research.test_metrics research.test_distances
```

## How this relates to mediamap (it is not a drop-in replacement)

`mediamap/reduction.py` and `research/ordering.py` do different jobs:

|                 | `reduction.py` (UMAP)                              | `ordering.py` (this folder)                 |
|-----------------|----------------------------------------------------|---------------------------------------------|
| input           | embedding vectors                                  | a complete distance matrix                  |
| output          | coordinates, plus a fitted model that can `transform` a **new** item | an order of the items you gave it |
| new items       | `transform` places one more item without touching the rest | nothing: no out-of-sample step yet |

`Collection` depends on the reducer API: `build_map` fits and saves a model, `_project` calls
`transform` for every new or pending item (`add_record`, `project_pending`). Swapping the file
would break those paths, so `ordering.py` should be added **next to** `reduction.py`, not in
place of it. `distances.py` and `metrics.py` are new and replace nothing.

Integrating the ordering approach means design work that this folder does not do:

1. **Build distance blocks from collection data.** `MediaSpec` has `text_from_row`; it would need
   a sibling hook that returns genre/subject tags, era and creator per row, plus tag embeddings.
2. **Store and show positions, not coordinates.** An order is a rank (or a float slot between
   neighbours), not a point in a space. `x`/`y`, `coord_cols`, `list`, `search` and the
   "Suggested slot" message all assume coordinates.
3. **Define how new and pending items are placed.** This is the missing piece. The likely rule
   is cheapest insertion between two existing neighbours (so the frozen order never moves), with
   a policy for when a full re-ordering is allowed. Not implemented, not evaluated.
4. **Decide what `build-map` and the model file become.** An order lives in the database; there
   is no model to pickle.

A plausible seam is a small "placement strategy" interface (`fit(rows)`, `place(row)`) with UMAP
and ordering as two implementations. That is a proposal, not something built here.

The tests in this folder check the algorithms on synthetic data. None of them exercises
integration with mediamap.

## Caveats

* Tags are written from memory, not fetched from MusicBrainz / Open Library.
* The default tag vectors are a structural stand-in for the sentence model (`--embedder fake`).
  `--embedder minilm` uses the real one but is untested here (no model download in the sandbox).
* "Family breaks" is scored against the same taxonomy that shapes the fake vectors, so it is
  partly circular. Mean step and neighbour recall depend on D alone.