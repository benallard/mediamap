# research/ — ordering a shelf by semantic similarity

Self-contained sandbox next to `mediamap/`. Needs numpy, scipy, matplotlib, umap-learn.

```
python -m research.run --medium lps                    # compare all algorithms, write out/
python -m research.run --medium books --weights genre=0.6,era=0.3,creator=0.1
python -m research.run --medium lps --explain tsp --around "Head Hunters"
python -m research.probes seeds|restarts|noise         # is a result an accident?
python -m unittest research.test_ordering research.test_metrics research.test_distances
```

## What moves back into mediamap, and what stays

| file            | role                                                              | destination            |
|-----------------|-------------------------------------------------------------------|------------------------|
| `ordering.py`   | `D -> permutation`: random, pcoa1d, spectral, umap1d, leaf, tsp   | `mediamap/ordering.py` (replaces `reduction.py`) |
| `distances.py`  | distance blocks, `combine`, `explain`                             | `mediamap/distances.py`|
| `metrics.py`    | mean step, neighbour recall, label breaks, worst joins            | `mediamap/metrics.py`  |
| `library.py`    | fake items, taxonomies, stand-in tag embedder                     | stays here             |
| `run.py`, `probes.py` | experiments                                                 | stay here              |

The three moving modules only see a distance matrix; none of them knows about books or music.
In the package, a spec would supply the blocks (genre, era, creator, ...) for its rows, and the
collection would store the resulting *position* per item instead of UMAP coordinates.

## Caveats

* Tags are written from memory, not fetched from MusicBrainz / Open Library.
* The default tag vectors are a structural stand-in for the sentence model (`--embedder fake`).
  `--embedder minilm` uses the real one but is untested here (no model download in the sandbox).
* "Family breaks" is scored against the same taxonomy that shapes the fake vectors, so it is
  partly circular. Mean step and neighbour recall depend on D alone.
