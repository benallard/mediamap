# BookMap — semantic book shelving prototype

A command-line prototype for manually cataloguing books, enriching them through Google Books (Open Library fallback), embedding descriptions/categories with a multilingual model, and placing books in a 2-D UMAP map.

## Requirements

Python 3.10+ recommended. The first run downloads a multilingual sentence-transformer model and may take a while. Metadata lookup requires an internet connection. The program makes no OCR or image-based guesses.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python library.py add
```

Repeat `python library.py add` for an initial representative batch. For a first trial, use at least 20–50 varied books if possible; UMAP with only a handful is unstable and not very informative. Then fit the initial map:

```bash
python library.py build-map
python library.py list
python library.py search "Kant"
```

For books added later, use `python library.py add`; the saved UMAP model projects them into the existing map. If a book was saved but projection failed, use `python library.py project-pending`.

## Stability and limitations

- `library.sqlite3` stores catalogue metadata and assigned coordinates. `map_model.joblib` stores the fitted UMAP transform. Back up both files together.
- **Do not run `build-map` again casually.** It refits UMAP and may move all existing books. The model is intentionally fixed between map rebuilds.
- The embedding model is `paraphrase-multilingual-MiniLM-L12-v2`, suitable for German/English similarity experiments. Model upgrades are out of scope for this prototype.
- UMAP coordinates are relative, arbitrary axes, not literal shelf numbers. Nearby coordinates are a starting point for grouping books into shelf sections.
- Cross-language equivalence and UMAP quality are not guaranteed. Sparse or missing summaries lead to weaker placement. Inspect the printed nearest neighbours and test on books you know well.
- The metadata providers may return multiple editions. Choose the correct record from the list. This prototype detects repeat entries by the same provider record ID; it does **not** yet robustly merge different provider records or editions of the same work. Content similarity helps placement, but is not proof of duplicate identity.
- Google Books is tried first; if it is rate-limited (HTTP 429), fails, or returns no matches, the program tries Open Library. If Open Library’s structured title/author search returns no matches, it tries one broader combined query. Open Library descriptions are not available for every record. The prototype currently does not combine results from both providers for every book. Open Library requests identify the app with a User-Agent; you can optionally set `BOOKMAP_CONTACT_EMAIL` to include a contact address as requested by Open Library.
- This is a local prototype, not a polished mobile app. It has not been tested against a real collection yet.

## Suggested next improvements

1. Query both metadata providers and combine the best description/subjects.
2. Add explicit ISBN/edition/work matching across providers.
3. Export a CSV/HTML shelf map and allow shelf dimensions and capacity constraints.
4. Add a review screen showing nearest semantic neighbours before physically shelving.
