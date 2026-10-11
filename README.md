# MediaMap

MediaMap keeps separate local collections for books, CDs, and LPs and arranges
each collection by semantic similarity. Books are mapped in two dimensions;
CDs and LPs each have their own one-dimensional map. Nothing is mixed across
these collections.

## Setup

Python 3.12+ is required. Install [uv](https://docs.astral.sh/uv/) if needed,
then sync dependencies and run the CLI from the project directory:

```bash
uv sync
uv run mediamap --help
```

Metadata lookups need an internet connection. On the first map operation,
MediaMap downloads its sentence-transformer model from Hugging Face and saves
it under `models/`; future runs use that local copy. To enable a Google Books
API key, add `GOOGLE_BOOKS_API_KEY=your_key` to a project-root `.env` file.
You can also export the key in your shell. Keep `.env` private and do not
commit it.

## Separate collections

Each collection has the same commands, but its own database and fitted map:

```bash
uv run mediamap books add
uv run mediamap cds add
uv run mediamap lps add
```

For each collection, repeat `add` to build a varied set, then create its map
and browse it locally:

```bash
uv run mediamap books build-map
uv run mediamap books list
uv run mediamap books search "Kant"

uv run mediamap cds build-map
uv run mediamap cds list
uv run mediamap lps build-map
uv run mediamap lps list
```

All collections also support `search QUERY` and `project-pending`. Books use
Google Books and Open Library. CD and LP lookups use MusicBrainz; results are
not filtered by format, since physical medium does not define an album's
musical semantics.
When a CD or LP is projected into an existing map, `add` reports its suggested
slot between adjacent releases (or before/after an endpoint release).
Set `BOOKMAP_CONTACT_EMAIL` in `.env` or the shell to identify requests to
Open Library and MusicBrainz.

## Storage and maps

- `library.sqlite3` and `map_model.joblib` are the existing book catalogue and
  2D map. Existing files are left in place.
- `cds.sqlite3` and `cds_map_model.joblib` store the CD collection and its
  independent 1D map.
- `lps.sqlite3` and `lps_map_model.joblib` store the LP collection and its
  independent 1D map.
- The downloaded embedding model is stored in the ignored `models/` folder.
  Back up these files together if you need to preserve the collections and
  avoid downloading the model again on another machine.

Fitting a map again moves its existing coordinates, so rebuild only when you
intend to refit that collection. For each 1D map, the coordinate is a relative
similarity position, not a physical shelf location. Books use UMAP's 2D
coordinates. Sparse metadata makes placement less meaningful.

## Limitations

MusicBrainz may return multiple releases or incomplete format information;
review the selected release, year, label, and format. The CD and LP maps embed
available genres, artist, and era; release format, label, and track count are
excluded because physical release details do not define musical semantics.
Sparse genre metadata can make placement less precise.
At least three albums are required to fit a 1D music map.
Book lookup may also return different editions of the same work; cross-source
metadata merging does not guarantee edition matching.

## Tests

Run the offline test suite with:

```bash
uv run python -m unittest discover -s tests
```

GitHub Actions runs the unit tests and checks all four CLI help entry points on
pushes and pull requests.
