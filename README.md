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
MusicBrainz identifies CD/LP releases; optional album tags come from Last.fm
when `MEDIAMAP_LASTFM_API_KEY=your_key` is configured there or in the shell.
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

To add books or browse LPs from a phone on the same trusted Wi-Fi network, serve
the web interface over HTTPS. Camera access requires a certificate the phone
trusts. Install [mkcert](https://github.com/FiloSottile/mkcert), install its
local CA on both the computer and phone, then create a certificate containing
the computer's `.local` name and LAN IP:

```bash
mkcert -install
mkcert -cert-file mediamap-web-cert.pem -key-file mediamap-web-key.pem \
  "$(scutil --get LocalHostName).local" localhost 127.0.0.1 <computer-LAN-IP>
uv run mediamap books web --cert mediamap-web-cert.pem --key mediamap-web-key.pem
```

Replace `<computer-LAN-IP>` with the computer's Wi-Fi IP address. For iOS,
install the mkcert root certificate as a profile and enable full trust for it
in Certificate Trust Settings; Android also requires trusting the local
CA. Find it with `mkcert -CAROOT` (the `rootCA.pem` file) and consult the
platform's certificate-install instructions. Then open
`https://<computer-LAN-IP>:8765` on the phone. Keep the private key and root CA
private. The server listens on the local network without authentication, so use
it only on a trusted network. The books page scans EAN-13 ISBN barcodes in the
browser, supports manual ISBN/title lookup, and asks you to confirm the edition
before saving. Only the ISBN or search text is sent to the metadata providers;
camera frames stay in the browser. The scanner library is loaded from unpkg.
For LPs, run the same command with `lps web` instead; it supports MusicBrainz
search, adding a selected release, listing the LP catalogue, and rebuilding
the map.

All collections also support `search QUERY` and `project-pending`. Books use
Google Books and Open Library. CD and LP identification uses MusicBrainz;
results are not filtered by format, since physical medium does not define an
album's musical semantics.
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
available MusicBrainz genres, Last.fm album tags, artist, and era; release
format, label, and track count are excluded because physical release details do
not define musical semantics.
Sparse genre/tag metadata can make placement less precise. Run `refresh` after
configuring a Last.fm key to backfill tags for existing CDs and LPs.
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
