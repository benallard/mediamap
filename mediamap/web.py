from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlparse

from .collection import Collection, MediaSpec
from .music import fetch_genres, search_releases

_build_lock = threading.Lock()
_build_status: dict[str, Any] = {"running": False, "message": ""}

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f4f2ec">
  <title>LP shelf | MediaMap</title>
  <style>
    :root { color-scheme: light; --paper: #f4f2ec; --ink: #242721; --muted: #77786f; --line: #d8d7ce; --accent: #b63e2d; --green: #596b48; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(100% - 32px, 760px); margin: 0 auto; padding: 28px 0 64px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 20px; }
    h1 { margin: 0; font: 500 36px/1.05 Georgia, "Times New Roman", serif; letter-spacing: 0; }
    .count { color: var(--muted); margin: 7px 0 0; font-size: 14px; }
    button { border: 0; border-radius: 3px; padding: 11px 14px; background: var(--ink); color: white; font: inherit; font-weight: 600; cursor: pointer; }
    button:disabled { cursor: wait; opacity: .55; }
    .rebuild { background: var(--green); white-space: nowrap; }
    section { padding: 24px 0; border-bottom: 1px solid var(--line); }
    h2 { margin: 0 0 14px; font: 500 21px/1.2 Georgia, "Times New Roman", serif; }
    form { display: grid; grid-template-columns: 1fr 1fr auto; align-items: end; gap: 10px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input { width: 100%; min-width: 0; border: 1px solid var(--line); border-radius: 3px; padding: 11px 10px; background: #fffefa; color: var(--ink); font: inherit; }
    .genres { margin-top: 11px; }
    .status { min-height: 20px; margin: 12px 0 0; color: var(--muted); font-size: 13px; }
    .results, .records { display: grid; }
    .result, .record { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--line); }
    .result:last-child, .record:last-child { border-bottom: 0; }
    .name { font-weight: 650; }
    .details { margin-top: 3px; color: var(--muted); font-size: 13px; }
    .pick { flex: 0 0 auto; padding: 8px 11px; border: 1px solid var(--accent); background: transparent; color: var(--accent); }
    .position { flex: 0 0 auto; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
    .empty { padding: 18px 0; color: var(--muted); }
    @media (max-width: 560px) {
      main { width: min(100% - 24px, 760px); padding-top: 20px; }
      h1 { font-size: 31px; }
      header { align-items: flex-start; }
      .rebuild { padding: 9px 10px; font-size: 14px; }
      form { grid-template-columns: 1fr 1fr; }
      form button { grid-column: 1 / -1; }
      .result, .record { align-items: flex-start; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>LP shelf</h1><p class="count" id="count">Loading collection…</p></div>
    <button class="rebuild" id="rebuild">Build map</button>
  </header>
  <section>
    <h2>Add a record</h2>
    <form id="search-form">
      <label>Album title<input id="title" name="title" required autocomplete="off" maxlength="200" placeholder="Album title"></label>
      <label>Artist<input id="artist" name="artist" autocomplete="off" maxlength="200" placeholder="Artist (optional)"></label>
      <button id="search-button">Find releases</button>
    </form>
    <label class="genres">Genres, if you want to add them yourself<input id="genres" autocomplete="off" placeholder="post-punk, new wave"></label>
    <p class="status" id="search-status" aria-live="polite"></p>
    <div class="results" id="results"></div>
  </section>
  <section>
    <h2>Your records</h2>
    <div class="records" id="records"></div>
  </section>
  <p class="status" id="build-status" aria-live="polite"></p>
</main>
<script>
const $ = (id) => document.getElementById(id);
const recordsNode = $('records');
const resultsNode = $('results');
const setStatus = (id, message) => { $(id).textContent = message; };
const details = (record) => [record.year, record.release_format, record.country].filter(Boolean).join(' · ');

async function loadRecords() {
  const response = await fetch('/api/albums');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load records.');
  $('count').textContent = `${data.length} ${data.length === 1 ? 'record' : 'records'}`;
  recordsNode.replaceChildren();
  if (!data.length) {
    const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'Your LP catalogue is empty.';
    recordsNode.append(empty); return;
  }
  for (const record of data) {
    const row = document.createElement('div'); row.className = 'record';
    const info = document.createElement('div');
    const name = document.createElement('div'); name.className = 'name'; name.textContent = record.title;
    const sub = document.createElement('div'); sub.className = 'details';
    sub.textContent = [record.artists, record.year, record.genres.join(', ')].filter(Boolean).join(' · ');
    info.append(name, sub);
    const pos = document.createElement('span'); pos.className = 'position';
    pos.textContent = record.x === null ? 'not mapped' : Number(record.x).toFixed(3);
    row.append(info, pos); recordsNode.append(row);
  }
}

$('search-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('search-button').disabled = true; resultsNode.replaceChildren();
  setStatus('search-status', 'Searching MusicBrainz…');
  try {
    const response = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({title: $('title').value, artist: $('artist').value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Search failed.');
    setStatus('search-status', data.length ? `${data.length} releases found. Choose the pressing you own.` : 'No releases found. Try another title or artist.');
    for (const record of data) {
      const row = document.createElement('div'); row.className = 'result';
      const info = document.createElement('div');
      const name = document.createElement('div'); name.className = 'name'; name.textContent = `${record.title} — ${record.artists.join(', ') || 'Artist unknown'}`;
      const sub = document.createElement('div'); sub.className = 'details'; sub.textContent = [details(record), record.label, record.track_count ? `${record.track_count} tracks` : ''].filter(Boolean).join(' · ');
      info.append(name, sub);
      const pick = document.createElement('button'); pick.className = 'pick'; pick.textContent = 'Add';
      pick.addEventListener('click', () => addRecord(record, pick));
      row.append(info, pick); resultsNode.append(row);
    }
  } catch (error) { setStatus('search-status', error.message); }
  finally { $('search-button').disabled = false; }
});

async function addRecord(record, button) {
  button.disabled = true;
  try {
    const genres = $('genres').value.split(',').map((genre) => genre.trim()).filter(Boolean);
    const response = await fetch('/api/albums', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({record, genres})});
    const data = await response.json();
    if (!response.ok && !data.duplicate) throw new Error(data.error || 'Could not add record.');
    setStatus('search-status', data.duplicate ? 'That release is already in your catalogue.' : 'Record added.');
    await loadRecords();
  } catch (error) { setStatus('search-status', error.message); }
  finally { button.disabled = false; }
}

async function checkBuild() {
  const response = await fetch('/api/status'); const state = await response.json();
  $('rebuild').disabled = state.running;
  $('rebuild').textContent = state.running ? 'Building…' : 'Build map';
  setStatus('build-status', state.message || '');
  return state.running;
}

$('rebuild').addEventListener('click', async () => {
  if (!confirm('Rebuilding refits the map and can move every record. Continue?')) return;
  $('rebuild').disabled = true; setStatus('build-status', 'Starting map build…');
  try {
    const response = await fetch('/api/build-map', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start map build.');
    while (await checkBuild()) await new Promise((resolve) => setTimeout(resolve, 1500));
    await loadRecords();
  } catch (error) { setStatus('build-status', error.message); $('rebuild').disabled = false; }
});

loadRecords().catch((error) => setStatus('search-status', error.message));
checkBuild();
</script>
</body>
</html>"""


@contextmanager
def _open_collection(spec: MediaSpec) -> Iterator[Collection]:
    collection = Collection(spec)
    try:
        yield collection
    finally:
        collection.close()


def _read_genres(raw: str) -> list[str]:
    try:
        genres = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [str(genre) for genre in genres] if isinstance(genres, list) else []


def _build_map(spec: MediaSpec, neighbors: int) -> None:
    try:
        with _open_collection(spec) as collection:
            collection.build_map(neighbors)
        message = "Map build complete."
    except Exception as exc:
        message = f"Map build failed: {exc}"
    with _build_lock:
        _build_status.update(running=False, message=message)


class MediaMapHandler(BaseHTTPRequestHandler):
    spec: MediaSpec

    def _send_json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 100_000:
            raise ValueError("Request is too large.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/albums":
            with _open_collection(self.spec) as collection:
                cols = self.spec.coord_cols
                order = ", ".join(reversed(cols))
                rows = collection.db.execute(
                    f"SELECT * FROM {self.spec.table} "
                    f"ORDER BY {cols[0]} IS NULL, {order}, title"
                ).fetchall()
                albums = [
                    {"id": row["id"], "title": row["title"], "artists": row[self.spec.creators_col],
                     "year": row["year"], "release_format": row["release_format"], "label": row["label"],
                     "country": row["country"], "genres": _read_genres(row["genres"]),
                     "x": row[cols[0]]}
                    for row in rows
                ]
            self._send_json(200, albums)
            return
        if path == "/api/status":
            with _build_lock:
                self._send_json(200, dict(_build_status))
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc) or "Invalid JSON."})
            return

        if path == "/api/search":
            title = payload.get("title", "")
            artist = payload.get("artist", "")
            if not isinstance(title, str) or not title.strip() or not isinstance(artist, str):
                self._send_json(400, {"error": "Enter an album title and a valid artist."})
                return
            self._send_json(200, search_releases(title.strip()[:200], artist.strip()[:200]))
            return

        if path == "/api/albums":
            record = payload.get("record")
            if not isinstance(record, dict) or not isinstance(record.get("artists"), list):
                self._send_json(400, {"error": "Choose a MusicBrainz release first."})
                return
            if any(not isinstance(record.get(key), str) or not record[key] for key in ("source_id", "title")):
                self._send_json(400, {"error": "The selected release is missing its title or ID."})
                return
            if not all(isinstance(name, str) for name in record["artists"]):
                self._send_json(400, {"error": "Invalid artist data."})
                return
            genres = payload.get("genres", [])
            if not isinstance(genres, list) or not all(isinstance(genre, str) for genre in genres):
                self._send_json(400, {"error": "Genres must be a list of names."})
                return
            record = {key: record.get(key, "") for key in (
                "source_id", "title", "artists", "year", "label", "release_format",
                "country", "barcode", "release_group_id", "artist_id",
            )}
            record["genres"] = [genre.strip().lower() for genre in genres if genre.strip()]
            if not record["genres"]:
                record["genres"] = fetch_genres(record["release_group_id"], record["artist_id"])
            with _open_collection(self.spec) as collection:
                item_id = collection.add_record(record, enrich=False)
            if item_id is None:
                self._send_json(409, {"error": "That release is already in the LP catalogue.", "duplicate": True})
            else:
                self._send_json(201, {"id": item_id})
            return

        if path == "/api/build-map":
            neighbors = payload.get("neighbors", 8)
            if type(neighbors) is not int or not 2 <= neighbors <= 50:
                self._send_json(400, {"error": "Neighbors must be between 2 and 50."})
                return
            with _build_lock:
                if _build_status["running"]:
                    self._send_json(409, {"error": "A map build is already running."})
                    return
                with _open_collection(self.spec) as collection:
                    count = collection.db.execute(
                        f"SELECT COUNT(*) FROM {self.spec.table}"
                    ).fetchone()[0]
                if count < self.spec.min_items:
                    self._send_json(400, {"error": f"Add at least {self.spec.min_items} LPs before building the map ({count} now)."})
                    return
                _build_status.update(running=True, message="Building the LP map…")
                threading.Thread(target=_build_map, args=(self.spec, neighbors), daemon=True).start()
            self._send_json(202, {"message": "Map build started."})
            return

        self._send_json(404, {"error": "Not found."})


def serve(spec: MediaSpec, host: str, port: int) -> None:
    handler = type("LPWebHandler", (MediaMapHandler,), {"spec": spec})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"LP web interface listening on http://{host}:{server.server_port}")
    print("On your phone, open http://<this-computer's-local-IP>:" f"{server.server_port} on a trusted Wi-Fi network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LP web interface.")
    finally:
        server.server_close()
