from __future__ import annotations

import json
import ssl
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlparse

from .books import lookup_metadata, normalize_isbn
from .collection import Collection, MediaSpec
from .music import MusicBrainzError, fetch_genres, fetch_lastfm_tags, search_releases

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
    .record.just-added { animation: added-flash 1.8s ease-out both; }
    @keyframes added-flash { from { background: #e1e7d8; } to { background: transparent; } }
    .busy-overlay { position: fixed; inset: 0; z-index: 5; display: grid; place-items: center; background: rgb(36 39 33 / 34%); }
    .busy-overlay[hidden] { display: none; }
    .busy-message { display: flex; align-items: center; gap: 12px; padding: 16px 20px; border-radius: 3px; background: var(--paper); box-shadow: 0 8px 30px rgb(0 0 0 / 15%); font-weight: 600; }
    .spinner { width: 18px; height: 18px; border: 2px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
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
<div class="busy-overlay" id="busy-overlay" hidden>
    <div class="busy-message" role="status" aria-live="polite"><span class="spinner" aria-hidden="true"></span><span id="busy-message">Adding record…</span></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const recordsNode = $('records');
const resultsNode = $('results');
const setStatus = (id, message) => { $(id).textContent = message; };
const details = (record) => [record.year, record.release_format, record.country].filter(Boolean).join(' · ');

function setBusy(busy, message = 'Adding record…') {
    $('busy-overlay').hidden = !busy;
    $('busy-message').textContent = message;
    document.querySelector('main').inert = busy;
    document.querySelector('main').setAttribute('aria-busy', String(busy));
}

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
        row.dataset.recordId = record.id;
        row.tabIndex = -1;
    const info = document.createElement('div');
    const name = document.createElement('div'); name.className = 'name'; name.textContent = record.title;
    const sub = document.createElement('div'); sub.className = 'details';
    sub.textContent = [record.artists, record.year, record.genres.join(', '),
      record.tags.length ? `Tags: ${record.tags.join(', ')}` : ''].filter(Boolean).join(' · ');
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
    setBusy(true);
  try {
    const genres = $('genres').value.split(',').map((genre) => genre.trim()).filter(Boolean);
    const response = await fetch('/api/albums', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({record, genres})});
    const data = await response.json();
    if (!response.ok && !data.duplicate) throw new Error(data.error || 'Could not add record.');
    await loadRecords();
        if (data.duplicate) {
            setStatus('search-status', 'That release is already in your catalogue.');
            return;
        }

        $('title').value = '';
        $('artist').value = '';
        $('genres').value = '';
        resultsNode.replaceChildren();

        const addedRow = document.querySelector(`[data-record-id="${data.id}"]`);
        if (addedRow) {
            addedRow.classList.add('just-added');
            const position = addedRow.querySelector('.position').textContent;
            setStatus('search-status', position === 'not mapped'
                ? `Added “${record.title}”; it is not mapped yet.`
                : `Added “${record.title}” at map position ${position}.`);
            requestAnimationFrame(() => {
                addedRow.focus({preventScroll: true});
                addedRow.scrollIntoView({behavior: 'smooth', block: 'center'});
            });
        } else {
            setStatus('search-status', `Added “${record.title}”.`);
        }
    } catch (error) { setStatus('search-status', error.message); }
    finally { button.disabled = false; setBusy(false); }
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


BOOK_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f4f2ec">
  <title>Book shelf | MediaMap</title>
  <script src="https://unpkg.com/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>
  <style>
    :root { color-scheme: light; --paper: #f4f2ec; --ink: #242721; --muted: #77786f; --line: #d8d7ce; --accent: #b63e2d; --green: #596b48; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(100% - 32px, 760px); margin: 0 auto; padding: 28px 0 64px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 20px; }
    h1 { margin: 0; font: 500 36px/1.05 Georgia, "Times New Roman", serif; }
    h2 { margin: 0 0 14px; font: 500 21px/1.2 Georgia, "Times New Roman", serif; }
    .count, .hint, .status { color: var(--muted); font-size: 13px; }
    .count { margin: 7px 0 0; }
    button { border: 0; border-radius: 3px; padding: 11px 14px; background: var(--ink); color: white; font: inherit; font-weight: 600; cursor: pointer; }
    button:disabled { cursor: wait; opacity: .55; }
    .rebuild { background: var(--green); white-space: nowrap; }
    section { padding: 24px 0; border-bottom: 1px solid var(--line); }
    form { display: grid; grid-template-columns: 1fr 1fr auto; align-items: end; gap: 10px; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    input { width: 100%; min-width: 0; border: 1px solid var(--line); border-radius: 3px; padding: 11px 10px; background: #fffefa; color: var(--ink); font: inherit; }
    .isbn-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .secondary { border: 1px solid var(--accent); background: transparent; color: var(--accent); }
    #scanner { max-width: 480px; margin-top: 12px; }
    #stop-scan { margin-top: 8px; }
    .status { min-height: 20px; margin: 12px 0 0; }
    .results, .records { display: grid; }
    .result, .record { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--line); }
    .name { font-weight: 650; }
    .details { margin-top: 3px; color: var(--muted); font-size: 13px; }
    .pick { flex: 0 0 auto; padding: 8px 11px; border: 1px solid var(--accent); background: transparent; color: var(--accent); }
    .position { flex: 0 0 auto; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
    .record.just-added { animation: added-flash 1.8s ease-out both; }
    @keyframes added-flash { from { background: #e1e7d8; } to { background: transparent; } }
    .empty { padding: 18px 0; color: var(--muted); }
    @media (max-width: 560px) { main { width: min(100% - 24px, 760px); padding-top: 20px; } h1 { font-size: 31px; } header { align-items: flex-start; } form { grid-template-columns: 1fr; } .result, .record { align-items: flex-start; } }
  </style>
</head>
<body>
<main>
  <header><div><h1>Book shelf</h1><p class="count" id="count">Loading collection…</p></div><button class="rebuild" id="rebuild">Build map</button></header>
  <section>
    <h2>Add a book</h2>
    <form id="search-form">
      <label>ISBN-10 or ISBN-13<div class="isbn-row"><input id="isbn" autocomplete="off" inputmode="numeric" placeholder="Scan or enter ISBN"><button class="secondary" type="button" id="scan">Scan barcode</button></div></label>
      <label>Book title<input id="title" autocomplete="off" maxlength="200" placeholder="Title (optional if ISBN entered)"></label>
      <label>Author<input id="author" autocomplete="off" maxlength="200" placeholder="Author (optional)"></label>
      <button id="search-button">Find editions</button>
    </form>
    <p class="hint">Scan the ISBN barcode on the back cover, or enter it manually. Check the edition before adding.</p>
    <div id="scanner" hidden></div>
    <button class="secondary" type="button" id="stop-scan" hidden>Stop camera</button>
    <p class="status" id="search-status" aria-live="polite"></p>
    <div class="results" id="results"></div>
  </section>
  <section><h2>Your books</h2><div class="records" id="records"></div></section>
  <p class="status" id="build-status" aria-live="polite"></p>
</main>
<script>
const $ = (id) => document.getElementById(id);
const setStatus = (message) => { $('search-status').textContent = message; };
const results = $('results');
let scanner;
let scanHandled = false;

async function loadBooks() {
  const response = await fetch('/api/albums');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load books.');
  $('count').textContent = `${data.length} ${data.length === 1 ? 'book' : 'books'}`;
  const list = $('records'); list.replaceChildren();
  if (!data.length) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'Your book catalogue is empty.'; list.append(empty); return; }
  for (const book of data) {
    const row = document.createElement('div'); row.className = 'record'; row.dataset.recordId = book.id;
    const info = document.createElement('div'); const name = document.createElement('div'); name.className = 'name'; name.textContent = book.title;
    const sub = document.createElement('div'); sub.className = 'details'; sub.textContent = [book.authors, book.year, book.isbn].filter(Boolean).join(' · ');
    info.append(name, sub); const position = document.createElement('span'); position.className = 'position';
    position.textContent = book.x === null ? 'not mapped' : `${Number(book.x).toFixed(3)}, ${Number(book.y).toFixed(3)}`;
    row.append(info, position); list.append(row);
  }
}

function addCandidate(book) {
  const row = document.createElement('div'); row.className = 'result'; const info = document.createElement('div');
  const name = document.createElement('div'); name.className = 'name'; name.textContent = book.title;
  const details = document.createElement('div'); details.className = 'details';
  details.textContent = [book.authors.join(', '), book.year, book.isbn, book.publisher, book.display_source].filter(Boolean).join(' · ');
  info.append(name, details);
  if (book.description) { const description = document.createElement('div'); description.className = 'details'; description.textContent = book.description.slice(0, 240); info.append(description); }
  const add = document.createElement('button'); add.className = 'pick'; add.textContent = 'Add this edition';
  add.addEventListener('click', () => saveBook(book, add)); row.append(info, add); results.append(row);
}

$('search-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('search-button').disabled = true; results.replaceChildren();
  const isbn = $('isbn').value.trim(), title = $('title').value.trim(), author = $('author').value.trim();
  if (!isbn && !title) { setStatus('Enter an ISBN or book title.'); $('search-button').disabled = false; return; }
  setStatus(isbn ? 'Looking up ISBN…' : 'Searching book catalogues…');
  try {
    const response = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({isbn, title, author})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Search failed.');
    data.forEach(addCandidate); setStatus(data.length ? `${data.length} match${data.length === 1 ? '' : 'es'} found. Confirm the edition, then add it.` : 'No matching book found. Check the ISBN or try a title search.');
  } catch (error) { setStatus(error.message); }
  finally { $('search-button').disabled = false; }
});

async function saveBook(book, button) {
  button.disabled = true; setStatus('Adding book…');
  try {
    const response = await fetch('/api/albums', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({record: book})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Could not add book.');
    await loadBooks(); results.replaceChildren(); $('isbn').value = ''; $('title').value = ''; $('author').value = '';
    const row = document.querySelector(`[data-record-id="${data.id}"]`); if (row) row.classList.add('just-added');
    setStatus(`Added “${book.title}”.`);
  } catch (error) { setStatus(error.message); }
  finally { button.disabled = false; }
}

$('scan').addEventListener('click', async () => {
  if (!window.Html5Qrcode) { setStatus('Barcode scanner did not load. Enter the ISBN manually or reload the page.'); return; }
  $('scan').disabled = true; $('scanner').hidden = false; scanHandled = false;
  $('scanner').scrollIntoView({behavior: 'smooth', block: 'center'});
  try {
    scanner = new Html5Qrcode('scanner', {formatsToSupport: [Html5QrcodeSupportedFormats.EAN_13]});
    await scanner.start({facingMode: 'environment'}, {fps: 10, qrbox: {width: 280, height: 130}}, async (code) => {
      if (scanHandled) return;
      if (!/^97[89][0-9]{10}$/.test(code)) { setStatus('That barcode is not an ISBN-13. Keep scanning or enter the ISBN manually.'); return; }
      scanHandled = true; $('isbn').value = code; await scanner.stop(); $('scanner').hidden = true;
      $('stop-scan').hidden = true; $('scan').disabled = false;
      $('search-form').requestSubmit();
    }, () => {});
    $('stop-scan').hidden = false;
    setStatus('Point the camera at the ISBN barcode on the book.');
  } catch (error) { $('scanner').hidden = true; $('stop-scan').hidden = true; $('scan').disabled = false; setStatus(`Could not start camera: ${error.message || error}`); }
});
$('stop-scan').addEventListener('click', async () => {
  try { if (scanner) await scanner.stop(); $('scanner').hidden = true; $('stop-scan').hidden = true; $('scan').disabled = false; setStatus('Scanning stopped.'); }
  catch (error) { setStatus(`Could not stop camera: ${error.message || error}`); }
});

async function checkBuild() {
  const response = await fetch('/api/status'); const state = await response.json();
  $('rebuild').disabled = state.running; $('rebuild').textContent = state.running ? 'Building…' : 'Build map';
  $('build-status').textContent = state.message || ''; return state.running;
}
$('rebuild').addEventListener('click', async () => {
  if (!confirm('Rebuilding refits the map and can move every book. Continue?')) return;
  $('rebuild').disabled = true; $('build-status').textContent = 'Starting map build…';
  try {
    const response = await fetch('/api/build-map', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Could not start map build.');
    while (await checkBuild()) await new Promise((resolve) => setTimeout(resolve, 1500)); await loadBooks();
  } catch (error) { $('build-status').textContent = error.message; $('rebuild').disabled = false; }
});
loadBooks().catch((error) => setStatus(error.message)); checkBuild();
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
            page = BOOK_PAGE if self.spec.table == "books" else PAGE
            body = page.encode("utf-8")
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
                if self.spec.table == "books":
                    items = [
                        {"id": row["id"], "title": row["title"], "authors": row["authors"],
                         "year": row["year"], "isbn": row["isbn"], "x": row[cols[0]], "y": row[cols[1]]}
                        for row in rows
                    ]
                else:
                    items = [
                        {"id": row["id"], "title": row["title"], "artists": row[self.spec.creators_col],
                         "year": row["year"], "release_format": row["release_format"], "label": row["label"],
                         "country": row["country"], "genres": _read_genres(row["genres"]),
                         "tags": _read_genres(row["lastfm_tags"]), "x": row[cols[0]]}
                        for row in rows
                    ]
            self._send_json(200, items)
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
            if self.spec.table == "books":
                isbn = payload.get("isbn", "")
                title = payload.get("title", "")
                author = payload.get("author", "")
                if not all(isinstance(value, str) for value in (isbn, title, author)):
                    self._send_json(400, {"error": "ISBN, title, and author must be text."})
                    return
                try:
                    normalized_isbn = normalize_isbn(isbn) if isbn.strip() else ""
                    if not normalized_isbn and not title.strip():
                        raise ValueError("Enter an ISBN or book title.")
                    metadata = lookup_metadata(
                        isbn=normalized_isbn or None,
                        title=title.strip()[:200] or None,
                        author=author.strip()[:200] or None,
                    )
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                sources = metadata.get("sources") or []
                source_ids = metadata.get("source_ids") or {}
                if not sources:
                    self._send_json(200, [])
                    return
                source = next((name for name in sources if source_ids.get(name)), sources[0])
                candidate = {
                    "source": source,
                    "source_id": source_ids.get(source, ""),
                    "display_source": " + ".join(sources),
                    "title": metadata.get("title") or title.strip(),
                    "authors": metadata.get("authors") or ([author.strip()] if author.strip() else []),
                    "year": str(metadata.get("published_date") or "")[:4],
                    "isbn": metadata.get("isbn_13") or metadata.get("isbn_10") or normalized_isbn,
                    "publisher": metadata.get("publisher") or "",
                    "description": metadata.get("description") or "",
                    "subjects": metadata.get("subjects") or [],
                    "language": metadata.get("language") or "",
                }
                self._send_json(200, [candidate])
                return
            title = payload.get("title", "")
            artist = payload.get("artist", "")
            if not isinstance(title, str) or not title.strip() or not isinstance(artist, str):
                self._send_json(400, {"error": "Enter an album title and a valid artist."})
                return
            try:
                releases = search_releases(title.strip()[:200], artist.strip()[:200])
            except MusicBrainzError as exc:
                self._send_json(502, {"error": str(exc)})
                return
            self._send_json(200, releases)
            return

        if path == "/api/albums":
            if self.spec.table == "books":
                record = payload.get("record")
                required = ("source", "source_id", "title")
                if not isinstance(record, dict) or any(
                    not isinstance(record.get(key), str) or not record[key].strip()
                    for key in required
                ):
                    self._send_json(400, {"error": "Choose a book edition from the search results first."})
                    return
                if not isinstance(record.get("authors", []), list) or not all(
                    isinstance(author, str) for author in record.get("authors", [])
                ):
                    self._send_json(400, {"error": "Invalid author data."})
                    return
                record = {
                    key: record.get(key, "") for key in (
                        "source", "source_id", "title", "authors", "year", "isbn",
                        "description", "subjects", "language",
                    )
                }
                if not isinstance(record["subjects"], list) or not all(
                    isinstance(subject, str) for subject in record["subjects"]
                ):
                    self._send_json(400, {"error": "Invalid subject data."})
                    return
                with _open_collection(self.spec) as collection:
                    item_id = collection.add_record(record, enrich=False)
                if item_id is None:
                    self._send_json(409, {"error": "That book edition is already in the catalogue.", "duplicate": True})
                else:
                    self._send_json(201, {"id": item_id})
                return
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
            record: dict[str, Any] = {key: record.get(key, "") for key in (
                "source_id", "title", "artists", "year", "label", "release_format",
                "country", "barcode", "release_group_id", "artist_id",
            )}
            record["genres"] = [genre.strip().lower() for genre in genres if genre.strip()]
            if not record["genres"]:
                record["genres"] = fetch_genres(record["release_group_id"], record["artist_id"])
            record["lastfm_tags"] = fetch_lastfm_tags(", ".join(record["artists"]), record["title"])
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
                    self._send_json(400, {"error": f"Add at least {self.spec.min_items} {self.spec.label}s before building the map ({count} now)."})
                    return
                _build_status.update(running=True, message=f"Building the {self.spec.label} map…")
                threading.Thread(target=_build_map, args=(self.spec, neighbors), daemon=True).start()
            self._send_json(202, {"message": "Map build started."})
            return

        self._send_json(404, {"error": "Not found."})


def serve(spec: MediaSpec, host: str, port: int, cert: str, key: str) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert, keyfile=key)
    handler = type("MediaMapWebHandler", (MediaMapHandler,), {"spec": spec})
    server = ThreadingHTTPServer((host, port), handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    medium = "book" if spec.table == "books" else "LP"
    print(f"{medium} web interface listening on https://{host}:{server.server_port}")
    print(f"On your phone, open https://<this-computer's-local-IP>:{server.server_port} on a trusted Wi-Fi network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping {medium} web interface.")
    finally:
        server.server_close()
