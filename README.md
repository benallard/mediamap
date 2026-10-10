# Semantic Book Mapper — prototype 0.1

This first prototype lets you manually enter a book's title and author. It creates a multilingual semantic embedding and places the book on a persistent 2D *semantic map*. It saves the embedding and coordinates in a local SQLite database.

## Requirements

- Python 3.10 or newer
- Internet access on the first run to download the sentence-transformers model
- A few GB of free disk space is a safe allowance for Python packages and the model

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Use

Add a book interactively:

```bash
python library.py add
```

Or supply details directly:

```bash
python library.py add --title "Der Name der Rose" --author "Umberto Eco" --notes "historical mystery, medieval monastery, semiotics"
```

List all books:

```bash
python library.py list
```

Find a saved book:

```bash
python library.py find "Eco"
```

The database `library.sqlite3` is created beside the script. Back it up along with the script.

## Important limitations of this prototype

1. The 2D coordinates are an incremental *visualization of semantic neighbourhoods*, not physical shelf coordinates. Physical shelf capacity and geometry are not yet modeled.
2. Existing coordinates are not moved when new books are added. That gives stable lookups, but an incremental map cannot guarantee a globally optimal arrangement for every future book.
3. Title/author is used as the duplicate key. Different editions with the same title and author may need separate IDs in a later version.
4. Embeddings depend on the installed model version. The stored vectors and coordinates are reused; do not delete the database if you want to preserve assignments.
5. For better German book placement, fill in optional notes with a short description, subject, or keywords when the title and author alone are ambiguous.
6. This is a prototype, not yet a shelf-packing solution. Before physically shelving hundreds of books, we should define the real bookcase/shelf layout and add a placement layer that respects capacity.

## Why this model?

`paraphrase-multilingual-MiniLM-L12-v2` supports German and many other languages. It runs locally after download, so book titles and notes do not need to be sent to an embedding API.
