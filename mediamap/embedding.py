from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIR = HERE / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def load_embedder() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependencies. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if (MODEL_DIR / "modules.json").is_file():
        return SentenceTransformer(str(MODEL_DIR), local_files_only=True)

    print(f"Downloading embedding model {EMBEDDING_MODEL} for local use …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    print(f"Saved embedding model locally to {MODEL_DIR}.")
    return model


def encode_text(text: str):
    return load_embedder().encode([text], normalize_embeddings=True)[0]
