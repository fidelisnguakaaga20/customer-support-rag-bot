import os
import json
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


KB_FILE = Path("knowledge_bases/restaurant/restaurant_faq.txt")
OUT_DIR = Path("indexes/restaurant")
INDEX_FILE = OUT_DIR / "faiss.index"
META_FILE = OUT_DIR / "chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = clean_text(text)
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def main():
    if not KB_FILE.exists():
        raise FileNotFoundError(f"Missing: {KB_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = KB_FILE.read_text(encoding="utf-8")
    chunks = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)

    if len(chunks) == 0:
        raise ValueError("No text chunks found. Your restaurant_faq.txt is empty.")

    print(f"[OK] Loaded FAQ and created {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    embeds = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # cosine similarity via inner product + normalization
    embeds = embeds.astype("float32")
    embeds = l2_normalize(embeds)

    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved index: {INDEX_FILE}")
    print(f"[OK] Saved chunks: {META_FILE}")
    print("[DONE] Restaurant index build complete.")


if __name__ == "__main__":
    main()
