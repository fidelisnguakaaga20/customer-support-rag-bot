import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_DIR = Path("indexes/restaurant")
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "chunks.json"


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


class RestaurantVectorStore:
    def __init__(self):
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError(
                "Restaurant index not found. Build it first: python build_restaurant_index.py"
            )

        self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)

        self.index = faiss.read_index(str(INDEX_FILE))

        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.chunks = meta["chunks"]

    def search(self, query: str, top_k: int = 4):
        q = query.strip()
        if not q:
            return []

        emb = self.model.encode([q], convert_to_numpy=True).astype("float32")
        emb = _l2_normalize(emb)

        scores, ids = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self.chunks[int(idx)],
                }
            )
        return results
