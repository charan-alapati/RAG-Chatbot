
"""
retriever.py
------------
Utilities for loading the FAISS index and performing vector retrieval.

Exposes:
- Retriever class with method retrieve(query, k)

Requires:
- sentence-transformers
- faiss
"""

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

INDEX_PATH = "faiss.index"
META_PATH = "docs_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH, model_name=EMBED_MODEL):
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run ingest.py first.")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}. Run ingest.py first.")

        print("Loading embedding model:", model_name)
        self.model = SentenceTransformer(model_name)
        print("Loading FAISS index:", index_path)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve(self, query, k=3):
        # encode and normalize
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = self.meta[idx]
            results.append({"score": float(score), "filename": item.get("filename"), "text": item.get("text")})
        return results

if __name__ == "__main__":
    # quick smoke test
    r = Retriever()
    q = "What are best practices for unit testing?"
    print("Query:", q)
    res = r.retrieve(q, k=3)
    for i, r in enumerate(res, 1):
        print(f"\nResult {i} (score={r['score']:.4f}):\nFilename: {r['filename']}\nText excerpt: {r['text'][:400]}...\n")
