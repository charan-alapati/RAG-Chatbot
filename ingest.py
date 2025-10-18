"""
ingest.py
---------
Read plain-text documents from ./docs/ and build:
- sentence embeddings (SentenceTransformer)
- FAISS index saved to faiss.index
- docs_meta.json with the raw document texts and filenames

Usage:
    python ingest.py
"""

import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

DOCS_DIR = "docs"
INDEX_PATH = "faiss.index"
META_PATH = "docs_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"  # compact, fast

def load_texts(folder=DOCS_DIR):
    texts = []
    meta = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                continue
            texts.append(txt)
            meta.append({"filename": fname, "text": txt})
    return texts, meta

def build_and_save_index(texts, model_name=EMBED_MODEL, index_path=INDEX_PATH, meta_path=META_PATH):
    print("Loading sentence-transformers model:", model_name)
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} documents ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize embeddings (optional but often helpful)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors == cosine similarity
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index written to: {index_path}")

def save_meta(meta, meta_path=META_PATH):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Docs metadata written to: {meta_path}")

if __name__ == "__main__":
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(f"Create a folder named '{DOCS_DIR}' and put .txt documents inside before running this script.")
    texts, meta = load_texts(DOCS_DIR)
    if len(texts) == 0:
        raise SystemExit("No .txt documents found in docs/. Add at least one .txt file.")
    build_and_save_index(texts)
    save_meta(meta)
    print("Ingestion complete.")
