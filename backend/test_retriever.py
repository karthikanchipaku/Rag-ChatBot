"""Test script to verify ChromaDB retrieval using Gemini embeddings.

This script loads the persisted ChromaDB from `backend/db`, performs a
similarity search for the question "What is Artificial Intelligence?", and
prints the top 3 chunk contents with similarity scores.
"""

import os
from typing import Optional

try:
    # prefer langchain_google_genai per requirements
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    try:
        from langchain.embeddings import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None


def test_retriever(persist_dir: Optional[str] = None):
    base = os.path.dirname(__file__)
    persist_dir = persist_dir or os.path.join(base, "db")

    if chromadb is None:
        raise SystemExit("chromadb client not installed; cannot run retriever test")
    if os.path.exists(persist_dir):
        try:
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
        except Exception:
            client = chromadb.Client()
    else:
        print(f"Warning: persist_dir {persist_dir} not found; using default chromadb client location")
        client = chromadb.Client()

    try:
        collection = client.get_collection("rag_collection")
    except Exception:
        collection = client.create_collection("rag_collection")

    query = "What is Artificial Intelligence?"

    # ensure GOOGLE_API_KEY is loaded
    if not os.environ.get("GOOGLE_API_KEY"):
        dotenv_path = os.path.join(base, ".env")
        if os.path.exists(dotenv_path):
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Missing GOOGLE_API_KEY in environment; set it in backend/.env or export it.")

    if GoogleGenerativeAIEmbeddings is None:
        raise SystemExit("GoogleGenerativeAIEmbeddings not available. Install langchain_google_genai.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # use LangChain embedding wrapper to create query embedding
    if hasattr(embeddings, "embed_query"):
        q_emb = embeddings.embed_query(query)
    elif hasattr(embeddings, "embed_documents"):
        q_emb = embeddings.embed_documents([query])[0]
    else:
        raise SystemExit("Embeddings object does not support embed_query/embed_documents")

    res = collection.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas", "distances"])

    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not docs:
        # Fallback: load lightweight persisted store from disk
        import json
        import math
        import numpy as np

        texts_path = os.path.join(persist_dir, "texts.json")
        embs_npy = os.path.join(persist_dir, "embeddings.npy")
        embs_json = os.path.join(persist_dir, "embeddings.json")

        if not os.path.exists(texts_path):
            print("No documents found in chromadb and no local store available.")
            return

        with open(texts_path, "r", encoding="utf-8") as f:
            texts = json.load(f)

        if os.path.exists(embs_npy):
            embs = np.load(embs_npy, allow_pickle=True)
            embs = [np.array(e, dtype=float) for e in embs]
        elif os.path.exists(embs_json):
            with open(embs_json, "r", encoding="utf-8") as f:
                embs = [np.array(e, dtype=float) for e in json.load(f)]
        else:
            print("No embeddings file found in local store.")
            return

        # compute similarity (cosine) using the same embedding for the query
        qv = np.array(q_emb, dtype=float)

        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        scores = [cos_sim(qv, np.array(e, dtype=float)) for e in embs]
        topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]

        print(f"Top {len(topk)} chunks for: '{query}'\n")
        for i, (idx, score) in enumerate(topk, start=1):
            print(f"--- Result {i} (score: {score}) ---")
            print(texts[idx])
            print()
        return

    print(f"Top {len(docs)} chunks for: '{query}'\n")
    for i, (doc_text, score) in enumerate(zip(docs, dists), start=1):
        print(f"--- Result {i} (distance: {score}) ---")
        print(doc_text)
        print()


if __name__ == "__main__":
    test_retriever()
