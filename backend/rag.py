"""RAG helper: retrieval pipeline and placeholder for generation.

This module provides utilities to load a persisted ChromaDB, perform a
retrieval, and (later) call a LLM to generate an answer from the top chunks.
Currently the generation step is a stub — retrieval is implemented and tested
by `test_retriever.py`.
"""

import os
from typing import List, Tuple

from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma


def _get_vectordb(persist_dir: str | None = None):
    base = os.path.dirname(__file__)
    persist_dir = persist_dir or os.path.join(base, "db")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb


def retrieve(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """Return the top-k retrieved documents (as LangChain Documents).

    Returns a list of Documents.
    """
    vectordb = _get_vectordb()
    results = vectordb.similarity_search_with_score(query, k=k)
    # results is list of (Document, score)
    return results


def answer(query: str, k: int = 3) -> dict:
    """Perform retrieval and (placeholder) generation.

    Current behavior: returns the top-k chunks and a placeholder for the
    eventual Gemini-generated answer.
    """
    results = retrieve(query, k=k)
    chunks = [doc.page_content for doc, _score in results]

    # Placeholder: do not call Gemini chat model yet. Implement generation later.
    return {
        "query": query,
        "top_k": k,
        "chunks": chunks,
        "answer": "<generation-not-implemented>",
    }

