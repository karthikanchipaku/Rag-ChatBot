"""Ingest PDFs into a ChromaDB vector store using LangChain + Gemini embeddings.

Steps performed:
- Load PDFs from `backend/data` using `PyPDFLoader`
- Split with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- Embed with `GoogleGenerativeAIEmbeddings` (Gemini embeddings) using model
  `models/embedding-001`
- Persist a ChromaDB to `backend/db`

Requirements: set Google credentials as required by the Google Generative AI SDK.
"""

import glob
import os
from typing import List

try:
    from langchain.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None
    # Fallback will use pypdf to extract page text
    from pypdf import PdfReader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Provide a minimal fallback splitter with similar behavior
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_texts(self, texts):
            out = []
            for text in texts:
                if not text:
                    continue
                start = 0
                step = max(1, self.chunk_size - self.chunk_overlap)
                while start < len(text):
                    chunk = text[start : start + self.chunk_size]
                    out.append(chunk)
                    start += step
            return out

        def split_text(self, text):
            return self.split_texts([text])

        def split_documents(self, docs):
            class Doc:
                def __init__(self, page_content, metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

            out = []
            for d in docs:
                text = getattr(d, "page_content", str(d)) or ""
                chunks = self.split_texts([text])
                for c in chunks:
                    out.append(Doc(c, getattr(d, "metadata", {}) or {}))
            return out
try:
    # Preferred import per requirements
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    try:
        # Fallback to langchain's shim if available
        from langchain.embeddings import GoogleGenerativeAIEmbeddings
    except Exception:
        GoogleGenerativeAIEmbeddings = None
try:
    from langchain.vectorstores import Chroma
except Exception:
    Chroma = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None


def build(persist_dir: str | None = None):
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")
    persist_dir = persist_dir or os.path.join(base, "db")

    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    print(f"Number of PDFs found: {len(pdf_paths)}")

    texts: List[str] = []
    metadatas: List[dict] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # verbose per-PDF processing report
    processed = []  # list of dicts: {path, name, pages, chunks}
    skipped = []  # list of dicts: {path, name, error}

    for p in pdf_paths:
        name = os.path.basename(p)
        print(f"Processing PDF: {p}")
        chunk_count_for_pdf = 0
        page_count = None
        try:
            if PyPDFLoader is not None:
                try:
                    loader = PyPDFLoader(p)
                    docs = loader.load()
                except Exception as e:
                    # loader-level failure
                    raise RuntimeError(f"PyPDFLoader failed: {e}")
                # docs may be a list of page-like Document objects
                page_count = len(docs) if hasattr(docs, '__len__') else None
                chunks = splitter.split_documents(docs)
                for c in chunks:
                    texts.append(c.page_content)
                    md = dict(c.metadata) if hasattr(c, "metadata") else {}
                    md["source_pdf"] = name
                    # add a short excerpt (first 200 chars) to metadata for preview
                    try:
                        excerpt = " ".join(str(c.page_content).split())[:200].strip()
                        if excerpt:
                            md["excerpt"] = excerpt
                    except Exception:
                        pass
                    metadatas.append(md)
                    chunk_count_for_pdf += 1
            else:
                # fallback: extract text per page using pypdf and split
                try:
                    reader = PdfReader(p)
                except Exception as e:
                    raise RuntimeError(f"PdfReader failed: {e}")
                try:
                    pages_iter = list(reader.pages)
                    page_count = len(pages_iter)
                except Exception as e:
                    raise RuntimeError(f"Failed to iterate pages: {e}")
                for page in pages_iter:
                    page_text = page.extract_text() or ""
                    # split_texts returns a list of strings
                    try:
                        chunk_texts = splitter.split_texts([page_text])
                    except Exception:
                        # older/newer langchain may expose split_text
                        chunk_texts = splitter.split_text(page_text)
                        if isinstance(chunk_texts, str):
                            chunk_texts = [chunk_texts]
                    for t in chunk_texts:
                        texts.append(t)
                        md = {"source_pdf": name}
                        try:
                            excerpt = " ".join(str(t).split())[:200].strip()
                            if excerpt:
                                md["excerpt"] = excerpt
                        except Exception:
                            pass
                        metadatas.append(md)
                        chunk_count_for_pdf += 1

            print(f"  Pages: {page_count if page_count is not None else 'unknown'}; Chunks: {chunk_count_for_pdf}")
            processed.append({"path": p, "name": name, "pages": page_count, "chunks": chunk_count_for_pdf})
        except Exception as e:
            err = str(e)
            print(f"  Error processing '{name}': {err}")
            skipped.append({"path": p, "name": name, "error": err})
            # continue to next PDF
            continue

    print(f"Number of chunks created: {len(texts)}")

    if len(texts) == 0:
        print("No chunks to index. Exiting.")
        return

    # Use Gemini / Google Generative AI embeddings (model: models/embedding-001)
    print("Creating embeddings (Google Generative AI / Gemini)")

    # Ensure GOOGLE_API_KEY is present (load .env if necessary)
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
        raise SystemExit(
            "Missing GOOGLE_API_KEY environment variable. Set it in backend/.env or export it before running."
        )

    if GoogleGenerativeAIEmbeddings is None:
        raise SystemExit(
            "GoogleGenerativeAIEmbeddings not available. Install 'langchain_google_genai' and the Google GenAI client."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Create and persist ChromaDB from texts
    print(f"Persisting ChromaDB to: {persist_dir}")

    def _get_embeddings_for_texts(texts_list):
        # embeddings may be a callable or a LangChain embeddings object
        if callable(embeddings):
            return embeddings(texts_list)
        if hasattr(embeddings, "embed_documents"):
            return embeddings.embed_documents(texts_list)
        if hasattr(embeddings, "embed_query"):
            return [embeddings.embed_query(t) for t in texts_list]
        raise RuntimeError("Unsupported embeddings object; cannot compute embeddings")

    if Chroma is not None:
        try:
            vectordb = Chroma.from_texts(texts, embedding_function=embeddings, metadatas=metadatas, persist_directory=persist_dir)
            try:
                vectordb.persist()
            except Exception:
                pass
            print("ChromaDB (LangChain) persisted successfully.")
            return
        except Exception as e:
            print("LangChain Chroma.from_texts failed, falling back to chromadb client:", e)

    if chromadb is None or Settings is None:
        print("chromadb not available. Install chromadb or ensure langchain.vectorstores.Chroma is available.")
        return

    # Use chromadb client directly
    try:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    except Exception:
        client = chromadb.Client()

    coll_name = "rag_collection"
    try:
        collection = client.get_collection(coll_name)
    except Exception:
        collection = client.create_collection(coll_name)

    print("Computing embeddings for texts...")
    embeddings_list = _get_embeddings_for_texts(texts)

    ids = [f"chunk-{i}" for i in range(len(texts))]
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings_list)

    try:
        client.persist()
    except Exception:
        pass

    if os.path.exists(persist_dir) or chromadb is not None:
        print("ChromaDB persisted successfully via chromadb client.")
    else:
        print("Warning: ChromaDB persistence may have failed.")

    # Also persist a lightweight local store (texts, metadatas, embeddings)
    try:
        os.makedirs(persist_dir, exist_ok=True)
        import json
        try:
            import numpy as np
            np.save(os.path.join(persist_dir, "embeddings.npy"), np.array(embeddings_list, dtype=object))
        except Exception:
            with open(os.path.join(persist_dir, "embeddings.json"), "w", encoding="utf-8") as f:
                json.dump(embeddings_list, f)

        with open(os.path.join(persist_dir, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f)
        with open(os.path.join(persist_dir, "metadatas.json"), "w", encoding="utf-8") as f:
            json.dump(metadatas, f)
        # write parse report summarizing processing and any errors
        try:
            report_path = os.path.join(persist_dir, "parse_report.txt")
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(f"PDFs found: {len(pdf_paths)}\n")
                rf.write(f"PDFs processed successfully: {len(processed)}\n")
                for p in processed:
                    rf.write(f"- {p['name']}: pages={p['pages']}, chunks={p['chunks']}\n")
                rf.write(f"PDFs skipped: {len(skipped)}\n")
                for s in skipped:
                    rf.write(f"- {s['name']}: ERROR: {s['error']}\n")
                rf.write(f"Total chunks indexed: {len(texts)}\n")
            print(f"Wrote parse report to: {report_path}")
        except Exception as e:
            print("Failed to write parse_report.txt:", e)
        print(f"Also persisted lightweight store to {persist_dir}")
    except Exception as e:
        print("Failed to persist lightweight store:", e)


if __name__ == "__main__":
    build()
