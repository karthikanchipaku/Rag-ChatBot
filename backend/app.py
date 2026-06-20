from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import traceback
import json
import uuid
from datetime import datetime

app = FastAPI(title="RAG ChatBot API")

# Allow CORS for development/testing (frontend file or local static server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[dict]


def load_dotenv_if_needed():
    base = os.path.dirname(__file__)
    dotenv = os.path.join(base, ".env")
    if os.path.exists(dotenv):
        with open(dotenv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@app.on_event("startup")
def startup():
    load_dotenv_if_needed()

    # Validate API key
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY in environment; set it in backend/.env or export it.")

    # Initialize Chroma client and collection
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        raise RuntimeError("chromadb is required but not installed")

    base = os.path.dirname(__file__)
    persist_dir = os.path.join(base, "db")
    try:
        app.state.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    except Exception:
        # fallback to default client
        app.state.chroma_client = chromadb.Client()

    # Try to load Chroma collection first; if missing, fall back to local numpy/json store
    try:
        app.state.collection = app.state.chroma_client.get_collection("rag_collection")
        print("Using ChromaDB retriever")
        app.state.use_chroma = True
        app.state.fallback = None
    except Exception:
        # Do not raise here; try to load lightweight persisted store as fallback
        app.state.collection = None
        app.state.use_chroma = False
        base = os.path.dirname(__file__)
        persist_dir = os.path.join(base, "db")
        texts_path = os.path.join(persist_dir, "texts.json")
        embs_npy = os.path.join(persist_dir, "embeddings.npy")
        embs_json = os.path.join(persist_dir, "embeddings.json")
        metas_path = os.path.join(persist_dir, "metadatas.json")

        try:
            import json
            import numpy as _np

            if not os.path.exists(texts_path) or not os.path.exists(metas_path) or (
                not os.path.exists(embs_npy) and not os.path.exists(embs_json)
            ):
                raise FileNotFoundError("Fallback store files missing")

            with open(texts_path, "r", encoding="utf-8") as f:
                texts = json.load(f)
            with open(metas_path, "r", encoding="utf-8") as f:
                metadatas = json.load(f)

            if os.path.exists(embs_npy):
                embs = _np.load(embs_npy, allow_pickle=True)
                embs = [ _np.array(e, dtype=float) for e in embs ]
            else:
                with open(embs_json, "r", encoding="utf-8") as f:
                    embs = [ _np.array(e, dtype=float) for e in json.load(f) ]

            app.state.fallback = {
                "texts": texts,
                "metadatas": metadatas,
                "embeddings": embs,
            }
            app.state.fallback_np = _np
            print("Using NumPy fallback retriever")
        except Exception as e:
            # no fallback available; record state and raise so startup fails loudly
            app.state.fallback = None
            print("Chroma collection not found and fallback load failed:", e)
            raise RuntimeError("Chroma collection 'rag_collection' not found and no fallback store available")

    # Embeddings wrapper (used to embed queries)
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception:
        try:
            from langchain.embeddings import GoogleGenerativeAIEmbeddings
        except Exception:
            GoogleGenerativeAIEmbeddings = None

    if GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("GoogleGenerativeAIEmbeddings not available. Install langchain_google_genai.")

    app.state.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Chat model for generating answers
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        ChatGoogleGenerativeAI = None

    if ChatGoogleGenerativeAI is None:
        # We will try to fall back to using the raw SDK client at request time
        app.state.chat_model = None
        # job tracking for background indexing (ensure exists)
        app.state.index_jobs = {}

    else:
        chat_model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        try:
            app.state.chat_model = ChatGoogleGenerativeAI(model=chat_model_name)
        except Exception:
            app.state.chat_model = None
    # Ensure job tracking dict always exists on app.state
    if not hasattr(app.state, 'index_jobs') or app.state.index_jobs is None:
        app.state.index_jobs = {}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        collection = app.state.collection
        embeddings = app.state.embeddings
    except Exception:
        raise HTTPException(status_code=500, detail="Server not initialized correctly")

    question = req.question
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # Embed query
    try:
        if hasattr(embeddings, "embed_query"):
            q_emb = embeddings.embed_query(question)
        elif hasattr(embeddings, "embed_documents"):
            q_emb = embeddings.embed_documents([question])[0]
        else:
            raise RuntimeError("Embeddings object missing embed_query/embed_documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute query embedding: {e}")

    # Query chroma for top 3 or use fallback
    docs = []
    metadatas = []
    dists = []

    if getattr(app.state, "use_chroma", False):
        try:
            res = app.state.collection.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas", "distances"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chroma query failed: {e}")

        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        if not docs:
            raise HTTPException(status_code=404, detail="No documents found in vector store. Run the ingestion pipeline first.")
    else:
        # Use numpy fallback stored at startup
        fb = getattr(app.state, "fallback", None)
        if fb is None:
            raise HTTPException(status_code=500, detail="No retriever available (neither Chroma nor fallback present)")

        np = app.state.fallback_np
        qv = np.array(q_emb, dtype=float)
        embs = fb["embeddings"]
        # compute cosine similarity
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        scores = [cos_sim(qv, np.array(e, dtype=float)) for e in embs]
        topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]

        docs = [fb["texts"][i] for i, _ in topk]
        metadatas = [fb["metadatas"][i] if i < len(fb["metadatas"]) else {} for i, _ in topk]
        dists = [float(s) for _, s in topk]

    # Build context string from retrieved chunks
    context_parts = []
    sources: List[dict] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, dists), start=1):
        tag = meta.get("source") if isinstance(meta, dict) and meta.get("source") else f"chunk_{i}"
        context_parts.append(f"[{tag}]\n{doc}")
        sources.append({"id": i, "source": tag, "score": float(dist) if dist is not None else None, "metadata": meta})

    context = "\n\n".join(context_parts)

    # Create prompt for Gemini
    # Improved prompt: force model to answer ONLY from provided context, avoid hallucination
    system_prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided context. "
        "If the answer cannot be found in the context, reply exactly: 'I could not find that information in the uploaded documents.' "
        "Do not invent information or refer to chunks by tags like [chunk_1]. Be concise and factual."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources by their tags."

    # Generate answer via Chat model if available, otherwise use raw genai client
    answer_text = None
    try:
        if getattr(app.state, "chat_model", None) is not None:
            # use langchain chat wrapper
            messages = [("system", system_prompt), ("human", user_prompt)]
            ai_msg = app.state.chat_model.invoke(messages)
            answer_text = ai_msg.text if hasattr(ai_msg, "text") else str(ai_msg)
        else:
            # fallback: use google.genai client directly
            try:
                from google import genai
                client = genai.Client()
                # use a safe default model name (Gemini family)
                model_name = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
                resp = client.generate_text(model=model_name, input=user_prompt)
                # attempt to extract text
                answer_text = getattr(resp, "output", None) or getattr(resp, "text", None) or str(resp)
            except Exception as e:
                raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    if not answer_text:
        raise HTTPException(status_code=500, detail="Model returned empty answer")

    return AskResponse(answer=answer_text, sources=sources)


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    info = app.state.index_jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": info.get("status"), "message": info.get("message")}


@app.get("/kb/list")
def kb_list():
    """Return all PDFs in backend/data along with indexed status, size and upload date."""
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    files = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(data_dir, fname)
        try:
            st = os.stat(path)
            uploaded = datetime.fromtimestamp(st.st_mtime).isoformat()
            size = st.st_size
        except Exception:
            uploaded = None
            size = None

        # determine indexed status and chunk count
        indexed = False
        chunks = 0
        try:
            if getattr(app.state, 'use_chroma', False) and getattr(app.state, 'collection', None) is not None:
                try:
                    # try fetching metadatas for this source_pdf
                    res = app.state.collection.get(where={"source_pdf": fname}, include=["metadatas", "ids"])
                    metas = res.get('metadatas', [])
                    # collection.get returns list-of-lists in some versions
                    if isinstance(metas, list) and len(metas) and isinstance(metas[0], list):
                        metas = metas[0]
                    chunks = len(metas) if metas else 0
                    indexed = chunks > 0
                except Exception:
                    # best-effort fallback: try query by metadata via query
                    try:
                        q = app.state.collection.query(n_results=0, where={"source_pdf": fname})
                        # if query didn't error, assume indexed
                        indexed = True
                    except Exception:
                        indexed = False
            else:
                fb = getattr(app.state, 'fallback', None)
                if fb and isinstance(fb.get('metadatas', None), list):
                    chunks = sum(1 for m in fb['metadatas'] if isinstance(m, dict) and m.get('source_pdf') == fname)
                    indexed = chunks > 0
        except Exception:
            indexed = False

        files.append({"filename": fname, "upload_date": uploaded, "size": size, "indexed": indexed, "num_chunks": chunks})

    return {"files": files}


@app.post('/kb/reindex/{filename}')
def kb_reindex(filename: str, background_tasks: BackgroundTasks):
    """Rebuild embeddings for the specified PDF (by filename)."""
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='File not found')

    job_id = uuid.uuid4().hex
    app.state.index_jobs[job_id] = {"status": "queued", "message": f"Reindex queued for {filename}"}

    def do_reindex(path, fname, jid):
        app.state.index_jobs[jid] = {"status": "running", "message": "Reindexing"}
        try:
            # remove existing entries for this file first
            if getattr(app.state, 'use_chroma', False) and getattr(app.state, 'collection', None) is not None:
                try:
                    app.state.collection.delete(where={"source_pdf": fname})
                except Exception:
                    # ignore deletion errors and proceed
                    pass

            else:
                fb = getattr(app.state, 'fallback', None)
                if fb:
                    # remove existing entries
                    keep_texts = []
                    keep_metas = []
                    keep_embs = []
                    for t, m, e in zip(fb.get('texts', []), fb.get('metadatas', []), fb.get('embeddings', [])):
                        if not (isinstance(m, dict) and m.get('source_pdf') == fname):
                            keep_texts.append(t)
                            keep_metas.append(m)
                            keep_embs.append(e)
                    fb['texts'] = keep_texts
                    fb['metadatas'] = keep_metas
                    fb['embeddings'] = keep_embs
                    # persist fallback store
                    persist_dir = os.path.join(os.path.dirname(__file__), 'db')
                    os.makedirs(persist_dir, exist_ok=True)
                    try:
                        import numpy as _np
                        _np.save(os.path.join(persist_dir, 'embeddings.npy'), _np.array(fb['embeddings'], dtype=object))
                    except Exception:
                        with open(os.path.join(persist_dir, 'embeddings.json'), 'w', encoding='utf-8') as f:
                            json.dump([e.tolist() if hasattr(e, 'tolist') else list(e) for e in fb['embeddings']], f)
                    with open(os.path.join(persist_dir, 'texts.json'), 'w', encoding='utf-8') as f:
                        json.dump(fb['texts'], f)
                    with open(os.path.join(persist_dir, 'metadatas.json'), 'w', encoding='utf-8') as f:
                        json.dump(fb['metadatas'], f)

            # now index the file (reuse upload indexing logic)
            # we'll reuse the same splitting/embedding flow as in /upload
            try:
                from langchain.document_loaders import PyPDFLoader
            except Exception:
                PyPDFLoader = None
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except Exception:
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
                            text = getattr(d, 'page_content', str(d)) or ''
                            chunks = self.split_texts([text])
                            for c in chunks:
                                out.append(Doc(c, getattr(d, 'metadata', {}) or {}))
                        return out

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = []
            metadatas = []
            if PyPDFLoader is not None:
                loader = PyPDFLoader(path)
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                for c in chunks:
                    texts.append(c.page_content)
                    md = dict(c.metadata) if hasattr(c, 'metadata') else {}
                    md['source_pdf'] = fname
                    try:
                        excerpt = ' '.join(str(c.page_content).split())[:200].strip()
                        if excerpt:
                            md['excerpt'] = excerpt
                    except Exception:
                        pass
                    metadatas.append(md)
            else:
                from pypdf import PdfReader
                reader = PdfReader(path)
                for page in list(reader.pages):
                    page_text = page.extract_text() or ''
                    chunk_texts = splitter.split_texts([page_text]) if hasattr(splitter, 'split_texts') else splitter.split_text(page_text)
                    for t in chunk_texts:
                        texts.append(t)
                        md = { 'source_pdf': fname }
                        try:
                            excerpt = ' '.join(str(t).split())[:200].strip()
                            if excerpt:
                                md['excerpt'] = excerpt
                        except Exception:
                            pass
                        metadatas.append(md)

            if not texts:
                app.state.index_jobs[jid] = {"status": "failed", "message": "No text extracted"}
                return

            emb_obj = app.state.embeddings
            if hasattr(emb_obj, 'embed_documents'):
                embeddings_list = emb_obj.embed_documents(texts)
            elif callable(emb_obj):
                embeddings_list = emb_obj(texts)
            else:
                embeddings_list = [emb_obj.embed_query(t) for t in texts]

            if getattr(app.state, 'use_chroma', False) and getattr(app.state, 'collection', None) is not None:
                ids = [f"reindex-{int(__import__('time').time())}-{i}" for i in range(len(texts))]
                app.state.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings_list)
                try:
                    if getattr(app.state, 'chroma_client', None) is not None and hasattr(app.state.chroma_client, 'persist'):
                        app.state.chroma_client.persist()
                except Exception:
                    pass
            else:
                fb = getattr(app.state, 'fallback', None)
                base2 = os.path.dirname(__file__)
                persist_dir = os.path.join(base2, 'db')
                os.makedirs(persist_dir, exist_ok=True)
                import numpy as _np
                if fb is None:
                    fb = {'texts': [], 'metadatas': [], 'embeddings': []}
                    app.state.fallback = fb
                    app.state.fallback_np = _np
                fb['texts'].extend(texts)
                fb['metadatas'].extend(metadatas)
                fb['embeddings'].extend([_np.array(e, dtype=float) for e in embeddings_list])
                try:
                    _np.save(os.path.join(persist_dir, 'embeddings.npy'), _np.array(fb['embeddings'], dtype=object))
                except Exception:
                    with open(os.path.join(persist_dir, 'embeddings.json'), 'w', encoding='utf-8') as f:
                        json.dump([e.tolist() if hasattr(e, 'tolist') else list(e) for e in fb['embeddings']], f)
                with open(os.path.join(persist_dir, 'texts.json'), 'w', encoding='utf-8') as f:
                    json.dump(fb['texts'], f)
                with open(os.path.join(persist_dir, 'metadatas.json'), 'w', encoding='utf-8') as f:
                    json.dump(fb['metadatas'], f)

            app.state.index_jobs[jid] = {"status": "success", "message": f"Reindexed {len(texts)} chunks from {fname}"}
        except Exception as e:
            traceback.print_exc()
            app.state.index_jobs[jid] = {"status": "failed", "message": str(e)}

    background_tasks.add_task(do_reindex, path, filename, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.delete('/kb/file/{filename}')
def kb_delete(filename: str):
    """Delete a PDF file and remove its vectors/chunks from the index."""
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='File not found')

    # remove from index
    try:
        if getattr(app.state, 'use_chroma', False) and getattr(app.state, 'collection', None) is not None:
            try:
                app.state.collection.delete(where={"source_pdf": filename})
            except Exception as e:
                # try deleting by metadata search fallback
                try:
                    # no-op if method unavailable
                    pass
                except Exception:
                    pass
        else:
            fb = getattr(app.state, 'fallback', None)
            if fb:
                keep_texts = []
                keep_metas = []
                keep_embs = []
                for t, m, e in zip(fb.get('texts', []), fb.get('metadatas', []), fb.get('embeddings', [])):
                    if not (isinstance(m, dict) and m.get('source_pdf') == filename):
                        keep_texts.append(t)
                        keep_metas.append(m)
                        keep_embs.append(e)
                fb['texts'] = keep_texts
                fb['metadatas'] = keep_metas
                fb['embeddings'] = keep_embs
                # persist fallback store
                persist_dir = os.path.join(os.path.dirname(__file__), 'db')
                os.makedirs(persist_dir, exist_ok=True)
                try:
                    import numpy as _np
                    _np.save(os.path.join(persist_dir, 'embeddings.npy'), _np.array(fb['embeddings'], dtype=object))
                except Exception:
                    with open(os.path.join(persist_dir, 'embeddings.json'), 'w', encoding='utf-8') as f:
                        json.dump([e.tolist() if hasattr(e, 'tolist') else list(e) for e in fb['embeddings']], f)
                with open(os.path.join(persist_dir, 'texts.json'), 'w', encoding='utf-8') as f:
                    json.dump(fb['texts'], f)
                with open(os.path.join(persist_dir, 'metadatas.json'), 'w', encoding='utf-8') as f:
                    json.dump(fb['metadatas'], f)
    except Exception:
        traceback.print_exc()

    # remove file
    try:
        os.remove(path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {e}")

    return {"filename": filename, "deleted": True}


@app.post("/upload")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and schedule indexing in the background. Returns a job id immediately."""
    try:
        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.basename(file.filename)
        save_path = os.path.join(data_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        import uuid
        job_id = uuid.uuid4().hex
        app.state.index_jobs[job_id] = {"status": "queued", "message": "Queued for indexing"}

        def index_file(path, fname, jid):
            app.state.index_jobs[jid] = {"status": "running", "message": "Indexing"}
            try:
                # reuse the same logic as before to extract/split/embed/add
                try:
                    from langchain.document_loaders import PyPDFLoader
                except Exception:
                    PyPDFLoader = None
                try:
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                except Exception:
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

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = []
                metadatas = []
                if PyPDFLoader is not None:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    chunks = splitter.split_documents(docs)
                    for c in chunks:
                        texts.append(c.page_content)
                        md = dict(c.metadata) if hasattr(c, "metadata") else {}
                        md["source_pdf"] = fname
                        # add excerpt for preview (first 200 chars)
                        try:
                            excerpt = " ".join(str(c.page_content).split())[:200].strip()
                            if excerpt:
                                md["excerpt"] = excerpt
                        except Exception:
                            pass
                        metadatas.append(md)
                else:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    for page in list(reader.pages):
                        page_text = page.extract_text() or ""
                        chunk_texts = splitter.split_texts([page_text]) if hasattr(splitter, "split_texts") else splitter.split_text(page_text)
                        for t in chunk_texts:
                            texts.append(t)
                            md = {"source_pdf": fname}
                            try:
                                excerpt = " ".join(str(t).split())[:200].strip()
                                if excerpt:
                                    md["excerpt"] = excerpt
                            except Exception:
                                pass
                            metadatas.append(md)

                if not texts:
                    app.state.index_jobs[jid] = {"status": "failed", "message": "No text extracted"}
                    return

                emb_obj = app.state.embeddings
                if hasattr(emb_obj, "embed_documents"):
                    embeddings_list = emb_obj.embed_documents(texts)
                elif callable(emb_obj):
                    embeddings_list = emb_obj(texts)
                else:
                    embeddings_list = [emb_obj.embed_query(t) for t in texts]

                if getattr(app.state, "use_chroma", False) and getattr(app.state, "collection", None) is not None:
                    ids = [f"upload-{int(__import__('time').time())}-{i}" for i in range(len(texts))]
                    app.state.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings_list)
                    try:
                        if getattr(app.state, "chroma_client", None) is not None and hasattr(app.state.chroma_client, "persist"):
                            app.state.chroma_client.persist()
                    except Exception:
                        pass
                else:
                    fb = getattr(app.state, "fallback", None)
                    base2 = os.path.dirname(__file__)
                    persist_dir = os.path.join(base2, "db")
                    os.makedirs(persist_dir, exist_ok=True)
                    import json
                    import numpy as _np
                    if fb is None:
                        fb = {"texts": [], "metadatas": [], "embeddings": []}
                        app.state.fallback = fb
                        app.state.fallback_np = _np
                    fb["texts"].extend(texts)
                    fb["metadatas"].extend(metadatas)
                    fb["embeddings"].extend([_np.array(e, dtype=float) for e in embeddings_list])
                    try:
                        _np.save(os.path.join(persist_dir, "embeddings.npy"), _np.array(fb["embeddings"], dtype=object))
                    except Exception:
                        with open(os.path.join(persist_dir, "embeddings.json"), "w", encoding="utf-8") as f:
                            json.dump([e.tolist() if hasattr(e, 'tolist') else list(e) for e in fb["embeddings"]], f)
                    with open(os.path.join(persist_dir, "texts.json"), "w", encoding="utf-8") as f:
                        json.dump(fb["texts"], f)
                    with open(os.path.join(persist_dir, "metadatas.json"), "w", encoding="utf-8") as f:
                        json.dump(fb["metadatas"], f)

                app.state.index_jobs[jid] = {"status": "success", "message": f"Indexed {len(texts)} chunks from {fname}"}
            except Exception as e:
                traceback.print_exc()
                app.state.index_jobs[jid] = {"status": "failed", "message": str(e)}

        background_tasks.add_task(index_file, save_path, filename, job_id)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
