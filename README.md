# 🤖 RAG ChatBot

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDFs, build a knowledge base, and ask natural-language questions grounded in your documents.

Built with a clean separation between a high-performance FastAPI backend and a modern static frontend.

---

## 🏗️ System Architecture

The application is decoupled into two primary tiers:

1. **Backend API** – FastAPI server that handles PDF ingestion, embedding generation, vector storage (ChromaDB), retrieval, and answer generation via Google Gemini.
2. **Frontend Dashboard** – Clean static HTML/CSS/JS interface for uploading PDFs, managing the knowledge base, and chatting with the RAG system.

---

## 🚀 Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, LangChain, Google Generative AI (Gemini), ChromaDB
- **Embeddings & LLM**: Google Gemini (`gemini-embedding-001` + `gemini-2.5-flash`)
- **Vector Store**: ChromaDB (with NumPy fallback)
- **PDF Processing**: pdfminer.six / pypdf + RecursiveCharacterTextSplitter
- **Frontend**: Vanilla HTML, CSS, JavaScript
- **DevOps**: Ready for Docker / local virtual environment

```text
Rag-ChatBot/
├── backend/                     # FastAPI Backend
│   ├── app.py                   # Main FastAPI application + API routes
│   ├── build_db.py              # ETL script – ingest PDFs → ChromaDB
│   ├── rag.py                   # Retrieval helpers
│   ├── requirements.txt         # Python dependencies
│   ├── data/                    # Uploaded PDFs (created at runtime)
│   └── db/                      # Persisted ChromaDB + fallback store
├── frontend/                    # Static Frontend
│   ├── index.html               # Main chat interface
│   ├── welcome.html             # Landing page
│   ├── script.js                # Frontend logic
│   ├── style.css
│   └── welcome.css
├── Screenshots_*.png            # Demo screenshots
├── .gitignore
└── README.md


## 📈 Key Features

- **PDF Knowledge Base** – Upload one or multiple PDFs; they are automatically chunked, embedded, and indexed.
- **Smart Retrieval** – Top-k similarity search over ChromaDB (with graceful NumPy fallback).
- **Grounded Answers** – Gemini generates responses strictly from retrieved context (hallucination-resistant prompt).
- **Source Citations** – Every answer returns the source documents/chunks used.
- **Knowledge Base Management** – List uploaded PDFs, see indexing status & chunk counts.
- **Background Indexing** – Asynchronous PDF processing with job status tracking.
- **Modern Chat UI** – Clean, responsive frontend for uploading documents and chatting.

---

## 💻 Local Setup & Installation

### Prerequisites
- Python 3.10+
- Google Gemini API Key [](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/karthikanchipaku/Rag-ChatBot.git
cd Rag-ChatBot
2. Create & activate virtual environment
Bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
3. Install dependencies
Bashpip install -r backend/requirements.txt
4. Set your Gemini API Key
Create a file backend/.env:
envGOOGLE_API_KEY=your_gemini_api_key_here
# Optional – override default chat model
# GEMINI_CHAT_MODEL=gemini-2.5-flash
5. (Optional) Pre-build the vector database
Place any PDF files inside backend/data/ and run:
Bashcd backend
python build_db.py
6. Start the Backend
Bashcd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
API will be available at:
→ http://127.0.0.1:8000
→ Interactive docs: http://127.0.0.1:8000/docs
7. Open the Frontend
Simply open the file in your browser:
textfrontend/index.html
(or serve it with any static server if you prefer)

📡 Sample API Endpoints
MethodEndpointDescriptionPOST/askAsk a question (returns answer + sources)POST/uploadUpload PDF(s) for indexingGET/upload/status/{job_id}Check indexing job statusGET/kb/listList all PDFs in the knowledge base
Full interactive documentation is available at /docs once the server is running.

🖼️ Screenshots
<img src="Screenshots_Question-Answer.png" alt="Chat Interface">
<img src="Screenshots_knowledge-Base.png" alt="Knowledge Base">
<img src="screenshorts_pdf-upload.png" alt="PDF Upload">
<img src="screenshorts_chart-ui.png" alt="Chart UI">

🛠️ How It Works (High Level)

Ingestion – PDFs are loaded → split into overlapping chunks → embedded with Gemini → stored in ChromaDB.
Query Time – User question is embedded → top-k similar chunks are retrieved.
Generation – Retrieved context + question are sent to Gemini with a strict “answer only from context” prompt.
Response – Answer + source citations are returned to the frontend.


📄 License
This project is open source and available under the MIT License.

