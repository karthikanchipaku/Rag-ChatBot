# RAG-CHATBOT

This repository contains a minimal scaffold for a Retrieval-Augmented Generation (RAG) chatbot.

Structure:

- `backend/`: FastAPI backend and ingestion scripts
- `frontend/`: Static UI files
- `venv/`: (placeholder) local virtual environment
 
 screenshorts:
 ![alt text](screenshorts_chart-ui.png) ![alt text](screenshorts_pdf-upload.png) ![alt text](Screenshots_knowledge-Base.png) ![alt text](Screenshots_Question-Answer.png)

Run the backend:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
```
