# Employee Policy Assistant

A retrieval-augmented HR policy assistant that answers employee questions strictly from company policy documents.

## Architecture

Policy Documents (`.pdf`, `.docx`, `.txt`)
  ↓
Text Extraction
  ↓
Chunking / Overlap
  ↓
Gemini Embeddings (`models/embedding-001`)
  ↓
ChromaDB
  ↓
Retriever (top-k)
  ↓
Prompt Template
  ↓
Gemini LLM (`gemini-2.5-flash`)
  ↓
Final Response

## Project Structure

- `data/` — store your HR policy documents (`.pdf`, `.docx`)
- `chroma_db/` — persisted Chroma vector store data
- `ingestion.py` — ingest documents, chunk text, generate embeddings, and persist Chroma
- `chatbot.py` — retrieval, prompt construction, Gemini integration, and hallucination control
- `api.py` — FastAPI endpoint for question answering
- `requirements.txt` — Python dependencies
- `.env` — environment configuration file

## Setup

1. Create and activate a Python virtual environment:

```powershell
cd c:\Users\siva9\OneDrive\Desktop\Employee_Policy_Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create or update `.env`:

```text
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
DATA_DIR=data
CHROMA_PERSIST_DIR=chroma_db
```

4. Add your HR policy PDFs and DOCX documents to `data/`.

## Installation

Dependencies are defined in `requirements.txt`:

- `fastapi` — web API framework
- `uvicorn[standard]` — ASGI server
- `python-dotenv` — `.env` configuration loader
- `chromadb` — local vector database for semantic retrieval
- `docx2txt` — DOCX text extraction
- `pypdf` — PDF text extraction
- `google-generativeai` — Gemini embeddings and chat model support

## Execution

### 1. Ingest documents and build ChromaDB

```powershell
python ingestion.py
```

This will:
- load all PDF, DOCX, and TXT documents from `data/`
- extract text and metadata
- split text into 1000-character chunks with 200-character overlap
- generate Gemini embeddings
- persist the Chroma vector store in `chroma_db/`

### 2. Run the RAG chatbot locally

```powershell
python chatbot.py
```

This executes sample queries through the retrieval pipeline and prints:
- retrieval debug output
- similarity scores
- generated answers grounded in policy documents

### 3. Start the FastAPI service

```powershell
uvicorn api:app --reload
```

Open API docs at:

- `http://127.0.0.1:8000/docs`

## Streamlit Frontend

A modern ChatGPT-style UI is available via Streamlit.

Start the frontend with:

```powershell
streamlit run streamlit_app.py
```

This interface includes:
- dark enterprise styling
- chat bubbles with timestamps
- markdown-rendered answers with code, tables, and bullets
- file upload support for policy documents
- conversation download and clear chat
- backend health status and document previews

## API Testing

Send a POST request to `/ask`:

```powershell
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Can I carry forward unused leaves?\"}"
```

Expected response:

```json
{
  "answer": "I could not find information related to your query in the available policy documents. Please contact HR for clarification."
}
```

If the information is present in the documents, the answer will be grounded in policy text.

## Sample Queries

- `How many annual leaves do I have?`
- `Can unused leaves be carried forward?`
- `What is the hotel reimbursement limit for domestic travel?`
- `Can I work from home more than three days a week?`

## Troubleshooting

- `GOOGLE_API_KEY is not set`
  - ensure the `.env` file contains a valid key and the service is restarted
- `Chroma persist directory does not exist`
  - run `python ingestion.py` first to create `chroma_db/`
- `No policy documents were loaded`
  - add PDF or DOCX files to `data/`
- `Failed to generate embeddings`
  - verify network access and Google API credentials

## Hallucination Control

The assistant uses similarity thresholding to avoid generating answers when retrieval confidence is low. If the top retrieved vector is below the similarity threshold, the system responds with a safe fallback:

`I could not find information related to your query in the available policy documents. Please contact HR for clarification.`

This ensures the assistant only answers when content is grounded in company policies.
