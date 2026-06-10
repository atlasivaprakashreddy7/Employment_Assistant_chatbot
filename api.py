import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot import ask_policy_question, load_environment

# Configuration
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
APP_TITLE = "Employment Policy Assistant"

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}


def is_allowed_upload_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Employee HR policy question.")


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Policy-grounded response to the employee question.")
    sources: List[str] = Field(default_factory=list, description="Retrieved source citations from policy documents.")


class UploadResponse(BaseModel):
    uploaded_files: List[str] = Field(..., description="Names of uploaded documents saved by the backend.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Backend health status.")
    data_dir: str = Field(..., description="Current policy data directory.")
    files: List[str] = Field(..., description="List of files currently available in the data directory.")


app = FastAPI(
    title="Employee Policy Assistant API",
    description="API for asking HR policy questions and receiving grounded responses from company policy documents.",
    version="1.0.0",
)

logger = logging.getLogger("employee_policy_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """Load environment variables on startup."""
    logger.info("Starting backend and loading environment variables.")
    load_dotenv()
    load_environment()


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Answer an employee's HR policy question using the RAG pipeline."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        logger.info("Received ask request: %s", request.question[:80])
        result = ask_policy_question(question, k=3, debug=False)
        logger.info("Ask result produced %s sources", len(result.get("sources", [])))
        return AnswerResponse(answer=result["answer"], sources=result["sources"])
    except EnvironmentError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to process the policy question at this time.")


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Accept uploaded policy documents and save them into the data directory."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    uploaded_files: List[str] = []

    for upload in files:
        logger.info("Uploading document: %s", upload.filename)
        if not is_allowed_upload_file(upload.filename):
            allowed = ", ".join(ext.lstrip(".") for ext in sorted(ALLOWED_UPLOAD_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {upload.filename}. Allowed types: {allowed}.",
            )

        file_path = data_dir / upload.filename
        try:
            content = await upload.read()
            file_path.write_bytes(content)
            uploaded_files.append(upload.filename)
        except Exception as exc:
            logger.exception("Failed to save uploaded file %s", upload.filename)
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {exc}")

    return UploadResponse(uploaded_files=uploaded_files)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report backend health and current document directory contents."""
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [str(path.name) for path in sorted(data_dir.glob("*")) if path.is_file()]
    logger.info("Health check requested. %s files present in %s", len(files), data_dir)
    return HealthResponse(status="healthy", data_dir=str(data_dir), files=files)
