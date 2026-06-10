from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pypdf import PdfReader
import docx2txt
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

CHROMA_COLLECTION_NAME = "employee_policy_documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_environment() -> None:
    """Load environment variables from the .env file and configure Google Gemini."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Check your .env file.")

    genai.configure(api_key=api_key)


def get_data_directory() -> Path:
    """Resolve the policy document data directory from environment variables."""
    data_dir = os.getenv("DATA_DIR", "data")
    return Path(data_dir).resolve()


def find_policy_files(data_dir: Path) -> List[Path]:
    """Find all supported policy files recursively in the data directory."""
    supported_extensions = {".pdf", ".docx", ".txt"}
    return [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.suffix.lower() in supported_extensions
    ]


def load_policy_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """Load text from PDF, DOCX, and TXT policy documents."""
    documents: List[Dict[str, Any]] = []
    policy_files = find_policy_files(data_dir)

    for path in policy_files:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append({
                        "source": path.name,
                        "page": page_number,
                        "content": text.strip(),
                    })
        elif path.suffix.lower() == ".docx":
            text = docx2txt.process(str(path)) or ""
            if text.strip():
                documents.append({
                    "source": path.name,
                    "page": None,
                    "content": text.strip(),
                })
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append({
                    "source": path.name,
                    "page": None,
                    "content": text.strip(),
                })

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
        if start >= len(text):
            break

    return chunks


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Convert loaded documents into embedded-ready chunks."""
    chunks: List[Dict[str, Any]] = []
    for document in documents:
        for chunk_text_content in chunk_text(document["content"], chunk_size, chunk_overlap):
            chunks.append(
                {
                    "source": document["source"],
                    "page": document["page"],
                    "content": chunk_text_content,
                }
            )
    return chunks


def summarize_chunks(chunks: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
    """Print summary statistics for generated text chunks."""
    total_chunks = len(chunks)
    average_length = sum(len(chunk["content"]) for chunk in chunks) / max(total_chunks, 1)

    print("=== Chunking Summary ===")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Total chunks produced: {total_chunks}")
    print(f"Average chunk length: {average_length:.1f} characters")
    print()


def inspect_chunks(chunks: List[Dict[str, Any]], sample_count: int = 3) -> None:
    """Print a small sample of chunk content and metadata for inspection."""
    print("=== Chunk Inspection ===")
    for index, chunk in enumerate(chunks[:sample_count], start=1):
        print(f"Chunk {index}")
        print(f"  source: {chunk['source']}")
        if chunk["page"] is not None:
            print(f"  page: {chunk['page']}")
        snippet = chunk["content"][:250].replace("\n", " ").strip()
        print(f"  content preview: {snippet}...")
        print()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate Gemini embeddings for a list of text chunks."""
    response = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
    
    embeddings: List[List[float]] = []
    if isinstance(response, dict) and "embedding" in response:
        embedding_data = response["embedding"]
        if isinstance(embedding_data, list):
            if embedding_data and isinstance(embedding_data[0], list):
                embeddings = [list(e) for e in embedding_data]
            else:
                embeddings = [list(embedding_data)]
    else:
        raise ValueError("Embedding response structure not recognized.")

    if not embeddings:
        raise ValueError("Embedding response did not include vectors.")
    return embeddings


def validate_embedding_dimensions(embeddings: List[List[float]]) -> bool:
    """Validate that all embeddings share the same vector dimension."""
    if not embeddings:
        return False
    first_dim = len(embeddings[0])
    return all(len(embedding) == first_dim for embedding in embeddings)


def get_persist_directory() -> Path:
    """Resolve the Chroma persistence directory from environment variables."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    return Path(persist_dir).resolve()


def create_vector_store(chunks: List[Dict[str, Any]]) -> chromadb.api.models.Collection.Collection:
    """Create and persist a Chroma vector store from document chunks."""
    persist_dir = get_persist_directory()
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    ids = [f"{index}_{chunk['source']}_{chunk['page'] or 0}" for index, chunk in enumerate(chunks, start=1)]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"] if chunk["page"] is not None else "N/A",
        }
        for chunk in chunks
    ]
    embeddings = embed_texts(documents)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    client.close()
    return collection


def load_vector_store() -> chromadb.api.models.Collection.Collection:
    """Load an existing persisted Chroma collection from disk."""
    persist_dir = get_persist_directory()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist directory does not exist: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = [item.name for item in client.list_collections()]
    if CHROMA_COLLECTION_NAME not in collections:
        raise FileNotFoundError(f"Chroma collection not found: {CHROMA_COLLECTION_NAME}")

    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def summarize_documents(documents: List[Dict[str, Any]]) -> None:
    """Print summary statistics for loaded documents and their metadata."""
    total_documents = len(documents)
    unique_sources = sorted({document["source"] for document in documents})

    print("=== Policy Document Ingestion Summary ===")
    print(f"Total policy document chunks loaded: {total_documents}")
    print(f"Unique document sources: {len(unique_sources)}")
    print(f"Data directory: {get_data_directory()}")
    print()

    print("=== Sample metadata ===")
    for index, document in enumerate(documents[:10], start=1):
        print(f"Document {index}")
        print(f"  source: {document['source']}")
        if document["page"] is not None:
            print(f"  page: {document['page']}")
        print(f"  metadata keys: {sorted(document.keys())}")
        print()

    if total_documents == 0:
        print("Warning: No policy documents were loaded. Verify that supported files exist in the data directory.")


def validate_documents(documents: List[Dict[str, Any]]) -> bool:
    """Validate that all loaded documents contain source metadata and non-empty content."""
    if not documents:
        return False

    for doc in documents:
        if not str(doc["content"]).strip():
            return False
        if "source" not in doc:
            return False

    return True


def main() -> None:
    """Run the ingestion pipeline and persist the Chroma database."""
    load_environment()
    data_dir = get_data_directory()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = load_policy_documents(data_dir)
    summarize_documents(documents)

    if not validate_documents(documents):
        print("Document ingestion failed validation. Please check document content and metadata.")
        return

    chunks = chunk_documents(documents)
    summarize_chunks(chunks)
    inspect_chunks(chunks)

    try:
        collection = create_vector_store(chunks)
        print(f"ChromaDB persisted at: {get_persist_directory()}")
        print("Vector store creation completed successfully.")
    except Exception as error:
        print(f"Vector store creation skipped due to configuration or API error: {error}")

    print("Document ingestion and chunking completed successfully.")


if __name__ == "__main__":
    main()
