from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

SAFE_FALLBACK_ANSWER = "I could not find information related to your query in the available policy documents. Please contact HR for clarification."
SIMILARITY_THRESHOLD = 0.6
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHROMA_COLLECTION_NAME = "employee_policy_documents"


def load_environment() -> None:
    """Load environment variables from the .env file and configure Gemini."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Check your .env file.")

    genai.configure(api_key=api_key)


def get_persist_directory() -> Path:
    """Resolve the Chroma persistence directory from environment variables."""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    return Path(persist_dir).resolve()


def get_vector_store_collection() -> chromadb.api.models.Collection.Collection:
    """Load the persisted Chroma collection for employee policy documents."""
    persist_dir = get_persist_directory()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist directory does not exist: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = [collection.name for collection in client.list_collections()]
    if CHROMA_COLLECTION_NAME not in collections:
        raise FileNotFoundError(f"Chroma collection not found: {CHROMA_COLLECTION_NAME}")

    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate Gemini embeddings for a list of texts."""
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


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    numerator = sum(x * y for x, y in zip(a, b))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def retrieve_documents_with_scores(collection: chromadb.api.models.Collection.Collection, question: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the top-k documents and their similarity scores."""
    question_embedding = embed_texts([question])[0]
    query_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas", "embeddings"],
    )

    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    embeddings = query_results.get("embeddings", [[]])[0]

    scored_results: List[Dict[str, Any]] = []
    for document, metadata, embedding in zip(documents, metadatas, embeddings):
        score = cosine_similarity(question_embedding, list(embedding))
        scored_results.append(
            {
                "source": metadata.get("source", "unknown source"),
                "page": metadata.get("page"),
                "content": document,
                "score": score,
            }
        )

    return scored_results


def build_context_from_documents(results: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a grounded context string."""
    sections: List[str] = []
    for result in results:
        header = f"Source: {result['source']}"
        if result["page"] is not None:
            header += f" | Page: {result['page']}"
        sections.append(f"{header}\n{result['content']}")
    return "\n\n".join(sections)


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build the prompt text for the Gemini chat model."""
    context = build_context_from_documents(results)
    return (
        "You are an HR policy assistant. Answer the question using only the supplied policy excerpts.\n"
        "Do not use any external knowledge beyond the provided content.\n"
        "Do not hallucinate. If the answer is not present in the policy excerpts, respond exactly with:\n"
        '"I could not find this information in the available policy documents."\n\n'
        f"Question:\n{question}\n\n"
        f"Retrieved policy context:\n{context}\n\n"
        "Answer based only on the policy documents above."
    )


def generate_answer(question: str, results: List[Dict[str, Any]]) -> str:
    """Generate a grounded policy answer from retrieved documents."""
    if not results:
        return SAFE_FALLBACK_ANSWER

    prompt = build_prompt(question, results)
    model = genai.GenerativeModel(CHAT_MODEL)
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return getattr(response, "text", SAFE_FALLBACK_ANSWER).strip()


def is_retrieval_confident(results: List[Dict[str, Any]], threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """Check whether the top retrieved results exceed the confidence threshold."""
    if not results:
        return False
    best_score = max(result["score"] for result in results)
    return best_score >= threshold


def build_source_references(results: List[Dict[str, Any]]) -> List[str]:
    """Create a human-readable list of retrieved source citations."""
    references: List[str] = []
    for result in results:
        page_info = f" (page {result['page']})" if result['page'] is not None and result['page'] != 'N/A' else ""
        references.append(f"{result['source']}{page_info}")
    return references


def ask_policy_question(question: str, k: int = 3, debug: bool = False) -> Dict[str, Any]:
    """Run the complete retrieval augmented generation pipeline for a user question."""
    load_environment()
    collection = get_vector_store_collection()
    results = retrieve_documents_with_scores(collection, question, k=k)
    if debug:
        print("=== Retriever Debug Output ===")
        print(f"Question: {question}")
        for index, item in enumerate(results, start=1):
            preview = item["content"][:240].replace("\n", " ").strip()
            print(f"Rank {index}")
            print(f"  source: {item['source']}")
            print(f"  page: {item['page']}")
            print(f"  score: {item['score']:.4f}")
            print(f"  preview: {preview}...")
            print()

    if not is_retrieval_confident(results):
        return {"answer": SAFE_FALLBACK_ANSWER, "sources": []}

    answer = generate_answer(question, results)
    return {"answer": answer, "sources": build_source_references(results)}


def run_retriever_example() -> None:
    """Run sample queries against the persisted vector store."""
    sample_questions = [
        "How many annual leaves do I have?",
        "Can unused leaves be carried forward?",
        "What is the hotel reimbursement limit for domestic travel?",
        "Can I work from home more than three days a week?",
    ]

    for question in sample_questions:
        result = ask_policy_question(question, k=3, debug=True)
        print("=== Generated Answer ===")
        print(result["answer"])
        if result["sources"]:
            print("Sources:", result["sources"])
        print()


if __name__ == "__main__":
    run_retriever_example()
