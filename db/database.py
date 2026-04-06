import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# In-memory database (replace with PostgreSQL in production)
_db = {
    "users": {},
    "queries": [],
    "documents": []
}


def save_query(
    username: str,
    question: str,
    answer: str,
    citations: list,
    latency: float,
    safety_passed: bool
) -> Dict[str, Any]:
    """Save a query to the database."""
    entry = {
        "id": len(_db["queries"]) + 1,
        "username": username,
        "question": question,
        "answer": answer[:500],
        "citation_count": len(citations),
        "latency_seconds": round(latency, 3),
        "safety_passed": safety_passed,
        "timestamp": datetime.utcnow().isoformat()
    }
    _db["queries"].append(entry)
    return entry


def get_user_queries(username: str) -> List[Dict[str, Any]]:
    """Get all queries for a user."""
    return [q for q in _db["queries"] if q["username"] == username]


def save_document(filename: str, chunks_count: int, username: str) -> Dict[str, Any]:
    """Save document metadata."""
    entry = {
        "id": len(_db["documents"]) + 1,
        "filename": filename,
        "chunks_count": chunks_count,
        "uploaded_by": username,
        "timestamp": datetime.utcnow().isoformat()
    }
    _db["documents"].append(entry)
    return entry


def get_all_documents() -> List[Dict[str, Any]]:
    """Get all ingested documents."""
    return _db["documents"]


def get_stats() -> Dict[str, Any]:
    """Get database statistics."""
    return {
        "total_queries": len(_db["queries"]),
        "total_documents": len(_db["documents"]),
        "total_users": len(_db["users"])
    }