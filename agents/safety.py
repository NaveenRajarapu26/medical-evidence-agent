from typing import Dict, Any


UNSAFE_PATTERNS = [
    "how to overdose",
    "lethal dose",
    "how to poison",
    "suicide method",
    "self harm instructions",
    "how to kill"
]

NON_MEDICAL_PATTERNS = [
    "write code",
    "hack ",
    "password",
    "credit card",
    "political opinion"
]


def check_query_safety(query: str) -> Dict[str, Any]:
    """Check if a query is safe to process."""
    query_lower = query.lower()
    
    for pattern in UNSAFE_PATTERNS:
        if pattern in query_lower:
            return {
                "safe": False,
                "reason": "unsafe_content",
                "message": "This query contains potentially harmful content and cannot be processed."
            }
    
    for pattern in NON_MEDICAL_PATTERNS:
        if pattern in query_lower:
            return {
                "safe": False,
                "reason": "off_topic",
                "message": "This system only answers medical questions based on uploaded documents."
            }
    
    return {"safe": True, "reason": None, "message": None}


def check_answer_safety(answer: str, citations: list) -> Dict[str, Any]:
    """Check if generated answer is safe to return."""
    
    if not citations:
        return {
            "safe": False,
            "reason": "no_citations",
            "message": "Answer could not be grounded in source documents."
        }
    
    if len(answer) < 20:
        return {
            "safe": False,
            "reason": "insufficient_answer",
            "message": "Could not generate a sufficient answer from the documents."
        }
    
    return {"safe": True, "reason": None, "message": None}