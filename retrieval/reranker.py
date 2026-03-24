from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# Load reranker model once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_results(query: str, 
                   dense_results: List[Dict[str, Any]], 
                   sparse_results: List[Dict[str, Any]], 
                   top_k: int = 5) -> List[Dict[str, Any]]:
    """Combine dense + sparse results and rerank using CrossEncoder."""
    
    # Merge and deduplicate results
    seen_texts = set()
    combined = []
    
    for result in dense_results + sparse_results:
        text = result["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            combined.append(result)
    
    if not combined:
        return []
    
    # Rerank using CrossEncoder
    pairs = [[query, result["text"]] for result in combined]
    scores = reranker.predict(pairs)
    
    # Attach reranker scores
    for i, result in enumerate(combined):
        result["rerank_score"] = float(scores[i])
    
    # Sort by reranker score
    reranked = sorted(combined, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked[:top_k]