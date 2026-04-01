from typing import List, Dict, Any

# Lazy load reranker
_reranker = None

def get_reranker():
    """Load reranker only when first needed."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker loaded!")
    return _reranker


def rerank_results(query: str,
                   dense_results: List[Dict[str, Any]],
                   sparse_results: List[Dict[str, Any]],
                   top_k: int = 5) -> List[Dict[str, Any]]:
    """Combine dense + sparse results and rerank using CrossEncoder."""
    
    seen_texts = set()
    combined = []
    
    for result in dense_results + sparse_results:
        text = result["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            combined.append(result)
    
    if not combined:
        return []
    
    reranker = get_reranker()
    pairs = [[query, result["text"]] for result in combined]
    scores = reranker.predict(pairs)
    
    for i, result in enumerate(combined):
        result["rerank_score"] = float(scores[i])
    
    reranked = sorted(combined, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked[:top_k]