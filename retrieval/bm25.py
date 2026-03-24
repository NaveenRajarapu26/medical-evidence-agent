from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built with {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using BM25."""
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(scores[idx])
            })

        return results