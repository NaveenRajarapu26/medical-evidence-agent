from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Load model once at module level (avoids reloading on every call)
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for each chunk."""
    
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    print("Embeddings generated successfully!")
    return chunks


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    embedding = model.encode(query)
    return embedding.tolist()