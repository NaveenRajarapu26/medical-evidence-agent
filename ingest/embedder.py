from typing import List, Dict, Any

# Lazy load model — only load when first needed
_model = None

def get_model():
    """Load model only when first needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded!")
    return _model


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for each chunk."""
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    print("Embeddings generated successfully!")
    return chunks


def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    model = get_model()
    embedding = model.encode(query)
    return embedding.tolist()