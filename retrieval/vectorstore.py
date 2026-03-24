import os
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(collection_name: str = "medical_docs"):
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks_to_vectorstore(chunks: List[Dict[str, Any]], 
                               collection_name: str = "medical_docs"):
    collection = get_or_create_collection(collection_name)
    
    ids, embeddings, documents, metadatas = [], [], [], []
    
    for chunk in chunks:
        chunk_id = f"{chunk['metadata']['source']}_p{chunk['metadata']['page']}_c{chunk['metadata']['chunk_index']}"
        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
    
    collection.add(ids=ids, embeddings=embeddings, 
                   documents=documents, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks to ChromaDB")


def query_vectorstore(query_embedding: List[float], 
                      top_k: int = 5,
                      collection_name: str = "medical_docs") -> List[Dict[str, Any]]:
    collection = get_or_create_collection(collection_name)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]
        })
    
    return hits


def delete_collection(collection_name: str = "medical_docs"):
    client.delete_collection(collection_name)
    print(f"Deleted collection: {collection_name}")