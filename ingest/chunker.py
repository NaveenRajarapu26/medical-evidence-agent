from typing import List, Dict, Any


from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: List[Dict[str, Any]], 
                    chunk_size: int = 512, 
                    chunk_overlap: int = 64) -> List[Dict[str, Any]]:
    """Split documents into chunks while preserving metadata."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        split_texts = splitter.split_text(text)
        
        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(split_texts)
                }
            })
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks