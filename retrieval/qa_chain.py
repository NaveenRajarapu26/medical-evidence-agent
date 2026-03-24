import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

SYSTEM_PROMPT = """You are a medical evidence assistant. 
Answer questions ONLY using the provided context passages.
Always cite the source document and page number for every claim.
If the context does not contain enough information, say: 
'I cannot answer this based on the provided medical documents.'
Never make up information or answer from general knowledge."""


def generate_answer(query: str, 
                    retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate answer with citations from retrieved chunks."""
    
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in the documents.",
            "citations": [],
            "query": query
        }
    
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "?")
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{chunk['text']}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Build messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]
    
    response = llm.invoke(messages)
    
    # Build citations list
    citations = []
    for chunk in retrieved_chunks:
        citations.append({
            "source": chunk["metadata"].get("source", "Unknown"),
            "page": chunk["metadata"].get("page", "?"),
            "passage": chunk["text"][:200] + "...",
            "score": chunk.get("rerank_score", chunk.get("score", 0))
        })
    
    return {
        "answer": response.content,
        "citations": citations,
        "query": query
    }