from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from ingest.embedder import embed_query
from retrieval.vectorstore import query_vectorstore
from retrieval.bm25 import BM25Retriever
from retrieval.reranker import rerank_results
from retrieval.qa_chain import generate_answer

# Define the agent state
class AgentState(TypedDict):
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    safety_passed: bool
    final_output: Dict[str, Any]
    retry_count: int
    error: Optional[str]


def retrieval_agent(state: AgentState) -> AgentState:
    """Agent 1: Retrieves relevant chunks using hybrid search."""
    print(f"[Retrieval Agent] Processing query: {state['query']}")
    
    try:
        query_emb = embed_query(state["query"])
        dense_results = query_vectorstore(query_emb, top_k=5)
        
        # Check if results are relevant enough
        if dense_results and max(r.get("score", 0) for r in dense_results) < 0.3:
            # Low confidence - retry with retry count check
            if state["retry_count"] < 2:
                print("[Retrieval Agent] Low confidence, will retry...")
                return {
                    **state,
                    "chunks": dense_results,
                    "retry_count": state["retry_count"] + 1
                }
        
        final_chunks = dense_results[:3]
        print(f"[Retrieval Agent] Retrieved {len(final_chunks)} chunks")
        
        return {
            **state,
            "chunks": final_chunks,
            "retry_count": state["retry_count"]
        }
    
    except Exception as e:
        return {**state, "chunks": [], "error": str(e)}


def reasoning_agent(state: AgentState) -> AgentState:
    """Agent 2: Generates answer with citations from retrieved chunks."""
    print("[Reasoning Agent] Generating answer...")
    
    if not state["chunks"]:
        return {
            **state,
            "answer": "I could not find relevant information in the documents.",
            "citations": []
        }
    
    try:
        result = generate_answer(state["query"], state["chunks"])
        print("[Reasoning Agent] Answer generated successfully")
        
        return {
            **state,
            "answer": result["answer"],
            "citations": result["citations"]
        }
    
    except Exception as e:
        return {
            **state,
            "answer": "An error occurred while generating the answer.",
            "citations": [],
            "error": str(e)
        }


def safety_agent(state: AgentState) -> AgentState:
    """Agent 3: Validates answer safety and quality."""
    print("[Safety Agent] Checking answer safety...")
    
    answer = state["answer"].lower()
    query = state["query"].lower()
    
    # Check for unsafe patterns
    unsafe_keywords = [
        "suicide", "self-harm", "overdose instructions",
        "how to poison", "lethal dose to kill"
    ]
    
    for keyword in unsafe_keywords:
        if keyword in query:
            return {
                **state,
                "safety_passed": False,
                "final_output": {
                    "answer": "I cannot provide information on this topic as it may be harmful.",
                    "citations": [],
                    "safety_passed": False,
                    "query": state["query"]
                }
            }
    
    # Check answer has citations
    has_citations = len(state["citations"]) > 0
    
    # Check answer isn't too short
    is_substantive = len(state["answer"]) > 50
    
    safety_passed = has_citations and is_substantive
    
    print(f"[Safety Agent] Safety check: {'PASSED' if safety_passed else 'FAILED'}")
    
    return {
        **state,
        "safety_passed": safety_passed,
        "final_output": {
            "answer": state["answer"],
            "citations": state["citations"],
            "safety_passed": safety_passed,
            "query": state["query"]
        }
    }


def should_retry(state: AgentState) -> str:
    """Conditional edge: decide whether to retry retrieval."""
    if state["retry_count"] > 0 and len(state["chunks"]) > 0:
        return "reasoning"
    return "reasoning"


def build_medical_agent():
    """Build and compile the LangGraph agent workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("safety", safety_agent)
    
    # Add edges
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "reasoning")
    workflow.add_edge("reasoning", "safety")
    workflow.add_edge("safety", END)
    
    return workflow.compile()


def run_agent(query: str) -> Dict[str, Any]:
    """Run the medical agent pipeline."""
    agent = build_medical_agent()
    
    initial_state = {
        "query": query,
        "chunks": [],
        "answer": "",
        "citations": [],
        "safety_passed": False,
        "final_output": {},
        "retry_count": 0,
        "error": None
    }
    
    result = agent.invoke(initial_state)
    return result["final_output"]