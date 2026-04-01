# 🏥 Medical Evidence Agent

> Production-grade medical Q&A system with hybrid RAG, citation grounding, and evaluation framework.

🔗 **[Live Demo](https://medical-evidence-agent.onrender.com)** | 📂 **[GitHub](https://github.com/NaveenRajarapu26/medical-evidence-agent)**

---

## What It Does

Upload medical textbooks, clinical guidelines, or research papers and ask complex questions. The system retrieves grounded evidence and answers with citations — never hallucinating beyond the source material.

## Architecture
```
PDF/DOCX Upload → Semantic Chunking → Embeddings
                                          ↓
User Query → BM25 Sparse Retrieval ──→ Hybrid Merge → CrossEncoder Reranking → LLM Answer + Citations
           → Dense Vector Retrieval ──↗
```

## Phase 1 — Medical RAG Assistant ✅

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Document Ingestion | PyMuPDF + python-docx | PDF/DOCX parsing with metadata |
| Semantic Chunking | LangChain text splitters | 512-token chunks with overlap |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, no API needed |
| Vector Store | ChromaDB | Persistent vector storage |
| Sparse Retrieval | BM25 (rank-bm25) | Keyword-based retrieval |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 | Hybrid result reranking |
| LLM | Groq (llama-3.3-70b-versatile) | Fast inference, citation-grounded answers |
| Evaluation | RAGAS | Faithfulness, relevancy, context recall |
| UI | Streamlit | Document upload + Q&A interface |
| Deployment | Render | Live production deployment |

## Phase 2 — Production Agent (In Progress) 🚧

- LangGraph multi-agent workflow (Retrieval → Reasoning → Safety agents)
- Guardrails AI safety layer
- FastAPI async backend with JWT auth
- PostgreSQL query history
- LangSmith observability
- Docker + GitHub Actions CI/CD
- AWS S3 document storage

## Tech Stack

Python · LangChain · ChromaDB · sentence-transformers · Groq API · RAGAS · Streamlit · Docker · Render

## Run Locally
```bash
git clone https://github.com/NaveenRajarapu26/medical-evidence-agent.git
cd medical-evidence-agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY
streamlit run app.py
```

## Project Structure
```
medical-evidence-agent/
├── ingest/          # Document processing pipeline
├── retrieval/       # Hybrid RAG retrieval layer  
├── agents/          # Phase 2 - LangGraph agents
├── evaluation/      # RAGAS evaluation framework
├── api/             # Phase 2 - FastAPI backend
├── app.py           # Streamlit UI
└── README.md
```

## Author

**Naveen Rajarapu** — Applied AI Engineer