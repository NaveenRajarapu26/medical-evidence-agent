import streamlit as st
import os
from ingest.loader import load_document
from ingest.chunker import chunk_documents
from ingest.embedder import generate_embeddings, embed_query
from retrieval.vectorstore import add_chunks_to_vectorstore, query_vectorstore
from retrieval.bm25 import BM25Retriever
from retrieval.reranker import rerank_results
from retrieval.qa_chain import generate_answer

st.set_page_config(page_title="Medical Evidence Agent", page_icon="🏥", layout="wide")
st.title("🏥 Medical Evidence Agent")
st.caption("Ask questions grounded in your medical documents")

# Initialize BM25 in session state
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = BM25Retriever()
if "chunks_loaded" not in st.session_state:
    st.session_state.chunks_loaded = False

# Sidebar - Document Upload
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            all_chunks = []
            os.makedirs("temp_uploads", exist_ok=True)

            for uploaded_file in uploaded_files:
                temp_path = f"temp_uploads/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                docs = load_document(temp_path)
                chunks = chunk_documents(docs)
                all_chunks.extend(chunks)

            # Generate embeddings
            st.info("Generating embeddings...")
            embedded_chunks = generate_embeddings(all_chunks)

            # Store in ChromaDB
            st.info("Storing in vector database...")
            add_chunks_to_vectorstore(embedded_chunks)

            # Build BM25 index
            st.session_state.bm25_retriever.build_index(all_chunks)
            st.session_state.chunks_loaded = True
            st.session_state.all_chunks = all_chunks

            st.success(f"✅ Processed {len(all_chunks)} chunks from {len(uploaded_files)} files!")

# Main - Question Answering
st.header("💬 Ask a Medical Question")

query = st.text_input("Enter your question:", 
                       placeholder="What are the first-line treatments for hypertension?")

if st.button("Get Answer") and query:
    if not st.session_state.chunks_loaded:
        st.warning("⚠️ Please upload and process documents first!")
    else:
        with st.spinner("Retrieving evidence and generating answer..."):
            # Dense retrieval
            query_emb = embed_query(query)
            dense_results = query_vectorstore(query_emb, top_k=5)

            # Sparse retrieval
            sparse_results = st.session_state.bm25_retriever.retrieve(query, top_k=5)

            # Rerank
            final_chunks = rerank_results(query, dense_results, sparse_results, top_k=3)

            # Generate answer
            result = generate_answer(query, final_chunks)

        # Display answer
        st.subheader("📋 Answer")
        st.write(result["answer"])

        # Display citations
        st.subheader("📚 Citations")
        for i, citation in enumerate(result["citations"]):
            with st.expander(f"Source {i+1}: {citation['source']} - Page {citation['page']}"):
                st.write(citation["passage"])
                st.caption(f"Relevance score: {citation['score']:.3f}")