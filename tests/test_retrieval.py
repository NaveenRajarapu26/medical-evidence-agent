from retrieval.bm25 import BM25Retriever


def test_bm25_build_and_retrieve():
    retriever = BM25Retriever()
    chunks = [
        {"text": "Hypertension treatment includes ACE inhibitors", 
         "metadata": {"source": "test.pdf", "page": 1}},
        {"text": "Diabetes management requires insulin monitoring",
         "metadata": {"source": "test.pdf", "page": 2}},
        {"text": "Cardiac arrest requires immediate CPR",
         "metadata": {"source": "test.pdf", "page": 3}}
    ]
    retriever.build_index(chunks)
    results = retriever.retrieve("hypertension treatment", top_k=2)
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    print("test_bm25_build_and_retrieve PASSED")