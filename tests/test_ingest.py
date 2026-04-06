from ingest.chunker import chunk_documents


def test_chunk_documents():
    docs = [{"text": "This is a test medical document about hypertension treatment.", 
              "metadata": {"source": "test.pdf", "page": 1, "file_type": "pdf"}}]
    chunks = chunk_documents(docs)
    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    print("test_chunk_documents PASSED")


def test_chunk_metadata_preserved():
    docs = [{"text": "Test content " * 50,
              "metadata": {"source": "test.pdf", "page": 2, "file_type": "pdf"}}]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test.pdf"
    print("test_chunk_metadata_preserved PASSED")