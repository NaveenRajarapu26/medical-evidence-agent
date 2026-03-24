import os
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from docx import Document


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text and metadata from a PDF file."""
    documents = []
    pdf = fitz.open(file_path)
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()
        
        if text:  # skip empty pages
            documents.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_num + 1,
                    "total_pages": len(pdf),
                    "file_type": "pdf"
                }
            })
    
    pdf.close()
    return documents


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text and metadata from a DOCX file."""
    documents = []
    doc = Document(file_path)
    
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    
    if full_text:
        documents.append({
            "text": "\n".join(full_text),
            "metadata": {
                "source": Path(file_path).name,
                "page": 1,
                "total_pages": 1,
                "file_type": "docx"
            }
        })
    
    return documents


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Load a document based on its file extension."""
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Load all PDF and DOCX files from a folder."""
    all_documents = []
    folder = Path(folder_path)
    
    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() in [".pdf", ".docx"]:
            print(f"Loading: {file_path.name}")
            docs = load_document(str(file_path))
            all_documents.extend(docs)
    
    print(f"Total pages/sections loaded: {len(all_documents)}")
    return all_documents