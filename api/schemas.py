from pydantic import BaseModel
from typing import List, Optional


class Citation(BaseModel):
    source: str
    page: int
    passage: str
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    filter_by_doc: Optional[str] = None


class MedicalAnswer(BaseModel):
    answer: str
    citations: List[Citation]
    safety_passed: bool
    query: str


class IngestRequest(BaseModel):
    collection_name: Optional[str] = "medical_docs"


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str