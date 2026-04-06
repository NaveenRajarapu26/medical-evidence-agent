import os
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from api.schemas import QueryRequest, MedicalAnswer, UserCreate, UserLogin, Token
from api.auth import hash_password, verify_password, create_access_token, verify_token
from agents.graph import run_agent
from agents.safety import check_query_safety
from ingest.loader import load_document
from ingest.chunker import chunk_documents
from ingest.embedder import generate_embeddings
from retrieval.vectorstore import add_chunks_to_vectorstore

app = FastAPI(
    title="Medical Evidence Agent API",
    description="Production-grade medical Q&A with RAG and multi-agent reasoning",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

security = HTTPBearer()

# Simple in-memory user store (replace with PostgreSQL in production)
users_db = {}
query_history = []


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username


@app.get("/")
def root():
    return {
        "message": "Medical Evidence Agent API",
        "version": "2.0.0",
        "status": "running"
    }


@app.post("/auth/register")
def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    users_db[user.username] = hash_password(user.password)
    return {"message": "User registered successfully"}


@app.post("/auth/login", response_model=Token)
def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(user.password, users_db[user.username]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/query", response_model=MedicalAnswer)
async def query(request: QueryRequest, username: str = Depends(get_current_user)):
    # Safety check
    safety_check = check_query_safety(request.question)
    if not safety_check["safe"]:
        raise HTTPException(status_code=400, detail=safety_check["message"])

    # Run agent
    result = run_agent(request.question)

    # Log to history
    query_history.append({
        "username": username,
        "question": request.question,
        "answer": result.get("answer", ""),
        "safety_passed": result.get("safety_passed", False)
    })

    return MedicalAnswer(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        safety_passed=result.get("safety_passed", False),
        query=request.question
    )


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = f"temp_uploads/{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    docs = load_document(temp_path)
    chunks = chunk_documents(docs)
    embedded = generate_embeddings(chunks)
    add_chunks_to_vectorstore(embedded)

    return {
        "message": f"Successfully ingested {file.filename}",
        "chunks_created": len(chunks)
    }


@app.get("/history")
def get_history(username: str = Depends(get_current_user)):
    user_history = [q for q in query_history if q["username"] == username]
    return {"history": user_history}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}