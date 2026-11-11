
from fastapi import FastAPI
from pydantic import BaseModel
from app import get_answer

app = FastAPI(title="Cipla RAG API")

class AskRequest(BaseModel):
    question: str
    role: str = "patient"

@app.get("/")
def home():
    return {"status": "Cipla RAG API is running"}

@app.post("/ask")
def ask(req: AskRequest):
    answer, sources, retrieved = get_answer(req.question, req.role)
    return {
        "answer": answer,
        "sources": sources,
        "top_score": retrieved[0]["score"] if retrieved else 0
    }
