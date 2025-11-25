from fastapi import FastAPI
from pydantic import BaseModel
from rag import save_document, query_similar, generate_answer

app = FastAPI()

class MemoryRequest(BaseModel):
    id: str
    text: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "travel-log-ai running"}


@app.post("/memory")
def store_memory(data: MemoryRequest):
    save_document(data.id, data.text)
    return {"status": "stored"}

@app.post("/ask")
def ask_question(data: QuestionRequest):
    chunks = query_similar(data.question)
    answer = generate_answer(data.question, chunks)
    return {"answer": answer}
