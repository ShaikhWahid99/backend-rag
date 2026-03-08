from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
from rag_engine import MultimodalRAG

GEMINI_API_KEY = "AIzaSyDfZK40EX0jN1JctVRUBEhrjGrXDKDR90Q"

app = FastAPI()

rag = MultimodalRAG(api_key=GEMINI_API_KEY)


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    
    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag.process_pdf(path)

    return {"message": "PDF processed successfully"}


@app.post("/ask")
async def ask_question(q: Question):

    answer = rag.ask(q.question)

    return {
        "question": q.question,
        "answer": answer
    }