from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import time

from rag_engine import MultimodalRAG
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MultimodalRAG(api_key=GEMINI_API_KEY)


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    
    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag.process_pdf(path)

    return {"message": "PDF processed successfully", "filename": file.filename}


@app.get("/files")
async def list_files():
    files = []
    if os.path.exists("uploads"):
        for filename in os.listdir("uploads"):
            path = os.path.join("uploads", filename)
            if os.path.isfile(path):
                stats = os.stat(path)
                files.append({
                    "id": filename,
                    "filename": filename,
                    "file_type": "pdf" if filename.lower().endswith(".pdf") else "text",
                    "status": "indexed",
                    "file_size": stats.st_size,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_ctime))
                })
    return files


@app.post("/ask")
async def ask_question(q: Question):

    answer = rag.ask(q.question)

    return {
        "question": q.question,
        "answer": answer
    }