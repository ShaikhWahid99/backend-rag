from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from rag_engine import MultimodalRAG
from db.database import engine, Base
from routes import auth_routes, file_routes, rag_routes


# Initialize Database tables
Base.metadata.create_all(bind=engine)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

if not os.path.exists("uploads"):
    os.makedirs("uploads")
    
if not os.path.exists("images"):
    os.makedirs("images")

app.mount("/images", StaticFiles(directory="images"), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MultimodalRAG(api_key=GEMINI_API_KEY)

app.include_router(auth_routes.router)
app.include_router(file_routes.router)
app.include_router(rag_routes.router)