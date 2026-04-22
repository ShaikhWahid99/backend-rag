from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from rag_engine import MultimodalRAG
from llm import GeminiProvider, OpenAIProvider, LocalLLMProvider
from db.database import engine, Base
from routes import auth_routes, file_routes, rag_routes


Base.metadata.create_all(bind=engine)

LLM_PROVIDER_TYPE = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_PROVIDER_TYPE == "openai":
    primary_llm = OpenAIProvider(api_key=OPENAI_API_KEY)
elif LLM_PROVIDER_TYPE == "local":
    primary_llm = LocalLLMProvider()
else:
    primary_llm = GeminiProvider(api_key=GEMINI_API_KEY)

# Automatic fallback to local if primary is not local
# fallback_llm = LocalLLMProvider() if LLM_PROVIDER_TYPE != "local" else None
fallback_llm = LocalLLMProvider(text_model="llama3.2:1b", vision_model="moondream") if LLM_PROVIDER_TYPE != "local" else None

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

rag = MultimodalRAG(llm_provider=primary_llm, fallback_provider=fallback_llm)

app.include_router(auth_routes.router)
app.include_router(file_routes.router)
app.include_router(rag_routes.router)