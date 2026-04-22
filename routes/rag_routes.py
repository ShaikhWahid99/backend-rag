from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from auth.auth import get_current_user
from db.database import get_db
from db.models import User, QueryHistory

router = APIRouter(tags=["rag"])

class Question(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(
    q: Question, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from main import rag
    from db.models import File as DBFile
    
    latest_file = db.query(DBFile).filter(DBFile.user_id == current_user.id).order_by(DBFile.created_at.desc()).first()
    file_id = latest_file.id if latest_file else -1
    
    result = rag.ask(q.question, current_user.id, file_id)
    
    image_url = f"/{result['image_path']}" if result.get("image_path") else None
    
    query_history = QueryHistory(
        user_id=current_user.id,
        question=q.question,
        answer=result["answer"]
    )
    db.add(query_history)
    db.commit()

    return {
        "question": q.question,
        "answer": result["answer"],
        "image": image_url
    }
