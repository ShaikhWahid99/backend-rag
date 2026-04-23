import os
import time
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from auth.auth import get_current_user
from db.database import get_db
from db.models import User, File as DBFile

router = APIRouter(tags=["files"])

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    path = f"uploads/{current_user.id}_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_file = DBFile(user_id=current_user.id, filename=file.filename, filepath=path)
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    from main import rag
    rag.process_pdf(path, current_user.id, db_file.id)

    return {"message": "PDF processed successfully", "filename": file.filename, "id": db_file.id}

@router.get("/files")
async def list_files(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    files = db.query(DBFile).filter(DBFile.user_id == current_user.id).all()
    result = []
    for f in files:
        if os.path.exists(f.filepath):
            stats = os.stat(f.filepath)
            result.append({
                "id": f.id,
                "filename": f.filename,
                "file_type": "pdf" if f.filename.lower().endswith(".pdf") else "text",
                "status": "indexed",
                "file_size": stats.st_size,
                "created_at": f.created_at
            })
    return result

@router.delete("/files/{file_id}")
async def delete_file(file_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_file = db.query(DBFile).filter(DBFile.id == file_id, DBFile.user_id == current_user.id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
        
    if os.path.exists(db_file.filepath):
        os.remove(db_file.filepath)
        
    db.delete(db_file)
    db.commit()
    
    return {"message": f"File {db_file.filename} deleted successfully"}
