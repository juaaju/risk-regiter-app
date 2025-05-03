from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Date, Table, MetaData, ForeignKey, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import date, datetime
import lmstudio as lms
import json
import os
import re
from dotenv import load_dotenv
import re
import random

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:toor@localhost:5432/risk_management")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Initialize LLM model
model = lms.llm("qwen2.5-7b-instruct-1m")

# FastAPI app
app = FastAPI(title="L2SA API", description="API for Level 2 Safety Analysis")

# CORS middleware - Allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, bisa diganti dengan specific origins di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TeamMemberCreate(BaseModel):
    name: str
    role: str

class L2SACreate(BaseModel):
    ptwType: str
    jobDescription: str
    location: str
    facility: str
    plannedDate: str
    teamMembers: List[TeamMemberCreate]

class TeamMemberResponse(BaseModel):
    id: str
    name: str
    role: str

class L2SAResponse(BaseModel):
    id: str
    displayId: str
    ptwType: str
    jobDescription: str
    location: str
    facility: str
    plannedDate: date
    status: str
    analysisResult: Optional[dict] = None
    teamMembers: List[TeamMemberResponse]
    createdAt: datetime

# THA item structure
class THAItem(BaseModel):
    id: str
    kategori: str
    item_checklist: str
    status: bool
    tanggal_update: datetime

# Database models
metadata = MetaData()

l2sa_table = Table(
    "l2sa",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("display_id", String, unique=True, nullable=False),
    Column("ptw_type", String, nullable=False),
    Column("job_description", String, nullable=False),
    Column("location", String, nullable=False),
    Column("facility", String, nullable=False),
    Column("planned_date", Date, nullable=False),
    Column("status", String, default="pending"),
    Column("analysis_result", JSONB, nullable=True),
    Column("created_at", TIMESTAMP(timezone=True), server_default=text("NOW()")),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=text("NOW()"))
)

l2sa_team_members = Table(
    "l2sa_team_members",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("l2sa_id", UUID(as_uuid=True), ForeignKey("l2sa.id", ondelete="CASCADE"), nullable=False),
    Column("name", String, nullable=False),
    Column("role", String, nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=text("NOW()"))
)

l2sa_tha_data = Table(
    "l2sa_tha_data",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("kategori", String, nullable=False),
    Column("item_checklist", String, nullable=False),
    Column("status", String, nullable=True),  # Will be updated by LLM
    Column("tanggal_update", TIMESTAMP(timezone=True), server_default=text("NOW()"))
)

# Create tables if they don't exist
metadata.create_all(engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper to generate L2SA display ID
import random
import string
from datetime import datetime
from sqlalchemy.sql import text

def generate_display_id(db: Session):
    """Completely rewritten function with extensive debugging"""
    try:
        # Buat ID dengan UUID yang dijamin unik
        year_code = datetime.now().strftime("%y")
        uuid_short = str(uuid.uuid4())[:6]
        display_id = f"L2SA{year_code}-{uuid_short}"
        
        print(f"GENERATED UUID-BASED ID: {display_id}")
        
        # Verifikasi ID tidak ada di database
        check_query = text("SELECT 1 FROM l2sa WHERE display_id = :id")
        exists = db.execute(check_query, {"id": display_id}).scalar()
        
        if exists:
            print(f"UNEXPECTED: UUID-based ID {display_id} already exists, generating another")
            display_id = f"L2SA{year_code}-{str(uuid.uuid4())[:8]}"
        
        print(f"FINAL ID TO BE USED: {display_id}")
        
        # Return sebagai string, bukan variabel
        return display_id
    except Exception as e:
        print(f"ERROR IN GENERATE_DISPLAY_ID: {e}")
        # Fallback yang sangat aman
        return f"L2SA{datetime.now().strftime('%y%m%d%H%M%S')}"
def extract_json_from_text(text):
    try:
        # If text is already a JSON object, return it
        if isinstance(text, dict):
            return text
            
        # Convert to string if it's not already
        if not isinstance(text, str):
            text = str(text)
            
        # Try to find JSON using regex
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
            
        # If no match found, try to parse the entire text as JSON
        return json.loads(text)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        print(f"Original text: {text}")
        # Return fallback JSON
        return {
            "analysis": {
                "categories": [],
                "summary": f"Failed to extract valid JSON: {str(e)[:100]}...",
                "risk_level": "unknown"
            }
        }

from schema_based_analyzer import analyze_with_schema_llm, analyze_with_lmstudio_fallback
# from schema_based_analyzer import analyze_with_schema_llm, analyze_with_lmstudio_fallback

def analyze_with_llm(job_description: str, ptw_type: str, db: Session):
    """
    Analisis keselamatan kerja menggunakan pendekatan schema terlebih dahulu,
    dengan fallback ke pendekatan tradisional jika diperlukan.
    """
    # Get checklist items from database
    checklist_items = db.execute(text("SELECT id, kategori, item_checklist FROM l2sa_tha_data")).fetchall()
    
    # Group items by category
    categories = {}
    for item in checklist_items:
        if item.kategori not in categories:
            categories[item.kategori] = []
        categories[item.kategori].append({
            "id": str(item.id),
            "item": item.item_checklist
        })
    
    try:
        # Coba gunakan pendekatan schema terlebih dahulu (lebih reliable)
        print("Using schema-based LLM analysis...")
        analysis_result = analyze_with_schema_llm(job_description, ptw_type, db)
        
        # Verifikasi hasil memiliki struktur yang diharapkan
        if (isinstance(analysis_result, dict) and 
            "analysis" in analysis_result and 
            "categories" in analysis_result["analysis"] and 
            len(analysis_result["analysis"]["categories"]) > 0):
            
            print("Schema-based analysis successful")
            return analysis_result
        
        # Jika pendekatan schema gagal, gunakan pendekatan lama dengan parser robust
        print("Schema-based analysis failed or empty, falling back to traditional approach...")
        
        # Gunakan fallback yang robust
        return analyze_with_lmstudio_fallback(job_description, ptw_type, db)
    
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback analysis jika error
        return {
            "analysis": {
                "categories": [],
                "summary": f"Error in hazard analysis: {str(e)}",
                "risk_level": "unknown"
            },
            "error": str(e)
        }
# API endpoints
@app.post("/api/l2sa/create", response_model=L2SAResponse)
def create_l2sa(l2sa_data: L2SACreate, db: Session = Depends(get_db)):
    try:
        # Generate display ID
        display_id = generate_display_id(db)
        
        # Create L2SA record
        l2sa_id = uuid.uuid4()
        planned_date = datetime.strptime(l2sa_data.plannedDate, "%Y-%m-%d").date()
        
        db.execute(
            l2sa_table.insert().values(
                id=l2sa_id,
                display_id=display_id,
                ptw_type=l2sa_data.ptwType,
                job_description=l2sa_data.jobDescription,
                location=l2sa_data.location,
                facility=l2sa_data.facility,
                planned_date=planned_date
            )
        )
        
        # Create team members
        for member in l2sa_data.teamMembers:
            db.execute(
                l2sa_team_members.insert().values(
                    id=uuid.uuid4(),
                    l2sa_id=l2sa_id,
                    name=member.name,
                    role=member.role
                )
            )
        
        # Run LLM analysis
        analysis_result = analyze_with_llm(l2sa_data.jobDescription, l2sa_data.ptwType, db)
        
        # Update analysis_result in database
        db.execute(
            l2sa_table.update()
            .where(l2sa_table.c.id == l2sa_id)
            .values(analysis_result=analysis_result)
        )
        
        # Commit transaction
        db.commit()
        
        # Retrieve complete record for response
        result = db.execute(
            text("""
                SELECT 
                    l.id, l.display_id, l.ptw_type, l.job_description, l.location, 
                    l.facility, l.planned_date, l.status, l.analysis_result, l.created_at
                FROM l2sa l
                WHERE l.id = :l2sa_id
            """),
            {"l2sa_id": l2sa_id}
        ).fetchone()
        
        # Get team members
        team_members = db.execute(
            text("""
                SELECT id, name, role
                FROM l2sa_team_members
                WHERE l2sa_id = :l2sa_id
            """),
            {"l2sa_id": l2sa_id}
        ).fetchall()
        
        team_members_response = [
            TeamMemberResponse(
                id=str(member.id),
                name=member.name,
                role=member.role
            ) for member in team_members
        ]
        
        # Prepare response
        response = L2SAResponse(
            id=str(result.id),
            displayId=result.display_id,
            ptwType=result.ptw_type,
            jobDescription=result.job_description,
            location=result.location,
            facility=result.facility,
            plannedDate=result.planned_date,
            status=result.status,
            analysisResult=result.analysis_result,
            teamMembers=team_members_response,
            createdAt=result.created_at
        )
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create L2SA: {str(e)}")


@app.get("/api/l2sa/{display_id}", response_model=L2SAResponse)
def get_l2sa(display_id: str, db: Session = Depends(get_db)):
    # Get L2SA record
    result = db.execute(
        text("""
            SELECT 
                l.id, l.display_id, l.ptw_type, l.job_description, l.location, 
                l.facility, l.planned_date, l.status, l.analysis_result, l.created_at
            FROM l2sa l
            WHERE l.display_id = :display_id
        """),
        {"display_id": display_id}
    ).fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail=f"L2SA with display ID {display_id} not found")
    
    # Get team members
    team_members = db.execute(
        text("""
            SELECT id, name, role
            FROM l2sa_team_members
            WHERE l2sa_id = :l2sa_id
        """),
        {"l2sa_id": result.id}
    ).fetchall()
    
    team_members_response = [
        TeamMemberResponse(
            id=str(member.id),
            name=member.name,
            role=member.role
        ) for member in team_members
    ]
    
    # Prepare response
    response = L2SAResponse(
        id=str(result.id),
        displayId=result.display_id,
        ptwType=result.ptw_type,
        jobDescription=result.job_description,
        location=result.location,
        facility=result.facility,
        plannedDate=result.planned_date,
        status=result.status,
        analysisResult=result.analysis_result,
        teamMembers=team_members_response,
        createdAt=result.created_at
    )
    
    return response

@app.get("/api/l2sa/list", response_model=List[L2SAResponse])
def list_l2sa(db: Session = Depends(get_db)):
    # Get all L2SA records
    results = db.execute(
        text("""
            SELECT 
                l.id, l.display_id, l.ptw_type, l.job_description, l.location, 
                l.facility, l.planned_date, l.status, l.analysis_result, l.created_at
            FROM l2sa l
            ORDER BY l.created_at DESC
        """)
    ).fetchall()
    
    l2sa_list = []
    
    for result in results:
        # Get team members for this L2SA
        team_members = db.execute(
            text("""
                SELECT id, name, role
                FROM l2sa_team_members
                WHERE l2sa_id = :l2sa_id
            """),
            {"l2sa_id": result.id}
        ).fetchall()
        
        team_members_response = [
            TeamMemberResponse(
                id=str(member.id),
                name=member.name,
                role=member.role
            ) for member in team_members
        ]
        
        # Add to response list
        l2sa_list.append(
            L2SAResponse(
                id=str(result.id),
                displayId=result.display_id,
                ptwType=result.ptw_type,
                jobDescription=result.job_description,
                location=result.location,
                facility=result.facility,
                plannedDate=result.planned_date,
                status=result.status,
                analysisResult=result.analysis_result,
                teamMembers=team_members_response,
                createdAt=result.created_at
            )
        )
    
    return l2sa_list

# Simple health check endpoint
@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Test endpoint for LLM
@app.post("/api/test/llm")
def test_llm(request: dict):
    """Test endpoint to check LLM functionality directly"""
    try:
        prompt = request.get("prompt", "What is the meaning of life?")
        result = model.respond(prompt)
        
        # Return details about the result
        return {
            "success": True,
            "result_type": str(type(result)),
            "result_dir": dir(result),
            "result_str": str(result),
            "has_find": hasattr(result, "find"),
            "has_text": hasattr(result, "text"),
            "has_content": hasattr(result, "content")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": str(type(e))
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)