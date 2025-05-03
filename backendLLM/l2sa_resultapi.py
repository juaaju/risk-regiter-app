from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Union
import os
from datetime import datetime
import uuid
import json
from openai import OpenAI
# Di main.py FastAPI Anda:
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Text, Boolean, DateTime, Table, MetaData

metadata = MetaData()
# Database configuration

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:toor@localhost/risk_management")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class L2saThAChecklist(Base):
    __tablename__ = "l2sa_tha_checklist"
    
    id = Column(UUID, primary_key=True)
    l2sa_id = Column(String)
    hazard_id = Column(String)
    category = Column(String)
    hazard_text = Column(Text)
    control_text = Column(Text)
    is_custom = Column(Boolean)
    created_at = Column(DateTime(timezone=True))
    created_by = Column(String)

# Import skema analisis yang diperbarui
from pydantic import BaseModel, Field

# Skema model Pydantic untuk respons
class ChecklistItem(BaseModel):
    id: Union[str, uuid.UUID]  # Menerima string atau UUID
    l2sa_id: str
    hazard_id: str
    category: str
    hazard_text: str
    control_text: str
    is_custom: bool
    created_at: datetime
    created_by: Optional[str] = None  # Mengizinkan nilai None
    
    class Config:
        orm_mode = True
        from_attributes = True  # Baru di Pydantic v2, menggantikan orm_mode di versi masa depan

# Definisi skema hasil analisis yang diperbarui berdasarkan struktur tabel frontend
class RiskLevel(BaseModel):
    severity: str  # Keparahan (5a/9a)
    likelihood: str  # Kemungkinan (5b/9b)
    risk_level: str  # Tingkat Risiko (5c/9c)

class WorkStep(BaseModel):
    number: int  # No (1)
    description: str  # Deskripsi Langkah Pekerjaan (2)
    hazard: str  # Bahaya (3)
    impact: str  # Dampak (4)
    initial_risk: RiskLevel  # Risiko Awal (5)
    control_action: str  # Tindakan Pengendalian (6)
    executor: str  # Pelaksana (7)
    initials: Optional[str] = None  # Paraf (8)
    residual_risk: RiskLevel  # Risiko Sisa (9)
    verification: Optional[str] = None  # Verifikasi (10)

class L2SAAnalysisResponse(BaseModel):
    l2sa_id: str
    work_steps: List[WorkStep]
    recommendations: List[str]
    overall_risk_assessment: str
    timestamp: datetime = Field(default_factory=datetime.now)

l2sa_table = Table(
    "l2sa",
    metadata,
    Column("display_id", String, unique=True, nullable=False),
    Column("job_description", String, nullable=False),
)

# Skema definisi LLM
L2SA_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "l2sa_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "l2sa_id": {"type": "string"},
                "work_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer"},
                            "description": {"type": "string"},
                            "hazard": {"type": "string"},
                            "impact": {"type": "string"},
                            "initial_risk": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string", "enum": ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]},
                                    "likelihood": {"type": "string", "enum": ["Jarang", "Kadang-kadang", "Mungkin", "Sering", "Hampir Pasti"]},
                                    "risk_level": {"type": "string", "enum": ["Rendah", "Sedang", "Tinggi", "Ekstrem"]}
                                },
                                "required": ["severity", "likelihood", "risk_level"]
                            },
                            "control_action": {"type": "string"},
                            "executor": {"type": "string"},
                            "initials": {"type": "string"},
                            "residual_risk": {
                                "type": "object",
                                "properties": {
                                    "severity": {"type": "string", "enum": ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]},
                                    "likelihood": {"type": "string", "enum": ["Jarang", "Kadang-kadang", "Mungkin", "Sering", "Hampir Pasti"]},
                                    "risk_level": {"type": "string", "enum": ["Rendah", "Sedang", "Tinggi", "Ekstrem"]}
                                },
                                "required": ["severity", "likelihood", "risk_level"]
                            },
                            "verification": {"type": "string"}
                        },
                        "required": ["number", "description", "hazard", "impact", "initial_risk", "control_action", "executor", "residual_risk"]
                    },
                    "minItems": 1
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "overall_risk_assessment": {"type": "string"}
            },
            "required": ["l2sa_id", "work_steps", "recommendations", "overall_risk_assessment"]
        }
    }
}

# FastAPI app
app = FastAPI(title="L2SA Analysis API")
# Setelah app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, di production sebaiknya spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize OpenAI client
def get_openai_client():
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "lm-studio")
    )
def get_job_description(db: Session, l2sa_id: str):
    """
    Mengambil job_description dari tabel l2sa berdasarkan l2sa_id
    """
    l2sa_data = db.query(l2sa_table).filter_by(display_id=l2sa_id).first()
    if not l2sa_data:
        # Jika data tidak ditemukan, berikan nilai default
        return "Tidak ada deskripsi pekerjaan yang tersedia"
    return l2sa_data.job_description
# Function to get checklist items
def get_checklist_items(id: str, db: Session):
    """
    Retrieve all checklist items associated with a specific L2SA ID
    """
    # Filter records based on the l2sa_id field to match the id parameter from the URL
    checklist_items = db.query(L2saThAChecklist).filter(L2saThAChecklist.l2sa_id == id).all()
    
    if not checklist_items:
        raise HTTPException(status_code=404, detail=f"No checklist items found for L2SA ID {id}")
    
    # Convert UUID objects to strings
    for item in checklist_items:
        if isinstance(item.id, uuid.UUID):
            item.id = str(item.id)
    
    return checklist_items

# Function to analyze checklist items using LLM
def analyze_with_llm(checklist_items: List[ChecklistItem], l2sa_id: str, db:Session):
    """
    Menganalisis item checklist menggunakan LLM dan mengembalikan analisis terstruktur
    """
    client = get_openai_client()
    job_description = get_job_description(db, l2sa_id)
    # Menyiapkan data item checklist untuk LLM
    checklist_data = [
        {
            "id": str(item.id) if hasattr(item.id, "hex") else item.id,
            "hazard_id": item.hazard_id,
            "category": item.category,
            "hazard_text": item.hazard_text,
            "control_text": item.control_text,
            "is_custom": item.is_custom
        }
        for item in checklist_items
    ]
    
    # Membuat prompt untuk LLM
    prompt = f"""
    Anda adalah seorang ahli penilaian keselamatan dan risiko. Analisis item checklist L2SA (Level 2 Safety Assessment) berikut untuk L2SA ID: {l2sa_id}:
    
    Job Description:
    {job_description}
    
    Checklist Items:
    {json.dumps(checklist_data, indent=2)}
    
    Berdasarkan data tersebut, buatkan analisis lengkap dengan format sebagai berikut:
    
    1. Buat daftar langkah kerja berdasarkan job description. Batasi minimal 1 langkah kerja, maksimal 20 langkah kerja.
    2. Untuk setiap langkah kerja, tentukan:
       - Nomor urut
       - Deskripsi langkah pekerjaan berdasarkan job description
       - Bahaya yang mungkin terjadi saat melakukan pekerjaan
       - Dampak dari bahaya tersebut
       - Risiko awal (keparahan, kemungkinan, dan tingkat risiko)
       - Tindakan pengendalian yang diperlukan berdasarkan item checklist
       - Pelaksana yang bertanggung jawab
       - Risiko sisa setelah pengendalian (keparahan, kemungkinan, dan tingkat risiko)
    
    3. Berikan rekomendasi tambahan untuk meningkatkan keselamatan
    4. Berikan penilaian risiko keseluruhan
    
    Gunakan skala berikut:
    - Keparahan: Rendah, Sedang, Tinggi, Sangat Tinggi
    - Kemungkinan: Jarang, Kadang-kadang, Mungkin, Sering, Hampir Pasti
    - Tingkat Risiko: Rendah, Sedang, Tinggi, Ekstrem
    
    Kembalikan analisis Anda dalam format yang terstruktur sesuai dengan skema yang disediakan.
    """
    
    # Mendapatkan respons dari LLM
    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "your-model"),
            messages=[
                {"role": "system", "content": "Anda adalah seorang ahli penilaian keselamatan dan risiko yang mengkhususkan diri dalam analisis L2SA."},
                {"role": "user", "content": prompt}
            ],
            response_format=L2SA_ANALYSIS_SCHEMA,
        )
        
        # Parsing respons
        analysis_result = json.loads(response.choices[0].message.content)
        
        # Memastikan l2sa_id diatur dengan benar
        analysis_result["l2sa_id"] = l2sa_id
        
        return analysis_result
        
    except Exception as e:
        # Log error (dalam lingkungan produksi, gunakan logger yang tepat)
        print(f"LLM Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing with LLM: {str(e)}")

# API endpoints
@app.get("/l2sa/{id}/analyze", response_model=List[ChecklistItem])
def get_checklist_endpoint(id: str, db: Session = Depends(get_db)):
    """
    Retrieve all checklist items associated with a specific L2SA ID
    """
    return get_checklist_items(id, db)

@app.get("/l2sa/{id}/analyze/result", response_model=L2SAAnalysisResponse)
def analyze_l2sa_endpoint(id: str, db: Session = Depends(get_db)):
    """
    Retrieve checklist items and analyze them with LLM
    """
    # Get checklist items
    checklist_items = get_checklist_items(id, db)
    
    # Pass to LLM for analysis
    analysis_result = analyze_with_llm(checklist_items, id, db)
    
    return analysis_result

# Tambahkan kode berikut ke main.py

# Import yang dibutuhkan (jika belum ada)
from fastapi import Body
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Column, String, DateTime, text

# Model database untuk menyimpan hasil analisis (jika belum ada)
class L2saAnalysisResult(Base):
    __tablename__ = "l2sa_analysis_results"
    
    id = Column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    l2sa_id = Column(String, nullable=False)  # Gunakan UUID jika tabel l2sa.id bertipe UUID
    analysis_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    created_by = Column(String)

# Fungsi untuk menyimpan hasil analisis ke database
def save_analysis_result(db: Session, l2sa_id: str, analysis_data: dict, created_by: str = None):
    """
    Menyimpan hasil analisis LLM ke database
    """
    # Hapus entri lama jika ada
    db.query(L2saAnalysisResult).filter(L2saAnalysisResult.l2sa_id == l2sa_id).delete()
    
    # Buat entri baru
    analysis_result = L2saAnalysisResult(
        l2sa_id=l2sa_id,
        analysis_data=analysis_data,
        created_by=created_by
    )
    
    # Simpan ke database
    db.add(analysis_result)
    db.commit()
    db.refresh(analysis_result)
    
    return analysis_result

# Fungsi untuk mendapatkan hasil analisis yang tersimpan
def get_saved_analysis(db: Session, l2sa_id: str):
    """
    Mengambil hasil analisis yang tersimpan dari database
    """
    analysis = db.query(L2saAnalysisResult).filter(
        L2saAnalysisResult.l2sa_id == l2sa_id
    ).first()
    
    return analysis

# Endpoint baru untuk menyimpan hasil analisis yang telah dibuat
@app.post("/l2sa/{id}/save-analysis", response_model=dict)
def save_analysis_endpoint(
    id: str, 
    analysis_data: dict = Body(...), 
    created_by: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Menyimpan hasil analisis yang telah dibuat ke database.
    Misalnya, hasil dari /l2sa/{id}/analyze/result
    """
    # Validasi bahwa l2sa_id di dalam data sesuai dengan id di path
    if "l2sa_id" in analysis_data and analysis_data["l2sa_id"] != id:
        raise HTTPException(
            status_code=400, 
            detail="l2sa_id dalam data tidak sesuai dengan ID di path"
        )
    
    # Pastikan l2sa_id ada dalam data
    if "l2sa_id" not in analysis_data:
        analysis_data["l2sa_id"] = id
    
    # Simpan ke database
    saved = save_analysis_result(db, id, analysis_data, created_by)
    
    return {
        "message": "Analysis saved successfully", 
        "analysis_id": str(saved.id)
    }

# Endpoint untuk mendapatkan hasil analisis yang tersimpan
@app.get("/l2sa/{id}/saved-analysis", response_model=L2SAAnalysisResponse)
def get_saved_analysis_endpoint(id: str, db: Session = Depends(get_db)):
    """
    Mengambil hasil analisis yang tersimpan dari database
    """
    analysis = get_saved_analysis(db, id)
    
    if not analysis:
        raise HTTPException(
            status_code=404, 
            detail=f"No saved analysis found for L2SA ID {id}"
        )
    
    return analysis.analysis_data

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)