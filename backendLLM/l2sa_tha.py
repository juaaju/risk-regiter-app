# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# Inisialisasi FastAPI
app = FastAPI(
    title="Checklist Keselamatan Kerja API",
    description="API sederhana untuk mengelola checklist keselamatan kerja",
    version="1.0.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain spesifik di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi database PostgreSQL
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "risk_management",
    "user": "postgres",
    "password": "toor"
}

# Konfigurasi SQLAlchemy
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fungsi untuk mendapatkan session SQLAlchemy
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fungsi untuk membuat koneksi database langsung dengan psycopg2
def get_db_connection():
    try:
        conn = psycopg2.connect(
            **DATABASE_CONFIG,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Model untuk respons
class ChecklistItem(BaseModel):
    id: int
    item_checklist: str
    status: bool = False

class KategoriWithItems(BaseModel):
    kategori: str
    items: List[ChecklistItem]

class SaveChecklistRequest(BaseModel):
    selectedHazards: List[str]
    customHazards: Optional[List[Dict]] = []

# Endpoint untuk mendapatkan semua kategori dengan item checklistnya
@app.get("/checklist", response_model=List[KategoriWithItems])
def get_all_categories_with_items():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Ambil semua data checklist dan kelompokkan berdasarkan kategori
        cursor.execute("""
            SELECT id, kategori, item_checklist, status
            FROM l2sa_tha_data
            ORDER BY kategori, id
        """)
        
        all_items = cursor.fetchall()
        
        # Kelompokkan berdasarkan kategori
        result = {}
        for item in all_items:
            kategori = item["kategori"]
            if kategori not in result:
                result[kategori] = {
                    "kategori": kategori,
                    "items": []
                }
                
            result[kategori]["items"].append({
                "id": item["id"],
                "item_checklist": item["item_checklist"],
                "status": item["status"] if item["status"] is not None else False
            })
            
        return list(result.values())
    finally:
        conn.close()

# Fungsi untuk mendapatkan data hazard dengan format yang sesuai
def get_hazard_data():
    """
    Mengambil data hazard dari database dan memformatnya untuk pencocokan
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Ambil semua data checklist
        cursor.execute("""
            SELECT id, kategori, item_checklist
            FROM l2sa_tha_data
            ORDER BY kategori, id
        """)
        
        all_items = cursor.fetchall()
        
        # Format data sesuai kebutuhan
        hazard_data = []
        current_category = None
        category_data = None
        
        for item in all_items:
            kategori = item["kategori"]
            
            # Jika kategori baru, buat entri baru
            if kategori != current_category:
                if category_data:
                    hazard_data.append(category_data)
                
                current_category = kategori
                category_data = {
                    "category": kategori,
                    "hazards": []
                }
            
            # Split item checklist jika perlu
            item_text = item["item_checklist"]
            control_text = "Ikuti prosedur standar"
            
            if " - " in item_text:
                parts = item_text.split(" - ")
                item_text = parts[0].strip()
                if len(parts) > 1:
                    control_text = parts[1].strip()
            
            # Tambahkan ke hazards
            category_data["hazards"].append({
                "id": item["id"],
                "hazard": item_text,
                "control": control_text
            })
        
        # Tambahkan kategori terakhir
        if category_data:
            hazard_data.append(category_data)
            
        return hazard_data
    finally:
        conn.close()

# Endpoint untuk mendapatkan data hazard terformat
@app.get("/checklist/formatted")
def get_formatted_hazard_data():
    return get_hazard_data()

# Endpoint untuk menyimpan checklist
@app.post("/checklist/save")
def save_checklist(request: SaveChecklistRequest, db: Session = Depends(get_db)):
    try:
        # Implementasi sederhana untuk demo
        print(f"Menyimpan hazard: {request.selectedHazards}")
        print(f"Custom hazards: {request.customHazards}")
        
        # Dalam implementasi nyata, simpan ke database
        return {"success": True, "message": "Checklist berhasil disimpan"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan checklist: {str(e)}")

# Perbarui endpoint save_checklist di l2sa_tha.py

@app.post("/l2sa/{l2sa_id}/checklist/save")
def save_l2sa_checklist(l2sa_id: str, request: SaveChecklistRequest, db: Session = Depends(get_db)):
    try:
        # Cek apakah tabel l2sa_tha_checklist sudah ada, jika belum buat tabel dengan tipe data yang benar
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS l2sa_tha_checklist (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                l2sa_id VARCHAR(50) NOT NULL,
                hazard_id VARCHAR(50) NOT NULL,
                category VARCHAR(100) NOT NULL,
                hazard_text TEXT NOT NULL,
                control_text TEXT NOT NULL,
                is_custom BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_by VARCHAR(100)
            )
        """))
        
        # Hapus checklist yang ada sebelumnya
        print(f"Deleting existing checklist for L2SA ID: {l2sa_id}")
        db.execute(
            text("DELETE FROM l2sa_tha_checklist WHERE l2sa_id = :l2sa_id"), 
            {"l2sa_id": l2sa_id}
        )
        
        # Dapatkan data hazard
        hazard_data = get_hazard_data()
        
        # Untuk debug
        print(f"Processing {len(request.selectedHazards)} selected hazards")
        
        # Simpan checklist baru
        for item in request.selectedHazards:
            # Parse hazard key (format: category-hazard-control)
            parts = item.split('-')
            if len(parts) < 2:
                print(f"Skipping invalid item format: {item}")
                continue  # Skip jika format tidak valid
                
            category = parts[0]
            control = parts[-1]
            hazard = '-'.join(parts[1:-1])  # Gabungkan kembali jika hazard mengandung hyphen
            
            print(f"Processing item - Category: {category}, Hazard: {hazard}, Control: {control}")
            
            # Cari ID dari hazard ini
            hazard_id = None
            for h_cat in hazard_data:
                if h_cat["category"] == category:
                    for h_item in h_cat["hazards"]:
                        if h_item["hazard"] == hazard and h_item["control"] == control:
                            hazard_id = str(h_item["id"])
                            print(f"  Found matching hazard ID: {hazard_id}")
                            break
                    
                    if hazard_id:
                        break
            
            # Jika tidak ditemukan, buat ID baru
            if not hazard_id:
                hazard_id = f"custom-{uuid.uuid4()}"
                print(f"  No matching hazard found, using generated ID: {hazard_id}")
            
            # Insert ke database
            print(f"  Inserting into database - L2SA ID: {l2sa_id}, Hazard ID: {hazard_id}")
            db.execute(
                text("""
                    INSERT INTO l2sa_tha_checklist 
                    (l2sa_id, hazard_id, category, hazard_text, control_text, is_custom)
                    VALUES (:l2sa_id, :hazard_id, :category, :hazard, :control, FALSE)
                """),
                {
                    "l2sa_id": l2sa_id,
                    "hazard_id": hazard_id,
                    "category": category,
                    "hazard": hazard,
                    "control": control
                }
            )
        
        # Simpan custom hazards
        print(f"Processing {len(request.customHazards)} custom hazards")
        for custom in request.customHazards:
            custom_id = f"custom-{custom['id']}"
            print(f"  Inserting custom hazard - ID: {custom_id}, Category: {custom['category']}")
            db.execute(
                text("""
                    INSERT INTO l2sa_tha_checklist 
                    (l2sa_id, hazard_id, category, hazard_text, control_text, is_custom)
                    VALUES (:l2sa_id, :hazard_id, :category, :hazard, :control, TRUE)
                """),
                {
                    "l2sa_id": l2sa_id,
                    "hazard_id": custom_id,
                    "category": custom["category"],
                    "hazard": custom["hazard"],
                    "control": custom["control"]
                }
            )
        
        # Commit perubahan
        db.commit()
        print(f"Successfully saved checklist for L2SA ID: {l2sa_id}")
        return {"success": True, "message": "Checklist berhasil disimpan"}
    
    except Exception as e:
        db.rollback()
        print(f"Error saving checklist: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan checklist: {str(e)}")
    
# Endpoint untuk mendapatkan checklist L2SA tertentu
@app.get("/l2sa/{l2sa_id}/checklist")
def get_l2sa_checklist(l2sa_id: str, db: Session = Depends(get_db)):
    try:
        # Ambil checklist dari database
        result = db.execute(
            text("""
                SELECT id, hazard_id, category, hazard_text, control_text, is_custom
                FROM l2sa_tha_checklist
                WHERE l2sa_id = :l2sa_id
            """), 
            {"l2sa_id": l2sa_id}
        ).fetchall()
        
        # Format hasil
        checklist = {
            "selectedHazards": [],
            "customHazards": []
        }
        
        for item in result:
            if item.is_custom:
                # Format custom hazard
                custom_id = item.hazard_id.replace("custom-", "")
                checklist["customHazards"].append({
                    "id": custom_id,
                    "category": item.category,
                    "hazard": item.hazard_text,
                    "control": item.control_text
                })
            else:
                # Format selected hazard
                hazard_key = f"{item.category}-{item.hazard_text}-{item.control_text}"
                checklist["selectedHazards"].append(hazard_key)
        
        return checklist
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal mengambil checklist: {str(e)}")
    
# Jalankan server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("l2sa_tha:app", host="0.0.0.0", port=8003, reload=True)