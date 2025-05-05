import os
import json
import time
import logging
import traceback
from typing import List, Optional
from enum import Enum
from datetime import datetime

import lmstudio as lms
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import Json, DictCursor
import asyncpg
import numpy as np

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag-chatbot")

# Inisialisasi model embedding dan LLM
try:
    logger.info("Mencoba inisialisasi model embedding...")
    embedding_model = lms.embedding_model("nomic-embed-text-v1.5")
    logger.info("Model embedding berhasil diinisialisasi.")
    
    logger.info("Mencoba inisialisasi model LLM...")
    llm_model = lms.llm()
    logger.info("Model LLM berhasil diinisialisasi.")
except Exception as e:
    logger.error(f"Error saat inisialisasi model: {str(e)}")
    logger.error(traceback.format_exc())

# Router FastAPI untuk API chatbot terpisah
app = FastAPI(title="Chatbot RAG API")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua origin - untuk development
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua HTTP methods
    allow_headers=["*"],  # Izinkan semua headers
)

# Konfigurasi database
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "risk_management",
    "user": "postgres",
    "password": "toor"
}

# Enum untuk kategori dokumen
class DocumentCategory(str, Enum):
    standard_safety = "standard_safety"
    general_document = "general_document"

# Model untuk kueri chatbot
class ChatQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    category: Optional[DocumentCategory] = None
    max_chunks: int = 5
    similarity_threshold: float = 0.5
    include_sources: bool = True

# Model untuk respons chatbot
class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    conversation_id: str
    query_id: int
    elapsed_time: float

# Model untuk riwayat percakapan
class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[dict]

# Fungsi untuk koneksi ke database
def get_psycopg2_connection():
    try:
        logger.debug("Mencoba membuat koneksi psycopg2...")
        conn = psycopg2.connect(
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname=DB_CONFIG["database"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"]
        )
        logger.debug("Koneksi psycopg2 berhasil dibuat.")
        return conn
    except Exception as e:
        logger.error(f"Error membuat koneksi psycopg2: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Fungsi untuk mencari dokumen berbasis embedding
def search_similar_chunks(query_embedding, category=None, max_chunks=5, similarity_threshold=0.5):
    logger.info(f"Mencari chunks yang mirip dengan query. Category: {category}, Max chunks: {max_chunks}")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Modifikasi query untuk mendapatkan original_filename
        if category:
            logger.info(f"Pencarian dengan filter kategori: {category}")
            query = """
            SELECT
                dc.id AS chunk_id,
                d.id AS document_id,
                d.filename,
                d.original_filename, -- Tambahkan original_filename
                d.category AS category,
                dc.content,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM
                document_chunks dc
            JOIN
                documents d ON dc.document_id = d.id
            WHERE
                d.category = %s
                AND 1 - (dc.embedding <=> %s::vector) > %s
            ORDER BY
                dc.embedding <=> %s::vector
            LIMIT %s
            """
            cursor.execute(query, (query_embedding, category, query_embedding, similarity_threshold, query_embedding, max_chunks))
        else:
            logger.info("Pencarian tanpa filter kategori")
            query = """
            SELECT
                dc.id AS chunk_id,
                d.id AS document_id,
                d.filename,
                d.original_filename, -- Tambahkan original_filename
                d.category AS category,
                dc.content,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM
                document_chunks dc
            JOIN
                documents d ON dc.document_id = d.id
            WHERE
                1 - (dc.embedding <=> %s::vector) > %s
            ORDER BY
                dc.embedding <=> %s::vector
            LIMIT %s
            """
            cursor.execute(query, (query_embedding, query_embedding, similarity_threshold, query_embedding, max_chunks))
        
        results = cursor.fetchall()
        logger.info(f"Pencarian mengembalikan {len(results)} hasil")
        
        # Format hasil - juga tambahkan original_filename ke output
        formatted_results = []
        for row in results:
            logger.debug(f"Hasil: chunk_id={row['chunk_id']}, dokumen={row['original_filename']}, similarity={row['similarity']}")
            formatted_results.append({
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "original_filename": row["original_filename"],
                "category": row["category"],
                "content": row["content"],
                "similarity": float(row["similarity"])
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error saat pencarian chunks: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        cursor.close()
        conn.close()
        logger.debug("Koneksi database ditutup")

# Fungsi untuk menyimpan riwayat kueri
def save_query_history(query_text, query_embedding, num_results, response_time_ms):
    logger.info(f"Menyimpan riwayat kueri: '{query_text}'")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO query_history (query_text, query_embedding, num_results, response_time_ms)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (query_text, query_embedding, num_results, response_time_ms)
        )
        query_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Riwayat kueri disimpan dengan ID: {query_id}")
        return query_id
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saat menyimpan riwayat kueri: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        cursor.close()
        conn.close()

# Fungsi untuk mencatat akses dokumen
def log_document_access(document_id, query_id, access_type="chat"):
    logger.debug(f"Mencatat akses dokumen ID {document_id} dari kueri ID {query_id}")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO document_access_logs (document_id, access_type, query_id)
            VALUES (%s, %s, %s)
            """,
            (document_id, access_type, query_id)
        )
        conn.commit()
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saat mencatat akses dokumen: {str(e)}")
    
    finally:
        cursor.close()
        conn.close()

# Fungsi untuk membuat prompt untuk LLM
def create_rag_prompt(query, chunks):
    # Kelompokkan chunks berdasarkan document_id
    grouped_chunks = {}
    for chunk in chunks:
        doc_id = chunk['document_id']
        if doc_id not in grouped_chunks:
            grouped_chunks[doc_id] = {
                'filename': chunk['original_filename'] if 'original_filename' in chunk else chunk['filename'],
                'category': chunk['category'],
                'chunks': []
            }
        grouped_chunks[doc_id]['chunks'].append(chunk['content'])
    
    # Buat konteks per dokumen
    context = ""
    doc_index = 1
    for doc_id, doc_info in grouped_chunks.items():
        # Gabungkan semua chunks dari dokumen yang sama
        doc_content = "\n".join(doc_info['chunks'])
        
        # Tambahkan ke konteks dengan judul dokumen menggunakan nama original
        context += f"\nDOKUMEN {doc_index}: {doc_info['filename']} ({doc_info['category']})\n"
        context += "--------------------------------------\n"
        context += doc_content + "\n\n"
        doc_index += 1
    
    prompt = f"""Kamu adalah asisten AI Safety Assistant yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan. 
Berikan jawaban yang sangat sangat lengkap, akurat, dan bermanfaat persis seperti dokumen. Rujuk ke dokumen dengan menyebutkan nama file asli.
JANGAN menyebutkan UUID atau ID internal dalam jawaban. Jika pengguna tidak bertanya tentang dokumen, jawab dengan pengetahuan umum anda.

DOKUMEN YANG TERSEDIA:
{context}

PERTANYAAN: {query}

JAWABAN:"""
    
    return prompt
# Endpoint untuk chatbot
@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_query: ChatQuery):
    logger.info(f"Menerima kueri chat: '{chat_query.query}'")
    start_time = time.time()
    
    try:
        # Buat embedding dari query
        logger.info("Membuat embedding dari query")
        query_embedding = embedding_model.embed(chat_query.query)
        
        # Cari chunk yang relevan
        similar_chunks = search_similar_chunks(
            query_embedding, 
            category=chat_query.category,
            max_chunks=chat_query.max_chunks,
            similarity_threshold=chat_query.similarity_threshold
        )
        
        if not similar_chunks:
            logger.warning("Tidak ditemukan dokumen yang relevan")
            # Jika tidak ada chunk yang relevan, gunakan LLM tanpa konteks
            answer_result = llm_model.respond(f"Pertanyaan: {chat_query.query}\n\nJawaban:")
            
            # Pastikan answer adalah string, bukan objek
            if hasattr(answer_result, 'content'):
                # Jika objek PredictionResult, ambil content
                answer = answer_result.content
            else:
                # Jika sudah string, gunakan langsung
                answer = str(answer_result)
                
            elapsed_time = time.time() - start_time
            
            # Simpan riwayat kueri - gunakan versi tanpa embedding untuk menghindari masalah dimensi
            query_id = save_query_history_without_embedding(
                chat_query.query,
                0,
                int(elapsed_time * 1000)
            )
            
            return {
                "answer": "Saya tidak menemukan informasi yang relevan dalam dokumen yang tersedia. " + answer,
                "sources": None,
                "conversation_id": chat_query.conversation_id or str(int(time.time())),
                "query_id": query_id,
                "elapsed_time": elapsed_time
            }
        
        # Buat prompt untuk LLM dengan konteks dari dokumen
        prompt = create_rag_prompt(chat_query.query, similar_chunks)
        
        # Dapatkan respons dari LLM
        logger.info("Mengirim prompt ke LLM")
        answer_result = llm_model.respond(prompt)
        
        # Pastikan answer adalah string, bukan objek
        if hasattr(answer_result, 'content'):
            # Jika objek PredictionResult, ambil content
            answer = answer_result.content
        else:
            # Jika sudah string, gunakan langsung
            answer = str(answer_result)
            
        elapsed_time = time.time() - start_time
        
        # Simpan riwayat kueri - gunakan versi tanpa embedding
        query_id = save_query_history_without_embedding(
            chat_query.query,
            len(similar_chunks), 
            int(elapsed_time * 1000)
        )
        
        # Catat akses dokumen
        for chunk in similar_chunks:
            log_document_access(chunk["document_id"], query_id)
        
        # Siapkan sumber untuk ditampilkan
        sources = None
        if chat_query.include_sources:
            sources = []
            for chunk in similar_chunks:
                sources.append({
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "category": chunk["category"],
                    "excerpt": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "similarity": chunk["similarity"],
                    "download_url": f"http://localhost:8006/api/documents/download/{chunk['document_id']}"  # URL untuk mengunduh
                })
        
        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": chat_query.conversation_id or str(int(time.time())),
            "query_id": query_id,
            "elapsed_time": elapsed_time
        }
    
    except Exception as e:
        logger.error(f"Error saat memproses kueri chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Fungsi untuk menyimpan riwayat kueri tanpa embedding
def save_query_history_without_embedding(query_text, num_results, response_time_ms):
    logger.info(f"Menyimpan riwayat kueri (tanpa embedding): '{query_text}'")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO query_history (query_text, num_results, response_time_ms)
            VALUES (%s, %s, %s) RETURNING id
            """,
            (query_text, num_results, response_time_ms)
        )
        query_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"Riwayat kueri disimpan dengan ID: {query_id}")
        return query_id
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saat menyimpan riwayat kueri: {str(e)}")
        logger.error(traceback.format_exc())
        # Return dummy ID jika gagal menyimpan
        return -1
    
    finally:
        cursor.close()
        conn.close()# Endpoint untuk mendapatkan riwayat kueri
        
@app.get("/api/query-history")
async def get_query_history(limit: int = 20, offset: int = 0):
    logger.info(f"Mendapatkan riwayat kueri. Limit: {limit}, Offset: {offset}")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    try:
        cursor.execute(
            """
            SELECT id, query_text, timestamp, response_time_ms, num_results
            FROM query_history
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset)
        )
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "id": row["id"],
                "query": row["query_text"],
                "timestamp": row["timestamp"].isoformat(),
                "response_time_ms": row["response_time_ms"],
                "num_results": row["num_results"]
            })
        
        # Dapatkan total count
        cursor.execute("SELECT COUNT(*) FROM query_history")
        total_count = cursor.fetchone()[0]
        
        return {
            "history": history,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    
    except Exception as e:
        logger.error(f"Error saat mendapatkan riwayat kueri: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()

# Endpoint untuk mendapatkan detail kueri
@app.get("/api/query-history/{query_id}")
async def get_query_detail(query_id: int):
    logger.info(f"Mendapatkan detail kueri ID: {query_id}")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # Dapatkan informasi kueri
        cursor.execute(
            """
            SELECT id, query_text, timestamp, response_time_ms, num_results
            FROM query_history
            WHERE id = %s
            """,
            (query_id,)
        )
        
        query_info = cursor.fetchone()
        if not query_info:
            raise HTTPException(status_code=404, detail="Kueri tidak ditemukan")
        
        # Dapatkan dokumen yang diakses
        cursor.execute(
            """
            SELECT dal.id, dal.timestamp, d.id as document_id, d.original_filename, d.category
            FROM document_access_logs dal
            JOIN documents d ON dal.document_id = d.id
            WHERE dal.query_id = %s
            ORDER BY dal.timestamp DESC
            """,
            (query_id,)
        )
        
        accessed_documents = []
        for row in cursor.fetchall():
            accessed_documents.append({
                "log_id": row["id"],
                "document_id": row["document_id"],
                "filename": row["original_filename"],
                "category": row["category"],
                "timestamp": row["timestamp"].isoformat()
            })
        
        return {
            "query_id": query_info["id"],
            "query_text": query_info["query_text"],
            "timestamp": query_info["timestamp"].isoformat(),
            "response_time_ms": query_info["response_time_ms"],
            "num_results": query_info["num_results"],
            "accessed_documents": accessed_documents
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error saat mendapatkan detail kueri: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()

# Endpoint untuk mendapatkan statistik penggunaan
@app.get("/api/stats")
async def get_stats():
    logger.info("Mendapatkan statistik penggunaan")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor()
    
    try:
        stats = {}
        
        # Total kueri
        cursor.execute("SELECT COUNT(*) FROM query_history")
        stats["total_queries"] = cursor.fetchone()[0]
        
        # Rata-rata waktu respons
        cursor.execute("SELECT AVG(response_time_ms) FROM query_history")
        avg_response_time = cursor.fetchone()[0]
        stats["avg_response_time_ms"] = round(avg_response_time, 2) if avg_response_time else 0
        
        # Total dokumen
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["total_documents"] = cursor.fetchone()[0]
        
        # Total chunk
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        stats["total_chunks"] = cursor.fetchone()[0]
        
        # Kueri per kategori
        cursor.execute("""
            SELECT d.category, COUNT(DISTINCT dal.query_id) as query_count
            FROM document_access_logs dal
            JOIN documents d ON dal.document_id = d.id
            GROUP BY d.category
        """)
        category_stats = {}
        for row in cursor.fetchall():
            category_stats[row[0]] = row[1]
        stats["queries_by_category"] = category_stats
        
        # Top 5 dokumen yang paling sering diakses
        cursor.execute("""
            SELECT d.id, d.original_filename, COUNT(dal.id) as access_count
            FROM document_access_logs dal
            JOIN documents d ON dal.document_id = d.id
            GROUP BY d.id, d.original_filename
            ORDER BY access_count DESC
            LIMIT 5
        """)
        top_documents = []
        for row in cursor.fetchall():
            top_documents.append({
                "document_id": row[0],
                "filename": row[1],
                "access_count": row[2]
            })
        stats["top_accessed_documents"] = top_documents
        
        return stats
    
    except Exception as e:
        logger.error(f"Error saat mendapatkan statistik: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()

from fastapi.responses import FileResponse

@app.get("/api/documents/download/{document_id}")
async def download_document(document_id: int):
    logger.info(f"Permintaan download dokumen ID: {document_id}")
    
    conn = get_psycopg2_connection()
    cursor = conn.cursor()
    
    try:
        # Dapatkan informasi file dari database
        cursor.execute(
            "SELECT file_path, original_filename, file_type FROM documents WHERE id = %s",
            (document_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"Dokumen ID {document_id} tidak ditemukan")
            raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan")
            
        file_path, original_filename, file_type = result
        
        # Periksa apakah file ada
        if not os.path.exists(file_path):
            logger.error(f"File tidak ditemukan di sistem: {file_path}")
            raise HTTPException(status_code=404, detail="File tidak ditemukan di sistem")
        
        # Tentukan media_type berdasarkan file_type atau ekstensi file
        media_type = file_type or "application/octet-stream"
        
        logger.info(f"Mengirim file: {original_filename}, path: {file_path}")
        
        # Kembalikan file dengan nama asli (original_filename)
        # tapi membaca dari lokasi file dengan nama UUID (file_path)
        return FileResponse(
            path=file_path,  # Path fisik menggunakan UUID
            filename=original_filename,  # Nama file yang terlihat pengguna adalah nama asli
            media_type=media_type
        )
    
    except Exception as e:
        logger.error(f"Error saat mengunduh dokumen: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        cursor.close()
        conn.close()

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Jalankan server dengan uvicorn
if __name__ == "__main__":
    import uvicorn
    logger.info("Memulai server API Chatbot...")
    uvicorn.run(app, host="0.0.0.0", port=8006)  # Gunakan port berbeda dari API utama