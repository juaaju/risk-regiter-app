"""
API untuk PostgreSQL Embedder dengan FastAPI
"""
# Tambahkan import untuk lmstudio
import lmstudio as lms
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from postgres_embedder import PostgresEmbedder
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables
# Tambahkan import yang diperlukan
import requests  # untuk memanggil API model bahasa eksternal
import traceback

load_dotenv()

app = FastAPI(
    title="PostgreSQL Embedder API",
    description="API untuk membuat embedding dan mencari dokumen yang mirip",
    version="1.0.0"
)

# Tambahkan ini setelah membuat app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Memperbolehkan akses dari semua origin (untuk development)
    allow_credentials=True,
    allow_methods=["*"],  # Memperbolehkan semua HTTP methods
    allow_headers=["*"],  # Memperbolehkan semua headers
)

# Konfigurasi database
def get_db_config():
    return {
        'db_host': os.environ.get('DB_HOST', 'localhost'),
        'db_port': int(os.environ.get('DB_PORT', '5432')),
        'db_name': os.environ.get('DB_NAME', 'postgres'),
        'db_user': os.environ.get('DB_USER', 'postgres'),
        'db_password': os.environ.get('DB_PASSWORD', ''),
    }

# Tambahkan model untuk request RAG
class RagRequest(BaseModel):
    query: str
    embedding_table: str
    source_table: str
    id_column: str = 'id'
    limit: int = 3

# Model untuk request dan response
class EmbedRequest(BaseModel):
    source_table: str
    target_table: str
    text_columns: List[str]
    id_column: str = 'id'

class SearchRequest(BaseModel):
    query: str
    embedding_table: str
    source_table: str
    id_column: str = 'id'
    limit: int = 5

class SearchResult(BaseModel):
    results: List[Dict[str, Any]]
    count: int

# API Endpoints
@app.post("/embed", summary="Buat embedding untuk dokumen dari tabel")
async def embed_documents(request: EmbedRequest):
    """
    Membuat embedding untuk dokumen dari tabel PostgreSQL dan menyimpannya
    ke tabel target
    """
    db_config = get_db_config()
    embedder = PostgresEmbedder(**db_config)
    
    try:
        embedder.connect()
        
        # Cek apakah user meminta semua kolom
        if request.text_columns == ["ALL"]:
            all_columns_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{request.source_table.lower()}'
            AND column_name != '{request.id_column.lower()}'
            """
            columns_data = embedder.fetch_data(all_columns_query)
            text_columns = [col['column_name'] for col in columns_data]
            
            # Buat dictionary dari nama kolom ke tipe data
            column_types = {col['column_name']: col['data_type'] for col in columns_data}
        else:
            text_columns = request.text_columns
        
        # Buat query dengan konversi eksplisit ke text untuk setiap kolom
        columns_concat_parts = []
        for col in text_columns:
            # Konversi eksplisit ke text untuk semua jenis data
            columns_concat_parts.append(f"COALESCE(CAST({col} AS TEXT), '')")
        
        columns_concat = " || ' ' || ".join(columns_concat_parts)
        
        query = f"""
        SELECT 
            {request.id_column},
            {columns_concat} AS combined_text
        FROM 
            {request.source_table}
        """
        
        # Ambil data dan buat embedding
        data = embedder.create_embeddings_for_query(
            query=query,
            text_column='combined_text',
            id_column=request.id_column
        )
        
        # Simpan embedding ke tabel target
        embedder.save_embeddings_to_table(
            data=data,
            table_name=request.target_table,
            id_column=request.id_column,
            create_table_if_not_exists=True,
            use_pgvector=True
        )
        
        return {
            "status": "success",
            "message": f"Created and saved embeddings for {len(data)} documents",
            "table": request.target_table,
            "count": len(data)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
    
    finally:
        embedder.disconnect()

@app.post("/search", response_model=SearchResult, summary="Cari dokumen yang mirip")
async def search_documents(request: SearchRequest):
    """
    Mencari dokumen yang paling mirip dengan query menggunakan embedding
    """
    db_config = get_db_config()
    embedder = PostgresEmbedder(**db_config)
    
    try:
        embedder.connect()
        
        # Cari dokumen yang mirip
        similar_docs = embedder.find_similar_documents(
            query_text=request.query,
            table_name=request.embedding_table,
            id_column=request.id_column,
            limit=request.limit
        )
        # Debug struktur data
        print("Similar docs structure:", similar_docs)
        if similar_docs and len(similar_docs) > 0:
            print("First doc:", similar_docs[0])
            # Periksa apakah similarity ada
            if 'similarity' in similar_docs[0]:
                # Urutkan berdasarkan similarity (untuk amannya)
                similar_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                # Ambil hanya yang teratas
                similar_docs = [similar_docs[0]]
            else:
                print("No similarity key found in results")
                
        if not similar_docs:
            return {"results": [], "count": 0}
        
        # Ambil data dari tabel sumber
        ids = [doc[request.id_column] for doc in similar_docs]
        placeholders = ','.join(['%s'] * len(ids))
        
        # Buat query untuk mengambil data dokumen
        docs_query = f"""
        SELECT * FROM {request.source_table}
        WHERE {request.id_column} IN ({placeholders})
        """
        
        documents = embedder.fetch_data(docs_query, tuple(ids))
        
        # Gabungkan hasil dengan similarity score
        id_to_similarity = {doc[request.id_column]: doc['similarity'] for doc in similar_docs}
        for doc in documents:
            doc['similarity'] = id_to_similarity.get(doc[request.id_column], 0)
        
        # Urutkan berdasarkan similarity
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {"results": documents, "count": len(documents)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")
    
    finally:
        embedder.disconnect()

@app.get("/tables", summary="Daftar tabel dalam database")
async def list_tables():
    """
    Mendapatkan daftar tabel dalam database
    """
    db_config = get_db_config()
    embedder = PostgresEmbedder(**db_config)
    
    try:
        embedder.connect()
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        tables = embedder.fetch_data(query)
        return {"tables": [table["table_name"] for table in tables]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")
    
    finally:
        embedder.disconnect()

@app.get("/columns/{table_name}", summary="Daftar kolom dalam tabel")
async def list_columns(table_name: str):
    """
    Mendapatkan daftar kolom dalam tabel
    """
    db_config = get_db_config()
    embedder = PostgresEmbedder(**db_config)
    
    try:
        embedder.connect()
        query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_name = '{table_name.lower()}'
        ORDER BY ordinal_position
        """
        columns = embedder.fetch_data(query)
        return {"table": table_name, "columns": columns}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing columns: {str(e)}")
    
    finally:
        embedder.disconnect()

# Endpoint baru untuk RAG
@app.post("/rag", summary="Cari dokumen dan hasilkan jawaban")
async def generate_from_documents(request: RagRequest):
    """
    Mencari dokumen yang paling mirip dengan query dan menghasilkan jawaban
    berdasarkan konteks dokumen tersebut
    """
    db_config = get_db_config()
    embedder = PostgresEmbedder(**db_config)
    
    try:
        embedder.connect()
        
        # Cari dokumen yang mirip (sama seperti endpoint search)
        similar_docs = embedder.find_similar_documents(
            query_text=request.query,
            table_name=request.embedding_table,
            id_column=request.id_column,
            limit=request.limit
        )
        
        if not similar_docs:
            return {
                "answer": "Maaf, tidak ditemukan dokumen yang relevan.",
                "documents_used": [],
                "count": 0
            }
        
        # Ambil data dari tabel sumber
        ids = [doc[request.id_column] for doc in similar_docs]
        placeholders = ','.join(['%s'] * len(ids))
        
        docs_query = f"""
        SELECT * FROM {request.source_table}
        WHERE {request.id_column} IN ({placeholders})
        """
        
        documents = embedder.fetch_data(docs_query, tuple(ids))
        
        # Gabungkan hasil dengan similarity
        id_to_similarity = {doc[request.id_column]: doc['similarity'] for doc in similar_docs}
        for doc in documents:
            doc['similarity'] = id_to_similarity.get(doc[request.id_column], 0)
        
        # Urutkan berdasarkan similarity
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Buat context dari dokumen
        context = "Konteks dari dokumen yang relevan:\n\n"
        for i, doc in enumerate(documents):
            # Konversi doc menjadi teks
            doc_text = "\n".join([f"{k}: {v}" for k, v in doc.items() 
                               if k not in ['similarity', 'embedding']])
            context += f"Dokumen {i+1} (ID: {doc[request.id_column]}):\n{doc_text}\n\n"
        
        # Buat prompt untuk model bahasa
        prompt = f"{context}\n\nPertanyaan: {request.query}\n\nBerdasarkan konteks di atas, jawablah pertanyaan tersebut:"
        
        # Gunakan lmstudio library langsung
        try:
            # Gunakan model yang sesuai - sesuaikan nama model sesuai kebutuhan
            model = lms.llm("qwen2.5-7b-instruct-1m")  # Ganti dengan model yang tersedia di lingkungan Anda
            result = model.respond(prompt)
            
            # Ekstrak teks dari hasil
            response_text = ""
            if hasattr(result, 'text'):
                response_text = result.text
            elif hasattr(result, 'content'):
                response_text = result.content
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
                
            return {
                "answer": response_text,
                "documents_used": documents,
                "count": len(documents)
            }
            
        except Exception as llm_error:
            print(f"Error calling LLM: {str(llm_error)}")
            return {
                "answer": f"Terjadi kesalahan saat menghasilkan jawaban: {str(llm_error)}",
                "documents_used": documents,
                "count": len(documents)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in RAG process: {str(e)}")
    
    finally:
        embedder.disconnect()

# Tambahkan model untuk request generate
class GenerateRequest(BaseModel):
    prompt: str
    use_rag: bool = False
    document_ids: Optional[List[str]] = None
    database: Optional[str] = None
    model_name: str = "qwen2.5-7b-instruct-1m"  # Default model

@app.post("/generate", summary="Generate text dengan model LLM")
async def generate_text(request: GenerateRequest):
    """
    Menghasilkan teks menggunakan LM Studio model
    """
    try:
        # Jika menggunakan RAG dengan document_ids
        if request.use_rag and request.document_ids:
            # Implementasi RAG dengan document_ids
            # Ini bisa diimplementasikan jika Anda memiliki akses ke dokumen yang diunggah
            pass
            
        # Jika menggunakan RAG dengan database
        elif request.use_rag and request.database:
            # Redirect ke endpoint /rag
            # Tentukan embedding_table dan source_table berdasarkan database
            # (Implementasi mapping database ke tabel)
            pass
        
        # Tidak menggunakan RAG, hanya LLM
        else:
            # Gunakan model LM Studio langsung
            model = lms.llm(request.model_name)
            result = model.respond(request.prompt)
            
            # Ekstrak teks dari hasil
            response_text = ""
            if hasattr(result, 'text'):
                response_text = result.text
            elif hasattr(result, 'content'):
                response_text = result.content
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
                
            # Ekstrak statistik jika ada
            stats = {}
            if hasattr(result, 'stats'):
                stats = {
                    'tokens': getattr(result.stats, 'tokens', None),
                    'time': getattr(result.stats, 'time', None)
                }
                
            return {
                "result": response_text,
                "stats": stats,
                "documents_used": []
            }
            
    except Exception as e:
        return {
            "error": f"Error generating text: {str(e)}",
            "traceback": traceback.format_exc()
        }

# Menjalankan server jika file dieksekusi secara langsung
# Di bagian bawah file riskrag_api.py, tambahkan:
if __name__ == "__main__":
    uvicorn.run("riskrag_api:app", host="0.0.0.0", port=8001, reload=True)