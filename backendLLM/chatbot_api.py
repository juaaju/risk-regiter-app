# app.py - File lengkap untuk API LM Studio dengan RAG

import os
import uuid
import json
import shutil
import numpy as np
from tempfile import NamedTemporaryFile
from pathlib import Path
import traceback

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import lmstudio as lms

# Inisialisasi FastAPI
app = FastAPI(title="LM Studio RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buat direktori untuk menyimpan dokumen
DOCUMENTS_DIR = Path("./documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Simpan dokumen dalam memori (untuk produksi, gunakan database)
documents = {}

# Model kelas
class Query(BaseModel):
    prompt: str
    use_rag: bool = True
    num_documents: int = 3
    document_ids: Optional[List[str]] = None

class Document(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    metadata: Optional[Dict[str, Any]] = None

# Fungsi untuk membuat embedding teks
def embed_text(text: str):
    try:
        print(f"Embedding text of length: {len(text)}")
        
        # Gunakan model embedding dari LM Studio
        try:
            embed_model = lms.embedding("nomic-embed-text-v1.5")
            embedding = embed_model.embed(text)
            print("Embedding created successfully with the model")
            return embedding
        except Exception as model_error:
            print(f"Error using embedding model: {str(model_error)}")
            print("Falling back to mock embedding")
            
            # Fallback ke embedding dummy jika model gagal
            return np.random.rand(384)  # Dimensi umum untuk embedding
    except Exception as e:
        print(f"Error in embed_text: {str(e)}")
        print(traceback.format_exc())
        raise

# Fungsi untuk menghitung kemiripan
def compute_similarity(query_embedding, document_embedding):
    try:
        # Normalisasi embedding
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(document_embedding)
        
        if query_norm == 0 or doc_norm == 0:
            return 0
        
        # Hitung kemiripan kosinus
        similarity = np.dot(query_embedding, document_embedding) / (query_norm * doc_norm)
        return similarity
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return 0

# Fungsi untuk mencari dokumen relevan
def retrieve_documents(query_embedding, num_documents=3, document_ids=None):
    try:
        results = []
        
        docs_to_search = documents
        if document_ids:
            docs_to_search = {doc_id: doc for doc_id, doc in documents.items() if doc_id in document_ids}
        
        for doc_id, doc in docs_to_search.items():
            if 'embedding' in doc:
                similarity = compute_similarity(query_embedding, doc['embedding'])
                results.append((doc_id, similarity))
        
        # Urutkan berdasarkan kemiripan
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_documents]
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return []

# Endpoint untuk menghasilkan respons
@app.post("/generate")
async def generate(query: Query):
    try:
        print(f"Received generate request: {query.prompt}")
        relevant_docs = []
        
        if query.use_rag:
            print("RAG is enabled")
            # Buat embedding untuk query
            query_embedding = embed_text(query.prompt)
            
            # Ambil dokumen relevan
            retrieved_docs = retrieve_documents(
                query_embedding, 
                num_documents=query.num_documents,
                document_ids=query.document_ids
            )
            
            print(f"Retrieved {len(retrieved_docs)} relevant documents")
            
            # Kumpulkan konten dokumen
            for doc_id, similarity in retrieved_docs:
                if doc_id in documents:
                    doc_content = documents[doc_id]['content']
                    relevant_docs.append({
                        "id": doc_id,
                        "content": doc_content,
                        "similarity": float(similarity)
                    })
        else:
            print("RAG is disabled")
        
        # Buat prompt RAG
        context = ""
        if relevant_docs:
            context = "Context information:\n"
            for i, doc in enumerate(relevant_docs):
                context += f"Document {i+1}:\n{doc['content']}\n\n"
        
        full_prompt = f"{context}\nUser query: {query.prompt}\n\nAnswer based on the provided context:"
        
        # Generate respons menggunakan LM Studio
        print(f"Calling LLM with prompt (RAG: {query.use_rag})")
        model = lms.llm("qwen2.5-7b-instruct-1m")
        result = model.respond(full_prompt if relevant_docs else query.prompt)
        
        # Ekstrak teks respons
        response_text = ""
        if hasattr(result, 'text'):
            response_text = result.text
        elif hasattr(result, 'content'):
            response_text = result.content
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result)
        
        print(f"Generated response of length: {len(response_text)}")
        
        # Kumpulkan statistik
        stats = {}
        if hasattr(result, 'stats'):
            stats = {
                'tokens': getattr(result.stats, 'tokens', None),
                'time': getattr(result.stats, 'time', None)
            }
        
        return {
            "result": response_text,
            "stats": stats,
            "documents_used": relevant_docs if query.use_rag else [],
        }
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in generate endpoint: {error_msg}")
        print(trace)
        return {"error": error_msg, "traceback": trace}

# Endpoint untuk menambahkan dokumen dengan teks
@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    title: str = Form(...),
    content: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    try:
        print(f"Received document upload: {title}")
        
        # Generate ID dokumen
        doc_id = str(uuid.uuid4())
        
        # Parse metadata jika ada
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                print(f"Invalid metadata JSON: {metadata}")
                meta_dict = {"error": "Invalid JSON"}
        
        # Buat embedding untuk dokumen
        embedding = embed_text(content)
        
        # Simpan dokumen
        documents[doc_id] = {
            "title": title,
            "content": content,
            "metadata": meta_dict,
            "embedding": embedding
        }
        
        print(f"Document stored with ID: {doc_id}")
        return {"id": doc_id, "title": title, "metadata": meta_dict}
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in upload_document: {error_msg}")
        print(trace)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint untuk mengunggah file
@app.post("/documents/file", response_model=DocumentResponse)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    try:
        print(f"Received file upload: {file.filename}")
        
        # Buat file temporary
        temp_path = None
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            try:
                # Tulis konten file yang diunggah ke file temporary
                shutil.copyfileobj(file.file, temp)
                temp_path = temp.name
                print(f"Saved to temp file: {temp_path}")
            except Exception as copy_error:
                print(f"Error copying file: {str(copy_error)}")
                raise HTTPException(status_code=500, detail=f"Error saving file: {str(copy_error)}")
        
        try:
            # Baca konten file dengan penanganan error yang lebih baik
            content = ""
            try:
                with open(temp_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                print(f"Successfully read file content, length: {len(content)}")
            except Exception as read_error:
                print(f"Error reading file as text: {str(read_error)}")
                
                # Coba dengan mode binary jika mode teks gagal
                try:
                    with open(temp_path, 'rb') as f:
                        binary_content = f.read()
                    content = binary_content.decode('utf-8', errors='replace')
                    print(f"Read file in binary mode, length: {len(content)}")
                except Exception as binary_error:
                    print(f"Error reading file in binary mode: {str(binary_error)}")
                    raise HTTPException(status_code=500, detail=f"Could not read file content: {str(binary_error)}")
            
            # Gunakan nama file sebagai judul jika tidak disediakan
            if not title:
                title = file.filename
            
            # Parse metadata jika ada
            meta_dict = {}
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    print(f"Invalid metadata JSON: {metadata}")
                    meta_dict = {"error": "Invalid JSON"}
            
            # Generate ID dokumen
            doc_id = str(uuid.uuid4())
            
            # Buat embedding untuk dokumen
            try:
                print("Creating embedding...")
                embedding = embed_text(content)
                print("Embedding created successfully")
            except Exception as embed_error:
                print(f"Error creating embedding: {str(embed_error)}")
                raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(embed_error)}")
            
            # Simpan dokumen
            try:
                documents[doc_id] = {
                    "title": title,
                    "content": content,
                    "metadata": meta_dict,
                    "embedding": embedding,
                    "file_type": os.path.splitext(file.filename)[1]
                }
                print(f"Document stored with ID: {doc_id}")
            except Exception as store_error:
                print(f"Error storing document: {str(store_error)}")
                raise HTTPException(status_code=500, detail=f"Error storing document: {str(store_error)}")
            
            return {"id": doc_id, "title": title, "metadata": meta_dict}
        finally:
            # Bersihkan file temporary
            if temp_path:
                try:
                    os.unlink(temp_path)
                    print(f"Temp file {temp_path} removed")
                except Exception as cleanup_error:
                    print(f"Error removing temp file: {str(cleanup_error)}")
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in upload_document_file: {error_msg}")
        print(trace)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint untuk mendapatkan daftar dokumen
@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    try:
        return [
            {"id": doc_id, "title": doc["title"], "metadata": doc.get("metadata", {})}
            for doc_id, doc in documents.items()
        ]
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in list_documents: {error_msg}")
        print(trace)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint untuk mendapatkan dokumen berdasarkan ID
@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    try:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = documents[doc_id]
        return {
            "id": doc_id,
            "title": doc["title"],
            "content": doc["content"],
            "metadata": doc.get("metadata", {})
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in get_document: {error_msg}")
        print(trace)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint untuk menghapus dokumen
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        del documents[doc_id]
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error in delete_document: {error_msg}")
        print(trace)
        raise HTTPException(status_code=500, detail=error_msg)

# Endpoint untuk health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Jalankan server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)