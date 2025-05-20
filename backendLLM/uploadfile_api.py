import os
import tempfile
import uuid
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lmstudio as lms
import PyPDF2
import docx
import pandas as pd
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Inisialisasi FastAPI
app = FastAPI(title="RAG Document API dengan LMStudio")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi model LLM dan Embedding
llm_model = lms.llm("qwen2.5-7b-instruct-1m")
embedding_model = lms.embedding_model("nomic-embed-text-v1.5")

# Penyimpanan dokumen dan embeddings
document_store = {}
faiss_index = faiss.IndexFlatL2(768)  # Dimensi embedding nomic-embed-text-v1.5

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    
# Fungsi untuk mengekstrak teks dari berbagai jenis file
def extract_text_from_file(file_path: str, file_type: str) -> str:
    try:
        if file_type == "pdf":
            text = ""
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        elif file_type == "docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        
        elif file_type in ["xlsx", "xls", "csv"]:
            if file_type == "csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Konversi DataFrame ke format teks
            text = df.to_string(index=False)
            return text
        
        else:
            raise ValueError(f"Format file tidak didukung: {file_type}")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal mengekstrak teks: {str(e)}")

# Endpoint untuk upload dokumen
@app.post("/upload-document", response_model=Dict)
async def upload_document(file: UploadFile = File(...)):
    # Periksa tipe file
    file_extension = file.filename.split(".")[-1].lower()
    supported_extensions = ["pdf", "txt", "docx", "xlsx", "xls", "csv"]
    
    if file_extension not in supported_extensions:
        raise HTTPException(status_code=400, detail=f"Format file tidak didukung. Hanya mendukung: {', '.join(supported_extensions)}")
    
    # Buat file temporary untuk menyimpan upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Ekstrak teks dari file
        document_text = extract_text_from_file(temp_file_path, file_extension)
        
        # Bagi teks menjadi chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(document_text)
        
        # Buat embeddings untuk setiap chunk
        embeddings = []
        for chunk in chunks:
            embedding = embedding_model.embed(chunk)
            embeddings.append(embedding)
        
        # Simpan dokumen dan embeddings
        document_id = str(uuid.uuid4())
        document_store[document_id] = {
            "filename": file.filename,
            "chunks": chunks,
            "embeddings": embeddings
        }
        
        # Tambahkan ke FAISS index
        for i, embedding in enumerate(embeddings):
            faiss_index.add(np.array([embedding], dtype=np.float32))
            
        # Hapus file temporary
        os.unlink(temp_file_path)
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "num_chunks": len(chunks)
        }
        
    except Exception as e:
        # Hapus file temporary jika terjadi error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error saat memproses dokumen: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Embed pertanyaan user
        question_embedding = embedding_model.embed(request.question)
        
        relevant_chunks = []
        
        # Jika tidak ada dokumen yang terupload, gunakan model langsung
        if len(document_store) == 0:
            # Dapatkan jawaban dari model
            model_response = llm_model.respond(request.question)
            
            # Pastikan kita mengembalikan string, bukan objek PredictionResult
            if hasattr(model_response, 'content'):
                answer = model_response.content  # Ekstrak konten jika respons adalah objek
            else:
                answer = str(model_response)  # Konversi ke string jika bukan objek
                
            return {"answer": answer, "sources": []}
            
        # Jika ada dokumen, cari di semua dokumen yang tersedia
        for doc_id, doc_data in document_store.items():
            # Cari chunks yang paling relevan dengan pertanyaan
            for i, emb in enumerate(doc_data["embeddings"]):
                # Hitung similarity dengan dot product
                similarity = np.dot(question_embedding, emb)
                
                # Tambahkan ke daftar jika similarity cukup tinggi
                if similarity > 0.65:  # Threshold similarity
                    relevant_chunks.append({
                        "text": doc_data["chunks"][i],
                        "similarity": float(similarity),
                        "source": doc_data["filename"]
                    })
        
        # Urutkan berdasarkan similarity
        relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Ambil top 5 chunks
        top_chunks = relevant_chunks[:5]
        
        if not top_chunks:
            # Tidak ada chunks yang relevan, gunakan model langsung
            model_response = llm_model.respond(request.question)
            
            # Ekstrak konten jika perlu
            if hasattr(model_response, 'content'):
                answer = model_response.content
            else:
                answer = str(model_response)
                
            return {"answer": answer, "sources": []}
        else:
            # Buat prompt dengan konteks dari chunks yang relevan
            context = "\n\n".join([chunk["text"] for chunk in top_chunks])
            prompt = f"""Berdasarkan informasi dari dokumen berikut:

{context}

Pertanyaan: {request.question}

Jawaban (sertakan sumber informasi yang digunakan):"""
            
            # Dapatkan jawaban dari model
            model_response = llm_model.respond(prompt)
            
            # Ekstrak konten jika perlu
            if hasattr(model_response, 'content'):
                answer = model_response.content
            else:
                answer = str(model_response)
            
            # Dapatkan sumber unik
            sources = list(set([chunk["source"] for chunk in top_chunks]))
            
            return {"answer": answer, "sources": sources}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat chatting: {str(e)}")
    
# Endpoint untuk mendapatkan daftar dokumen yang sudah diupload
@app.get("/documents", response_model=Dict)
async def get_documents():
    result = {}
    for doc_id, doc_data in document_store.items():
        result[doc_id] = {
            "filename": doc_data["filename"],
            "num_chunks": len(doc_data["chunks"])
        }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)