"""
Module untuk mengambil data dari PostgreSQL dan membuat embedding dengan LM Studio
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import lmstudio as lms
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union

class PostgresEmbedder:
    """
    Kelas untuk mengambil data dari PostgreSQL dan membuat embedding dengan LM Studio
    """
    
    def __init__(
        self, 
        db_host: str = None,
        db_port: int = None, 
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
        embedding_model: str = "nomic-embed-text-v1.5",
        connection_string: str = None
    ):
        """
        Inisialisasi koneksi ke database dan model embedding

        Args:
            db_host: Host database PostgreSQL (default: dari env DB_HOST)
            db_port: Port database PostgreSQL (default: dari env DB_PORT)
            db_name: Nama database PostgreSQL (default: dari env DB_NAME)
            db_user: Username database PostgreSQL (default: dari env DB_USER)
            db_password: Password database PostgreSQL (default: dari env DB_PASSWORD)
            embedding_model: Nama model embedding di LM Studio
            connection_string: String koneksi PostgreSQL (jika disediakan, parameter db lainnya diabaikan)
        """
        # Gunakan parameter yang diberikan atau ambil dari environment variables
        self.db_host = db_host or os.environ.get('DB_HOST', 'localhost')
        self.db_port = db_port or int(os.environ.get('DB_PORT', '5432'))
        self.db_name = db_name or os.environ.get('DB_NAME', 'postgres')
        self.db_user = db_user or os.environ.get('DB_USER', 'postgres')
        self.db_password = db_password or os.environ.get('DB_PASSWORD', '')
        
        self.connection_string = connection_string
        self.embedding_model_name = embedding_model
        self._conn = None
        self._embedding_model = None
        
    def connect(self) -> None:
        """Membuat koneksi ke database PostgreSQL"""
        try:
            if self.connection_string:
                self._conn = psycopg2.connect(self.connection_string)
            else:
                self._conn = psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    dbname=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
            print(f"Connected to PostgreSQL database: {self.db_name}")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise
    
    def disconnect(self) -> None:
        """Menutup koneksi database"""
        if self._conn:
            self._conn.close()
            self._conn = None
            print("Disconnected from PostgreSQL database")
    
    def _get_connection(self):
        """Helper untuk mendapatkan koneksi aktif"""
        if not self._conn:
            self.connect()
        return self._conn
    
    def _get_embedding_model(self):
        """Helper untuk memuat model embedding jika belum diinisialisasi"""
        if not self._embedding_model:
            try:
                self._embedding_model = lms.embedding_model(self.embedding_model_name)
                print(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                raise
        return self._embedding_model
    
    def create_embedding(self, text: str):
        """
        Membuat embedding untuk teks menggunakan model LM Studio
        
        Args:
            text: Teks yang akan di-embed
                
        Returns:
            Vector embedding
        """
        model = self._get_embedding_model()
        embedding = model.embed(text)
        
        # Periksa tipe data embedding
        if isinstance(embedding, np.ndarray):
            return embedding  # Sudah berupa numpy array, tidak perlu dikonversi
        elif isinstance(embedding, list):
            return np.array(embedding)  # Konversi ke numpy array jika masih berupa list
        else:
            # Tipe data lain, coba konversi ke numpy array
            return np.array(embedding)
    
    def fetch_data(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Mengambil data dari PostgreSQL menggunakan query SQL
        
        Args:
            query: Query SQL untuk dieksekusi
            params: Parameter untuk query
            
        Returns:
            List dari dictionary hasil query
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            print(f"Error executing query: {e}")
            conn.rollback()
            raise

    def create_embeddings_for_query(
        self, 
        query: str, 
        params: tuple = None,
        text_column: str = 'content',
        id_column: str = 'id'
    ) -> List[Dict[str, Any]]:
        """
        Mengambil data dari database dan membuat embedding untuk setiap baris
        """
        data = self.fetch_data(query, params)
        result = []
        
        for item in data:
            try:
                text = item.get(text_column, '')
                if text:
                    embedding = self.create_embedding(text)
                    
                    # Simpan embedding dalam format yang tepat
                    if isinstance(embedding, np.ndarray):
                        item['embedding'] = embedding.tolist()  # Konversi numpy array ke list
                    elif isinstance(embedding, list):
                        item['embedding'] = embedding  # Sudah berupa list, tidak perlu konversi
                    else:
                        # Coba konversi apa pun ke list
                        item['embedding'] = list(embedding)
                        
                    result.append(item)
                else:
                    print(f"Warning: Empty text for item with ID {item.get(id_column, 'unknown')}")
            except Exception as e:
                print(f"Error creating embedding for item {item.get(id_column, 'unknown')}: {e}")
        
        return result
    
    def save_embeddings_to_table(
        self,
        data: List[Dict[str, Any]],
        table_name: str,
        id_column: str = 'id',
        embedding_column: str = 'embedding',
        create_table_if_not_exists: bool = True,
        vector_dimension: int = None,
        use_pgvector: bool = True
    ) -> None:
        """
        Menyimpan embedding ke tabel PostgreSQL
        
        Args:
            data: List dari dictionary dengan data dan embedding
            table_name: Nama tabel untuk menyimpan embedding
            id_column: Nama kolom ID
            embedding_column: Nama kolom untuk menyimpan vector embedding
            create_table_if_not_exists: Buat tabel jika belum ada
            vector_dimension: Dimensi vector untuk pgvector (opsional)
            use_pgvector: Gunakan tipe data vector dari pgvector (True) atau REAL[] (False)
        """
        conn = self._get_connection()
        
        try:
            with conn.cursor() as cur:
                # Cek apakah ekstensi pgvector sudah terinstal jika use_pgvector=True
                if use_pgvector:
                    try:
                        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                        pgvector_installed = cur.fetchone() is not None
                        
                        if not pgvector_installed:
                            try:
                                # Coba instal ekstensi pgvector
                                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                                conn.commit()
                                print("Successfully installed pgvector extension")
                                pgvector_installed = True
                            except Exception as e:
                                print(f"Could not install pgvector extension: {e}")
                                print("Falling back to REAL[] data type")
                                use_pgvector = False
                                conn.rollback()
                    
                    except Exception as e:
                        print(f"Error checking for pgvector extension: {e}")
                        print("Falling back to REAL[] data type")
                        use_pgvector = False
                
                # Tentukan dimensi vector jika tidak diberikan
                if vector_dimension is None and data and embedding_column in data[0]:
                    vector_dimension = len(data[0][embedding_column])
                    print(f"Detected vector dimension: {vector_dimension}")
                
                # Buat tabel jika diperlukan
                if create_table_if_not_exists:
                    if use_pgvector and pgvector_installed:
                        # Gunakan tipe data vector dari pgvector
                        create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            {id_column} TEXT PRIMARY KEY,
                            {embedding_column} vector({vector_dimension})
                        )
                        """
                    else:
                        # Gunakan array biasa jika pgvector tidak tersedia
                        create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            {id_column} TEXT PRIMARY KEY,
                            {embedding_column} REAL[]
                        )
                        """
                    cur.execute(create_table_query)
                
                # Insert atau update data
                for item in data:
                    if id_column in item and embedding_column in item:
                        if use_pgvector and pgvector_installed:
                            # Untuk pgvector, gunakan sintaks vector
                            embedding_str = '[' + ','.join(str(x) for x in item[embedding_column]) + ']'
                            # Upsert query untuk pgvector
                            upsert_query = f"""
                            INSERT INTO {table_name} ({id_column}, {embedding_column})
                            VALUES (%s, %s::vector)
                            ON CONFLICT ({id_column}) DO UPDATE
                            SET {embedding_column} = EXCLUDED.{embedding_column}
                            """
                        else:
                            # Untuk REAL[], gunakan sintaks array
                            embedding_str = '{' + ','.join(str(x) for x in item[embedding_column]) + '}'
                            # Upsert query untuk array
                            upsert_query = f"""
                            INSERT INTO {table_name} ({id_column}, {embedding_column})
                            VALUES (%s, %s)
                            ON CONFLICT ({id_column}) DO UPDATE
                            SET {embedding_column} = EXCLUDED.{embedding_column}
                            """
                        
                        cur.execute(upsert_query, (item[id_column], embedding_str))
                
                conn.commit()
                
                # Cek apakah tabel berhasil dibuat dan data berhasil disimpan
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                
                if use_pgvector and pgvector_installed:
                    print(f"Saved {len(data)} embeddings to table {table_name} using pgvector data type")
                else:
                    print(f"Saved {len(data)} embeddings to table {table_name} using REAL[] data type")
                print(f"Total records in table: {count}")
                
        except Exception as e:
            conn.rollback()
            print(f"Error saving embeddings to table: {e}")
            raise

    def find_similar_documents(
        self,
        query_text: str,
        table_name: str,
        embedding_column: str = 'embedding',
        id_column: str = 'id',
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Mencari dokumen yang paling mirip dengan query menggunakan cosine similarity di pgvector
        
        Args:
            query_text: Teks query yang akan dicari kesamaannya
            table_name: Nama tabel yang berisi embedding
            embedding_column: Nama kolom vector embedding
            id_column: Nama kolom ID
            limit: Jumlah hasil yang dikembalikan
            
        Returns:
            List dari dictionary dengan ID dokumen dan similarity score
        """
        # Create embedding for query text
        query_embedding = self.create_embedding(query_text)
        
        # Format vector untuk query SQL - gunakan sintaks yang tepat untuk pgvector
        vector_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
        
        conn = self._get_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Gunakan string SQL yang dibentuk lengkap (bukan parameter)
                # untuk memastikan tipe data vector ditangani dengan benar
                cosine_query = f"""
                SELECT 
                    {id_column},
                    1 - ({embedding_column} <=> '{vector_str}'::vector) AS similarity
                FROM 
                    {table_name}
                ORDER BY 
                    similarity DESC
                LIMIT {limit}
                """
                
                try:
                    # Debug
                    print("Executing query:")
                    print(cosine_query)
                    
                    # Eksekusi query
                    cur.execute(cosine_query)
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
                except Exception as e:
                    print(f"Error using cosine similarity: {e}")
                    
                    # Coba gunakan L2 distance sebagai alternatif
                    l2_query = f"""
                    SELECT 
                        {id_column},
                        1 / (1 + ({embedding_column} <-> '{vector_str}'::vector)) AS similarity
                    FROM 
                        {table_name}
                    ORDER BY 
                        similarity DESC
                    LIMIT {limit}
                    """
                    
                    print("Trying L2 distance query:")
                    print(l2_query)
                    
                    cur.execute(l2_query)
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            raise
    
    def __enter__(self):
        """Context manager method untuk memudahkan penggunaan dengan 'with'"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager method untuk otomatis menutup koneksi"""
        self.disconnect()

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh penggunaan dasar
    embedder = PostgresEmbedder()
    
    with embedder:
        # Contoh 1: Ambil data dan buat embedding
        data = embedder.create_embeddings_for_query(
            "SELECT id, title, content FROM documents",
            text_column='content'
        )
        
        # Contoh 2: Simpan embedding ke tabel
        embedder.save_embeddings_to_table(data, 'document_embeddings')
        
        # Contoh 3: Cari dokumen yang mirip
        similar_docs = embedder.find_similar_documents(
            "Bagaimana cara mengelola risiko keselamatan kerja?",
            'document_embeddings'
        )
        print(f"Found {len(similar_docs)} similar documents")
        for doc in similar_docs:
            print(f"Document ID: {doc['id']}, Similarity: {doc['similarity']}")