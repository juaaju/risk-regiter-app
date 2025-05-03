"""
Contoh penggunaan modul postgres_embedder.py
"""

from postgres_embedder import PostgresEmbedder
import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def embed_documents_from_table(
    source_table: str,
    target_table: str,
    text_columns: list,
    id_column: str = 'id',
    db_config: dict = None
):
    """
    Membuat embedding untuk dokumen dari tabel PostgreSQL
    
    Args:
        source_table: Nama tabel sumber data
        target_table: Nama tabel untuk menyimpan embedding
        text_columns: List kolom teks yang akan digabungkan dan di-embed
        id_column: Nama kolom ID
        db_config: Konfigurasi database (opsional)
    """
    # Siapkan konfigurasi database
    if db_config is None:
        db_config = {}
    
    # Buat instance PostgresEmbedder
    embedder = PostgresEmbedder(**db_config)
    
    try:
        with embedder:
            # TAMBAHKAN KODE INI DI SINI
            # Cek apakah user meminta semua kolom
            # Dapatkan juga tipe data kolom untuk menangani casting
            if text_columns == ["ALL"]:
                all_columns_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{source_table.lower()}'
                AND column_name != '{id_column.lower()}'
                """
                columns_data = embedder.fetch_data(all_columns_query)
                text_columns = [col['column_name'] for col in columns_data]
                
                # Buat dictionary dari nama kolom ke tipe data
                column_types = {col['column_name']: col['data_type'] for col in columns_data}
                print(f"Using all columns: {text_columns}")
            
            # Buat query dengan konversi eksplisit ke text untuk setiap kolom
            columns_concat_parts = []
            for col in text_columns:
                # Konversi eksplisit ke text untuk semua jenis data
                columns_concat_parts.append(f"COALESCE(CAST({col} AS TEXT), '')")
            
            columns_concat = " || ' ' || ".join(columns_concat_parts)
            
            query = f"""
            SELECT 
                {id_column},
                {columns_concat} AS combined_text
                {', ' + ', '.join(text_columns) if text_columns else ''}
            FROM 
                {source_table}
            """
            print(f"Executing query: {query}")
            
            # Ambil data dan buat embedding
            data = embedder.create_embeddings_for_query(
                query=query,
                text_column='combined_text',
                id_column=id_column
            )
            
            print(f"Created embeddings for {len(data)} documents")
            
            # Simpan embedding ke tabel target
            embedder.save_embeddings_to_table(
                data=data,
                table_name=target_table,
                id_column=id_column,
                create_table_if_not_exists=True
            )
            
            print(f"Successfully saved embeddings to {target_table}")
    
    except Exception as e:
        print(f"Error in embed_documents_from_table: {e}")
        raise

def search_similar_documents(
    query: str,
    embedding_table: str,
    source_table: str,
    id_column: str = 'id',
    limit: int = 5,
    db_config: dict = None
):
    """
    Mencari dokumen yang mirip dengan query
    
    Args:
        query: Teks query
        embedding_table: Nama tabel yang berisi embedding
        source_table: Nama tabel sumber data asli
        id_column: Nama kolom ID
        limit: Jumlah hasil yang dikembalikan
        db_config: Konfigurasi database (opsional)
    """
    if db_config is None:
        db_config = {}
    
    embedder = PostgresEmbedder(**db_config)
    
    try:
        with embedder:
            # Cari dokumen yang mirip
            similar_docs = embedder.find_similar_documents(
                query_text=query,
                table_name=embedding_table,
                id_column=id_column,
                limit=limit
            )
            
            if not similar_docs:
                print("No similar documents found")
                return []
            
            print(f"Found {len(similar_docs)} similar documents")
            
            # Ambil data dari tabel sumber
            ids = [doc[id_column] for doc in similar_docs]
            placeholders = ','.join(['%s'] * len(ids))
            
            # Buat query untuk mengambil data dokumen
            docs_query = f"""
            SELECT * FROM {source_table}
            WHERE {id_column} IN ({placeholders})
            """
            
            documents = embedder.fetch_data(docs_query, tuple(ids))
            
            # Gabungkan hasil dengan similarity score
            id_to_similarity = {doc[id_column]: doc['similarity'] for doc in similar_docs}
            for doc in documents:
                doc['similarity'] = id_to_similarity.get(doc[id_column], 0)
            
            # Urutkan berdasarkan similarity
            documents.sort(key=lambda x: x['similarity'], reverse=True)
            
            return documents
    
    except Exception as e:
        print(f"Error in search_similar_documents: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postgres Embedder CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Parser untuk command 'embed'
    embed_parser = subparsers.add_parser('embed', help='Create embeddings')
    embed_parser.add_argument('--source', required=True, help='Source table name')
    embed_parser.add_argument('--target', required=True, help='Target table name')
    embed_parser.add_argument('--columns', required=True, nargs='+', help='Text columns to embed')
    embed_parser.add_argument('--id-column', default='id', help='ID column name')
    
    # Parser untuk command 'search'
    search_parser = subparsers.add_parser('search', help='Search similar documents')
    search_parser.add_argument('--query', required=True, help='Search query')
    search_parser.add_argument('--embedding-table', required=True, help='Table containing embeddings')
    search_parser.add_argument('--source-table', required=True, help='Source data table')
    search_parser.add_argument('--id-column', default='id', help='ID column name')
    search_parser.add_argument('--limit', type=int, default=5, help='Max number of results')
    
    args = parser.parse_args()
    
    # Konfigurasi database dari environment variables
    db_config = {
        'db_host': os.environ.get('DB_HOST'),
        'db_port': os.environ.get('DB_PORT'),
        'db_name': os.environ.get('DB_NAME'),
        'db_user': os.environ.get('DB_USER'),
        'db_password': os.environ.get('DB_PASSWORD'),
    }
    
    # Filter None values
    db_config = {k: v for k, v in db_config.items() if v is not None}
    
    if args.command == 'embed':
        embed_documents_from_table(
            source_table=args.source,
            target_table=args.target,
            text_columns=args.columns,
            id_column=args.id_column,
            db_config=db_config
        )
    
    elif args.command == 'search':
        results = search_similar_documents(
            query=args.query,
            embedding_table=args.embedding_table,
            source_table=args.source_table,
            id_column=args.id_column,
            limit=args.limit,
            db_config=db_config
        )
        
        # Tampilkan hasil
        print("\n=== Search Results ===")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} (Similarity: {doc['similarity']:.4f}) ---")
            for key, value in doc.items():
                if key != 'embedding' and key != 'similarity':
                    if isinstance(value, str) and len(value) > 100:
                        print(f"{key}: {value[:100]}...")
                    else:
                        print(f"{key}: {value}")
    
    else:
        parser.print_help()
        