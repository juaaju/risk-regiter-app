import requests

response = requests.post(
    "http://localhost:8001/rag",
    json={
        "query": "Jelaskan apa saja isi dokumen R-002",
        "embedding_table": "embeddrisk_mng",
        "source_table": "risks",
        "limit": 3
    }
)
print(response.json())