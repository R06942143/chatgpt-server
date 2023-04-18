import chromadb

from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        chroma_api_impl="rest",
        chroma_server_host="3.223.141.233",
        chroma_server_http_port=8000,
    )
)
collection = client.get_collection(name="conrad")
collection.delete
