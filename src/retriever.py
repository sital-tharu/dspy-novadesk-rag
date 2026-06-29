import chromadb
from chromadb.utils import embedding_functions
import dspy

# ── Config ──────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "novadesk_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB Client ─────────────────────────────────────────────────
def get_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    return collection