import chromadb
from chromadb.utils import embedding_functions
import dspy

# ── Config ──────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "novadesk_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"