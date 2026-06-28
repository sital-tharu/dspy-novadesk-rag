# src/ingest.py
import os
import json
import re
import chromadb
from chromadb.utils import embedding_functions

# ── Config ─────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = "data/raw/knowledge_base.txt"
CHUNKS_PATH         = "data/processed/chunks.json"
CHROMA_DB_PATH      = "chroma_db"
COLLECTION_NAME     = "novadesk_kb"
CHUNK_SIZE          = 200   # words per chunk
CHUNK_OVERLAP       = 40    # words overlap between chunks


# ── Step 1: Load Raw Text ───────────────────────────────────────────
def load_text(path=KNOWLEDGE_BASE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Loaded knowledge base: {len(text)} characters")
    return text
