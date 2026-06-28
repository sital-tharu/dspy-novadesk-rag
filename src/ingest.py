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


# ── Step 2: Split into Sections first, then Chunks ─────────────────
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping word-level chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    # Remove markdown headers and clean up
    text = re.sub(r"#{1,3} ", "", text)       # remove # ## ###
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse blank lines
    text = text.strip()

    # Split into words
    words = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append({
            "id"   : f"chunk_{len(chunks):03d}",
            "text" : chunk,
            "start": start,
            "end"  : min(end, len(words))
        })
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap

    print(f"✅ Created {len(chunks)} chunks "
          f"(size={chunk_size} words, overlap={overlap} words)")
    return chunks
