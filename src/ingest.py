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

# ── Step 3: Save Chunks to JSON ────────────────────────────────────
def save_chunks(chunks, path=CHUNKS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"✅ Saved chunks to {path}")

# ── Step 4: Load into ChromaDB ─────────────────────────────────────
def load_into_chromadb(chunks, db_path=CHROMA_DB_PATH):
    """
    Embeds each chunk using sentence-transformers
    and stores in ChromaDB for vector search.
    """
    # Use local sentence-transformers model (no API key needed)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"   # small, fast, good quality
    )

    # Init ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🔄 Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create fresh collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    # Add chunks
    collection.add(
        ids       = [c["id"]   for c in chunks],
        documents = [c["text"] for c in chunks],
    )

    print(f"✅ Loaded {len(chunks)} chunks into ChromaDB")
    print(f"✅ Collection '{COLLECTION_NAME}' ready at '{db_path}/'")
    return collection

# ── Step 5: Test Retrieval ─────────────────────────────────────────
def test_retrieval(query="How much does the Pro plan cost?"):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    print(f"\n── Test Retrieval ──")
    print(f"Query: {query}\n")
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(f"  ID   : {results['ids'][0][i]}")
        print(f"  Text : {doc[:200]}...")
        print()


# ── Main ───────────────────────────────────────────────────────────
def run_ingestion():
    print("🚀 Starting ingestion pipeline...\n")

    text       = load_text()
    chunks     = split_into_chunks(text)
    save_chunks(chunks)
    load_into_chromadb(chunks)
    test_retrieval()

    print("🎉 Ingestion complete! ChromaDB is ready.")


if __name__ == "__main__":
    run_ingestion()