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

# ── DSPy-Compatible Retriever ────────────────────────────────────────
class NovaRetriever(dspy.Retrieve):
    """
    Custom DSPy retriever backed by ChromaDB.
    Takes a query string, returns top-k relevant passages.
    """

    def __init__(self, k=3):
        super().__init__(k=k)
        self.collection = get_collection()
        self.k          = k

    def forward(self, query: str) -> dspy.Prediction:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.k
        )

        # Format as DSPy passages
        passages = results["documents"][0]

        return dspy.Prediction(passages=passages)
    
# ── Test Function ────────────────────────────────────────────────────
def test_retriever():
    retriever = NovaRetriever(k=3)

    test_queries = [
        "How much does the Pro plan cost?",
        "Can I downgrade my plan?",
        "What happens to my data if I cancel?",
        "How many agents can I add on Starter plan?",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        result = retriever(query)
        for i, passage in enumerate(result.passages):
            print(f"\n  Passage {i+1}: {passage[:150]}...")


if __name__ == "__main__":
    test_retriever()