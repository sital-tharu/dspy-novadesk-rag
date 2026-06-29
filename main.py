# main.py
import dspy
from src.pipeline import NovaRAGPipeline, test_pipeline

# ── Configure Ollama ───────────────────────────────────────────────
lm = dspy.LM("ollama/llama3", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# ── Run Pipeline Test ──────────────────────────────────────────────
test_pipeline()