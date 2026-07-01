# 🧠 DSPy NovaDesk RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built with [DSPy](https://dspy.ai),
answering customer support questions for **NovaDesk** — a fictional SaaS
support platform — using a local knowledge base, ChromaDB, and Ollama.

No API keys. No cloud costs. Fully local inference.

## What It Does

Given a customer support question like *"Does NovaDesk integrate with
Shopify?"*, the pipeline:
1. Retrieves the most relevant passages from a custom knowledge base
2. Reasons over that context using Chain-of-Thought
3. Generates a grounded, concise answer

If the knowledge base doesn't contain the answer, it says so instead
of hallucinating.

## Tech Stack

- **DSPy** — LLM programming and optimization framework
- **Ollama (llama3)** — local LLM inference
- **ChromaDB** — local vector database for retrieval
- **sentence-transformers (all-MiniLM-L6-v2)** — embeddings, also used
  for the semantic evaluation metric

## Project Structure
src/
├── ingest.py      # Chunks knowledge_base.txt and loads it into ChromaDB
├── retriever.py   # DSPy-compatible retriever backed by ChromaDB
├── signature.py   # GenerateAnswer signature (context + question -> answer)
├── pipeline.py     # Full RAG module: retrieve -> reason -> answer
├── dataset.py     # 30 Q&A pairs (20 train / 10 dev) across all KB sections
├── evaluate.py    # Keyword + semantic similarity metrics
├── optimize.py    # BootstrapFewShot optimizer + demo inspection
└── save_load.py   # Save/load optimized pipeline state
data/
└── raw/knowledge_base.txt   # Fictional NovaDesk product documentation
main.py            # Runs baseline eval -> optimization -> optimized eval
spot_check.py      # Manual spot-check: runs 5 questions through optimized pipeline

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/sital-tharu/dspy-novadesk-rag
cd dspy-novadesk-rag

# 2. Set up environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Start Ollama and pull the model
ollama serve
ollama pull llama3

# 4. Ingest the knowledge base into ChromaDB
python src/ingest.py

# 5. Run the full pipeline (baseline -> optimize -> compare)
python main.py

# 6. (Optional) Run manual spot-check on 5 sample questions
python spot_check.py
```

## Results

| Stage     | Accuracy | Correct |
|-----------|----------|---------|
| Baseline  | 90.0%    | 9/10    |
| Optimized | 100.0%   | 10/10   |
| **Delta** | **+10.0%** | +1   |

Unlike a simpler classification task, this dataset was hard enough
that BootstrapFewShot produced a real, measurable improvement instead
of a wash.

### What Optimization Fixed

**Shopify integration question** — the baseline pipeline retrieved
generic "Pro plan" passages and hedged: *"it does not specifically
mention Shopify... unclear whether NovaDesk integrates with Shopify."*
After optimization, the few-shot demos taught the model to read the
integrations list more carefully, and it correctly answered: *"Yes,
NovaDesk integrates with Shopify."*

## Key Learnings

### Chunking matters more than chunk count
The first ingestion pass used 200-word chunks and produced only 9
chunks total — each one blending multiple unrelated topics (all three
pricing plans crammed into one chunk). Retrieval was unreliable as a
result. Dropping to 80-word chunks with 15-word overlap, split by
section first, produced 25 cleaner, single-topic chunks and immediately
improved retrieval precision.

### Exact-match metrics don't work for generative answers
A correct answer rarely matches the reference word-for-word ("Data is
retained for 60 days" vs "Your data will be kept for 60 days then
permanently deleted"). This project uses two complementary metrics —
keyword overlap and embedding cosine similarity — and counts an answer
correct if either one passes.

### Small local models can echo prompt templates verbatim
With longer prompts (5 retrieved passages + 4 few-shot demos), llama3
occasionally returned the literal string `"{answer}"` instead of
generating real content — a known reliability quirk of smaller models
under long context. Fixed with a simple retry loop in `forward()`:
regenerate up to twice if the output still contains template braces,
then fall back to a safe default.

### `dspy.Retrieve` doesn't support default serialization
Calling `.save()` on a pipeline that wraps a custom `dspy.Retrieve`
subclass throws `TypeError: Retrieve.dump_state() got an unexpected
keyword argument 'json_mode'`. Worked around this by catching the
error and falling back to saving just the optimized few-shot demos as
JSON, which is the part that actually matters for reuse.

### BootstrapFewShot demo selection is order-sensitive
As seen in the earlier news classifier project, the optimizer walks
the trainset in order and stops once it hits `max_bootstrapped_demos`
correct traces — so an unshuffled, category-grouped trainset biases
which demos get picked. Worth keeping in mind even though it wasn't
the dominant issue in this run.

## What Didn't Work (and Why)

- An early version used `max_bootstrapped_demos=6` with `k=5`
  retrieval, which produced prompts long enough to reliably trigger
  the template-echo bug above. Reducing back toward `k=3-4` with retry
  logic was more reliable than chasing a larger context window.
- Increasing the semantic similarity threshold to 0.75 caused false
  negatives on genuinely correct paraphrased answers (e.g. "2FA" vs
  "Two-Factor Authentication"). Lowering to 0.65 and adding an
  abbreviation-substitution step to the keyword metric fixed this.

## Next Steps

- [ ] Try MIPROv2 optimizer and compare against BootstrapFewShot
- [ ] Add multi-hop questions that require combining 3+ chunks
- [ ] Expand knowledge base with intentionally conflicting/outdated
      sections to test retrieval robustness
- [ ] Swap in a larger model (or hosted API) to see if the
      template-echo bug disappears entirely
- [ ] Build a simple Gradio chat UI on top of the optimized pipeline

## Related Project

[`dspy-news-classifier`](https://github.com/sital-tharu/dspy-news-classifier) —
an earlier DSPy project (text classification) that explores why
optimization *doesn't* always help, as a contrast to this one.