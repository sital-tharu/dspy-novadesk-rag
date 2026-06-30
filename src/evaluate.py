# src/evaluate.py
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import dspy
from dspy.evaluate import Evaluate
from sentence_transformers import SentenceTransformer, util

# ── Load Embedding Model (same one used in retriever) ───────────────
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Metric 1: Exact Keyword Match ───────────────────────────────────
def keyword_metric(example, prediction, trace=None):
    expected  = example.answer.lower()
    predicted = prediction.answer.lower()

    # Handle common abbreviations
    substitutions = {
        "2fa" : "two-factor authentication",
        "sso" : "single sign-on",
        "csat": "customer satisfaction",
    }
    for abbr, full in substitutions.items():
        predicted = predicted.replace(full, abbr)
        expected  = expected.replace(full, abbr)

    keywords = [w for w in expected.split() if len(w) > 4]

    if not keywords:
        return expected in predicted

    matches = sum(1 for k in keywords if k in predicted)
    score   = matches / len(keywords)

    return score >= 0.5


# ── Metric 2: Semantic Similarity ───────────────────────────────────
def semantic_metric(example, prediction, trace=None):
    """
    Uses sentence embeddings to measure meaning similarity.
    More robust than keyword matching.
    Score >= 0.75 = semantically similar answer.
    """
    expected  = example.answer.strip()
    predicted = prediction.answer.strip()

    if not predicted or predicted.lower() == "i don't have information about that.":
        # Check if expected answer is also unknown
        return expected.lower() in ["unknown", "not available", "n/a"]

    # Embed both answers
    emb_expected  = _embedder.encode(expected,  convert_to_tensor=True)
    emb_predicted = _embedder.encode(predicted, convert_to_tensor=True)

    # Cosine similarity
    similarity = util.cos_sim(emb_expected, emb_predicted).item()

    return similarity >= 0.65   # threshold for "correct"


# ── Metric 3: Combined (keyword + semantic) ─────────────────────────
def combined_metric(example, prediction, trace=None):
    """
    Passes if EITHER keyword OR semantic metric passes.
    More lenient — good for optimization.
    """
    return (
        keyword_metric(example, prediction, trace) or
        semantic_metric(example, prediction, trace)
    )


# ── Evaluator ───────────────────────────────────────────────────────
def evaluate_pipeline(pipeline, devset, metric=combined_metric):
    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=True,
    )
    score = evaluator(pipeline)
    return score


# ── Detailed Evaluation (shows per-question results) ────────────────
def evaluate_detailed(pipeline, devset):
    """
    Runs pipeline on each question and shows
    expected vs predicted with pass/fail.
    """
    print(f"\n{'='*70}")
    print(f"{'DETAILED EVALUATION':^70}")
    print(f"{'='*70}")

    correct = 0

    for i, example in enumerate(devset):
        result    = pipeline(question=example.question)
        predicted = result.answer
        expected  = example.answer

        kw_pass  = keyword_metric(example, result)
        sem_pass = semantic_metric(example, result)
        passed   = kw_pass or sem_pass

        if passed:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"\n{status} Q{i+1}: {example.question}")
        print(f"   Expected : {expected}")
        print(f"   Got      : {predicted}")
        print(f"   Keyword  : {'✅' if kw_pass else '❌'}  "
              f"Semantic : {'✅' if sem_pass else '❌'}")

    accuracy = (correct / len(devset)) * 100
    print(f"\n{'='*70}")
    print(f"Final Score: {correct}/{len(devset)} = {accuracy:.1f}%")
    print(f"{'='*70}")
    return accuracy