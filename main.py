# main.py
import dspy
from src.pipeline import NovaRAGPipeline
from src.dataset import get_dataset
from src.evaluate import evaluate_detailed, combined_metric
from src.optimize import optimize_pipeline, inspect_demos
from src.save_load import save_pipeline

# ── Configure ──────────────────────────────────────────────────────
lm = dspy.LM("ollama/llama3", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# ── Load Data ──────────────────────────────────────────────────────
trainset, devset = get_dataset()
print(f"Trainset : {len(trainset)} examples")
print(f"Devset   : {len(devset)} examples\n")

# ── Baseline ───────────────────────────────────────────────────────
print("── Baseline Evaluation ──")
baseline_pipeline = NovaRAGPipeline(k=3)
baseline_score    = evaluate_detailed(baseline_pipeline, devset)

# ── Optimize ───────────────────────────────────────────────────────
optimized_pipeline = optimize_pipeline(trainset)

# ── Inspect Demos ──────────────────────────────────────────────────
inspect_demos(optimized_pipeline)

# ── Optimized Evaluation ───────────────────────────────────────────
print("\n── Optimized Evaluation ──")
optimized_score = evaluate_detailed(optimized_pipeline, devset)

# ── Save ───────────────────────────────────────────────────────────
save_pipeline(optimized_pipeline)

# ── Final Comparison ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'FINAL RESULTS':^60}")
print(f"{'='*60}")
print(f"  Baseline  : {baseline_score:.1f}%")
print(f"  Optimized : {optimized_score:.1f}%")
print(f"  Delta     : {optimized_score - baseline_score:+.1f}%")
print(f"{'='*60}")