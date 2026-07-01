# spot_check.py
import dspy
from src.pipeline import NovaRAGPipeline
from src.dataset import get_dataset
from src.optimize import optimize_pipeline

lm = dspy.LM("ollama/llama3", api_base="http://localhost:11434")
dspy.configure(lm=lm)

trainset, _ = get_dataset()
pipeline = optimize_pipeline(trainset)

questions = [
    ("Does the Starter plan support live chat?",               "No"),
    ("What happens to my data if I cancel?",                   "Retained 60 days then deleted"),
    ("Can I get a refund after 2 months on a monthly plan?",   "No"),
    ("Is PayPal accepted?",                                    "No"),
    ("Which plans include Shopify integration?",               "Pro and Enterprise"),
]

for i, (question, expected) in enumerate(questions, 1):
    print(f"\n{'='*65}")
    print(f"Q{i}: {question}")
    print(f"    Expected: {expected}")

    result = pipeline(question=question)

    print(f"\n  RETRIEVED CHUNKS:")
    for j, passage in enumerate(result.context, 1):
        print(f"  [{j}] {passage[:120].strip()}...")

    print(f"\n  REASONING: {result.reasoning[:250].strip()}")
    print(f"\n  ANSWER: {result.answer}")
    print(f"\n  Correct? Check expected above ^")

print(f"\n{'='*65}")
print("Spot check complete.")
