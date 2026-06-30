# src/optimize.py
import dspy
from dspy.teleprompt import BootstrapFewShot
from src.pipeline import NovaRAGPipeline
from src.evaluate import combined_metric


def optimize_pipeline(trainset):
    """
    Uses BootstrapFewShot to optimize the RAG pipeline.

    What it optimizes:
    - Few-shot examples for the GenerateAnswer module
    - Teaches the LLM better reasoning patterns
      over retrieved context
    """

    optimizer = BootstrapFewShot(
        metric=combined_metric,
        max_bootstrapped_demos=4,   # 4 reasoning examples
        max_labeled_demos=4,        # 4 labeled Q&A examples
        max_rounds=2,               # 2 rounds of bootstrapping
    )

    print("🔄 Optimizing RAG pipeline with BootstrapFewShot...")
    print("   This will take several minutes...\n")

    # Use k=5 for optimization — wider retrieval net
    optimized = optimizer.compile(
        NovaRAGPipeline(k=4),
        trainset=trainset,
    )

    print("\n✅ Optimization complete!")
    return optimized


def inspect_demos(optimized_pipeline):
    """
    Shows what few-shot examples the optimizer
    selected to inject into the prompt.
    """
    print(f"\n{'='*60}")
    print(f"{'FEW-SHOT DEMOS SELECTED BY OPTIMIZER':^60}")
    print(f"{'='*60}")

    try:
        predictor = optimized_pipeline.generate.predict
        demos     = predictor.demos

        if not demos:
            print("No bootstrapped demos found.")
            return

        for i, demo in enumerate(demos):
            print(f"\nDemo {i+1}:")
            print(f"  Q : {demo.get('question', 'N/A')}")
            print(f"  A : {demo.get('answer',   'N/A')}")
            if 'reasoning' in demo:
                print(f"  R : {str(demo['reasoning'])[:120]}...")

    except Exception as e:
        print(f"Could not inspect demos: {e}")
        try:
            for name, predictor in optimized_pipeline.named_predictors():
                if hasattr(predictor, 'demos') and predictor.demos:
                    print(f"\nPredictor: {name}")
                    for i, demo in enumerate(predictor.demos):
                        print(f"\n  Demo {i+1}:")
                        print(f"  Q : {demo.get('question', 'N/A')}")
                        print(f"  A : {demo.get('answer',   'N/A')}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")