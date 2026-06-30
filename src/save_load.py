# src/save_load.py
import os
import json
import dspy

def save_pipeline(pipeline, path="models/optimized_pipeline.json"):
    os.makedirs("models", exist_ok=True)
    try:
        # Try native DSPy save first
        pipeline.save(path)
        print(f"✅ Pipeline saved to {path}")
    except TypeError:
        # Fallback: save only the predictor state (demos)
        state = {}
        for name, predictor in pipeline.named_predictors():
            if hasattr(predictor, 'demos') and predictor.demos:
                state[name] = [
                    {k: str(v) for k, v in demo.items()}
                    for demo in predictor.demos
                ]
        fallback_path = path.replace(".json", "_demos.json")
        with open(fallback_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"✅ Demos saved to {fallback_path}")

def load_pipeline(pipeline_class, path="models/optimized_pipeline.json"):
    pipeline = pipeline_class()
    pipeline.load(path)
    print(f"✅ Pipeline loaded from {path}")
    return pipeline