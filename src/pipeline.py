# src/pipeline.py
import dspy
from src.signature import GenerateAnswer
from src.retriever import NovaRetriever


class NovaRAGPipeline(dspy.Module):
    """
    Full RAG pipeline for NovaDesk customer support.

    Step 1: Retrieve relevant passages from ChromaDB
    Step 2: Generate answer using retrieved context
    """

    def __init__(self, k=3):
        super().__init__()
        self.retriever = NovaRetriever(k=k)
        self.generate  = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        # Step 1 — Retrieve
        retrieved = self.retriever(question)
        context   = retrieved.passages

        # Step 2 — Generate (with retry on template-echo bug)
        max_retries = 2
        answer      = None
        reasoning   = None

        for attempt in range(max_retries + 1):
            result = self.generate(
                context=context,
                question=question
            )

            # Check if the model echoed the template placeholder
            if result.answer and "{" not in result.answer and "}" not in result.answer:
                answer    = result.answer
                reasoning = result.reasoning
                break

        # If all retries failed, fall back gracefully
        if answer is None:
            answer    = "I don't have enough information to answer that."
            reasoning = "Generation failed after retries."

        return dspy.Prediction(
            answer=answer,
            context=context,
            reasoning=reasoning
        )


# ── Test Function ────────────────────────────────────────────────────
def test_pipeline():
    test_questions = [
        "How much does the Pro plan cost?",
        "Can I downgrade my plan mid billing cycle?",
        "What happens to my data if I cancel my account?",
        "How many agents can I add on the Starter plan?",
        "Does NovaDesk support PayPal payments?",
    ]

    pipeline = NovaRAGPipeline(k=3)

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = pipeline(question=question)
        print(f"\nReasoning: {result.reasoning[:200]}...")
        print(f"\nA: {result.answer}")
        print(f"\nSources used:")
        for i, passage in enumerate(result.context):
            print(f"  [{i+1}] {passage[:100]}...")