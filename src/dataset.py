import dspy

def get_dataset():
    """
    30 Q&A pairs for evaluating and optimizing the NovaDesk RAG pipeline.
    Covers all 10 sections of the knowledge base.
    Includes easy, medium, and hard questions.
    """

    examples = [

        # ── SECTION 2: PRICING (6 questions) ───────────────────────
        dspy.Example(
            question="How much does the Pro plan cost per month?",
            answer="The Pro plan costs $79 per month."
        ),
        dspy.Example(
            question="How many agents are allowed on the Starter plan?",
            answer="The Starter plan allows up to 3 agents."
        ),
        dspy.Example(
            question="What is the price of the Enterprise plan?",
            answer="The Enterprise plan costs $199 per month."
        ),
        dspy.Example(
            question="How much discount do you get with annual billing?",
            answer="Annual billing gives a 20% discount on all plans."
        ),
        dspy.Example(
            question="Do I need a credit card to start a free trial?",
            answer="No, no credit card is required for the 14-day free trial."
        ),
        dspy.Example(
            question="How many integrations does the Pro plan support?",
            answer="The Pro plan supports up to 10 integrations."
        ),