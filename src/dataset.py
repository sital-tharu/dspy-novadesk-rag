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
        # ── SECTION 3: BILLING (6 questions) ───────────────────────
        dspy.Example(
            question="Can I downgrade my plan in the middle of a billing cycle?",
            answer="Downgrading takes effect at the start of the next billing cycle, not immediately."
        ),
        dspy.Example(
            question="What payment methods does NovaDesk accept?",
            answer="NovaDesk accepts Visa, Mastercard, and American Express. PayPal is not supported."
        ),
        dspy.Example(
            question="Does NovaDesk support PayPal payments?",
            answer="No, PayPal is not supported. NovaDesk accepts Visa, Mastercard, and American Express."
        ),
        dspy.Example(
            question="How long is data retained after I cancel my account?",
            answer="Data is retained for 60 days after cancellation before permanent deletion."
        ),
        dspy.Example(
            question="Can I get a refund if I cancel mid month?",
            answer="No, NovaDesk does not offer refunds for partial months."
        ),
        dspy.Example(
            question="When does an upgrade take effect?",
            answer="Upgrades take effect immediately and you are charged a prorated amount for the remainder of the billing cycle."
        ),

        # ── SECTION 4: AGENTS (4 questions) ────────────────────────
        dspy.Example(
            question="How long does an agent invitation last before it expires?",
            answer="Agent invitations expire after 48 hours."
        ),
        dspy.Example(
            question="What happens to tickets when an agent is removed?",
            answer="Tickets previously handled by the removed agent are reassigned to the admin who removed them."
        ),
        dspy.Example(
            question="What is the difference between Agent and Viewer roles?",
            answer="Agents can view and respond to tickets. Viewers have read-only access and cannot respond to tickets or change settings."
        ),
        dspy.Example(
            question="Can a Viewer role access billing settings?",
            answer="No, Viewers have read-only access to tickets and reports only. They cannot access billing or team settings."
        ),