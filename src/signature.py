import dspy

class GenerateAnswer(dspy.Signature):
    """
    Answer a customer support question using the
    retrieved context from the NovaDesk knowledge base.
    If the context does not contain the answer, say
    'I don't have information about that.'
    """

    context  : list[str] = dspy.InputField(
        desc="Relevant passages from the NovaDesk knowledge base"
    )
    question : str = dspy.InputField(
        desc="Customer support question about NovaDesk"
    )
    answer   : str = dspy.OutputField(
        desc="Clear, concise answer based only on the context provided"
    )