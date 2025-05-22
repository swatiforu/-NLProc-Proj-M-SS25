from transformers import pipeline

class Generator:
    def __init__(self, model_name = "google/flan-t5-base"):
        """
        Initialize the text-to-text generation model pipeline.
        """
        self.model_name = model_name
        self.generator = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, context, question):
        """
        Construct a prompt for the language model using context and question.
        """
        prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt

    def generate_answer(self, context, question, max_new_tokens = 100):
        """
        Generate an answer to the question using the context provided.
        """
        prompt = self.build_prompt(context, question)
        response = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return response[0]['generated_text'].strip()
