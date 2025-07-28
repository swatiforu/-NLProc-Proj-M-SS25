from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import threading
import sys

class StopOnPhraseCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_phrase):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_phrase = stop_phrase
        self.stop_ids = tokenizer.encode(stop_phrase, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) >= len(self.stop_ids):
            if list(input_ids[0][-len(self.stop_ids):]) == self.stop_ids:
                return True
        return False

class Generator:
    def __init__(self, llm_pipeline):
        self.pipeline = llm_pipeline
        self.tokenizer = llm_pipeline.tokenizer
        self.model = llm_pipeline.model

    def generate(self, query, retrieved_chunks, max_new_tokens=300):
        if "No relevant" in retrieved_chunks[0][0]:
            print("No relevant legal information found in the document. Please try rephrasing your question or ask about a different GDPR topic.")
            return "No relevant legal information found in the document. Please try rephrasing your question or ask about a different GDPR topic."

        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)])
        stop_phrase = "End of answer."
        prompt = f"""You are a legal assistant specializing in Contracts and Laws. Based on the following context chunks, provide a comprehensive answer to the legal question.

IMPORTANT INSTRUCTIONS:
- ALWAYS include specific article numbers, section numbers, and paragraph references when they exist in the context.
- Even if the question doesn't ask for specific references, include them in your answer.
- Provide sufficient information to answer the question.
- DO NOT add anything extra, just the answer, no notes, no disclaimers, no summaries, no signatures.
- End your answer with 'End of answer.'

Context:
{context}

Question: {query}

Please provide a detailed legal answer with specific article/section references:"""

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping_criteria = StoppingCriteriaList([StopOnPhraseCriteria(self.tokenizer, stop_phrase)])
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            ),
        )
        thread.start()
        output_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            output_text += new_text
            if stop_phrase in output_text:
                break
        thread.join()
        # Trim at stop phrase
        output_text = output_text.split(stop_phrase)[0].rstrip()
        print()  # for newline after streaming
        return output_text