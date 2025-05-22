from retriever import Retriever
from generator import Generator
from logger import log_query

retriever = Retriever()
retriever.add_txt_files([".data/sample_file.txt"])

generator = Generator()

def run_pipeline(question, group_id="TeamQuery"):
    retrieved_chunks = retriever.query(question, top_k=3)
    context = " ".join(retrieved_chunks)
    answer = generator.generate_answer(context, question)

    log_query({
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": generator.build_prompt(context, question),
        "generated_answer": answer,
    })

    return answer
