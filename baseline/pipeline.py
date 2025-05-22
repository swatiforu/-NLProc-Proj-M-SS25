import Retriever
import Generator

retriever = Retriever()
retriever.add_txt_files(["./data/sample_file.txt", "./data/deeplearning.txt"])
vals = retriever.query("Define NLP?", top_k=7)

retriever.save("retriever_data")

context = "\n\n".join(vals)

generator = Generator()
answer = generator.generate_answer(context, "Define NLP?")
print(answer)