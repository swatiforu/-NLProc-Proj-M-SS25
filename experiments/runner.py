import os
import sys
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from retriever import Retriever
from generator import Generator
from legal_assistant import load_model_and_tokenizer

# Disable Triton compilation to avoid errors
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def check_gpu():
    """Check GPU availability and return device info"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        return device, gpu_memory
    else:
        print("⚠️  No GPU detected, using CPU")
        return "cpu", 0

def main():
    # Load model and pipeline
    pipe, device = load_model_and_tokenizer()
    print(f"Loaded model on {device}")

    # Load retriever with your PDF
    retriever = Retriever()
    retriever.load_pdf("CELEX_32016R0679_EN_TXT.pdf")  # Change to your PDF if needed

    # Create generator
    gen = Generator(pipe)

    print("Legal Assistant Ready! Type 'quit' to exit.")
    while True:
        query = input("\nEnter your legal question: ").strip()
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a question.")
            continue
        print("\nGenerating answer (streaming):")
        chunks = retriever.retrieve(query, threshold=0.5)
        gen.generate(query, chunks, max_new_tokens=512)  # This will stream output to terminal

if __name__ == "__main__":
    main() 