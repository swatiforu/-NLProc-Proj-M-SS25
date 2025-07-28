import os
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from retriever import Retriever
from generator import Generator

# Disable Triton compilation to avoid errors
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def check_gpu():
    if torch.cuda.is_available():
        device = "cuda"
        return device
    else:
        return "cpu"

def load_model_and_tokenizer(local_model_dir="./gemma-3-1b-it"):
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"Model directory '{local_model_dir}' not found.")
    device = check_gpu()
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    if device == "cuda":
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    if device == "cuda":
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return pipe, device

def create_retriever(pdf_path):
    retriever = Retriever()
    retriever.load_pdf(pdf_path)
    return retriever

def create_generator(pipe):
    return Generator(pipe) 