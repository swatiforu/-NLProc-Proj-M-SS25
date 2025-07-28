import fitz
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_map = {}

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)

    def smart_chunk(self, text, min_size=300):
        lines = text.split("\n")
        chunks = []
        current = []
        patterns = [
            r"^\s*(ARTICLE\s+\d+)",
            r"^\s*(SECTION\s+\d+(\.\d+)?)",
            r"^\s*(§+\s*\d+)",
            r"^\s*(\d+\.\d+)",
            r"^\s*\(\w\)",
            r"^\s*[A-Z ]{10,}$"
        ]
        headers = re.compile("|".join(patterns), re.IGNORECASE)

        for line in lines:
            if headers.match(line.strip()):
                if current:
                    chunk = "\n".join(current).strip()
                    if len(chunk) >= min_size:
                        chunks.append(chunk)
                    current = []
            current.append(line)

        if current:
            chunk = "\n".join(current).strip()
            if len(chunk) >= min_size:
                chunks.append(chunk)

        return chunks

    def embed_and_index(self, chunks):
        print("Creating embeddings...")
        embeddings = self.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.chunk_map = {i: chunk for i, chunk in enumerate(chunks)}

    def load_pdf(self, pdf_path):
        print(f"Processing document: {pdf_path}")
        text = self.extract_text(pdf_path)
        chunks = self.smart_chunk(text)
        print(f"Created {len(chunks)} chunks from document")
        self.embed_and_index(chunks)

    def retrieve(self, query, top_k=3, threshold=0.5):
        if self.index is None:
            raise ValueError("Index not built. Call load_pdf() first.")
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        results = [(self.chunk_map[i], D[0][j]) for j, i in enumerate(I[0]) if D[0][j] >threshold]
        return results if results else [("No relevant legal information found.", None)]
