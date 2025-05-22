import nltk
nltk.download('punkt_tab')

import os
import re
import faiss
import pickle
import nltk
from typing import List
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name = "all-MiniLM-L6-v2", sentences_per_chunk = 2):
        self.model = SentenceTransformer(model_name)
        self.sentences_per_chunk = sentences_per_chunk
        self.index = None
        self.documents = []
        self.embeddings = []

    def _clean_text(self, text):
        """
        Clean the input text by collapsing whitespace.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def _chunk_text(self, text):
        """
        Split text into sentence-based chunks with optional overlap.
        """
        sentences = sent_tokenize(text)
        chunks = []
        step = self.sentences_per_chunk
        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i:i + self.sentences_per_chunk])
            if chunk:
                chunks.append(chunk)
        return chunks

    def _load_txt(self, filepath):
        """
        Load and clean a .txt file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return self._clean_text(f.read())

    def add_txt_files(self, filepaths):
        """
        Load .txt files, chunk them, embed, and add to the FAISS index.
        """
        all_chunks = []
        for filepath in filepaths:
            if not filepath.endswith('.txt'):
                print(f"Skipping unsupported file type: {filepath}")
                continue
            text = self._load_txt(filepath)
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)
            all_chunks.extend(chunks)

        if all_chunks:
            embeddings = self.model.encode(all_chunks, show_progress_bar=True)
            self.embeddings.extend(embeddings)
            dim = embeddings[0].shape[0]
            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)

    def query(self, text, top_k = 5):
        """
        Perform a semantic search against the indexed chunks.
        """
        query_vec = self.model.encode([text])
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in I[0]]

    def save(self, path):
        """
        Save the FAISS index and documents to disk.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path):
        """
        Load the FAISS index and documents from disk.
        """
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)
