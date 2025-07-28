import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", embeddings_dir="../embeddings"):
        self.model = SentenceTransformer(model_name)
        self.embeddings_dir = embeddings_dir
        os.makedirs(embeddings_dir, exist_ok=True)
        
    def create_embeddings(self, texts, save_name=None):
        """Create embeddings for a list of texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if save_name:
            self.save_embeddings(embeddings, save_name)
        
        return embeddings
    
    def save_embeddings(self, embeddings, filename):
        """Save embeddings to a pickle file"""
        filepath = os.path.join(self.embeddings_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return filepath
    
    def load_embeddings(self, filename):
        """Load embeddings from a pickle file"""
        filepath = os.path.join(self.embeddings_dir, filename)
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        return embeddings
    
    def compute_similarity(self, query_embedding, document_embeddings):
        """Compute cosine similarity between query and documents"""
        similarities = np.dot(document_embeddings, query_embedding) / (
            np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        return similarities 