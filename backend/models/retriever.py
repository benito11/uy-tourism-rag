from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import Dict, List

class DocumentRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Update paths to be relative to the backend directory
        self.embeddings_path = "/app/backend/data/embeddings/embeddings.npy"
        self.documents_path = "/app/backend/data/embeddings/documents.json"
        self.metadata_path = "/app/backend/data/embeddings/metadata.json"
        self.initialize_index()

    def initialize_index(self):
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            
            # Load embeddings and documents
            if not all(os.path.exists(p) for p in [self.embeddings_path, self.documents_path, self.metadata_path]):
                print("Index files not found. Please run setup_index.py first.")
                self.embeddings = None
                self.documents = []
                self.metadata = []
                return

            self.embeddings = np.load(self.embeddings_path)
            with open(self.documents_path, 'r') as f:
                self.documents = json.load(f)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("Index loaded successfully")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            self.embeddings = None
            self.documents = []
            self.metadata = []

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.embeddings is None or not self.documents:
            return [{
                "text": "Index not initialized. Please run setup_index.py first.",
                "metadata": {},
                "similarity": 0.0
            }]
        
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            return [{
                "text": f"Error during retrieval: {str(e)}",
                "metadata": {},
                "similarity": 0.0
            }]