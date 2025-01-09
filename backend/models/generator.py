from sentence_transformers import SentenceTransformer

class Generator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def generate(self, prompt: str) -> str:
        """
        Generates embeddings for the given text.
        """
        embedding = self.model.encode([prompt])
        return f"Embedding generated with shape: {embedding.shape}"