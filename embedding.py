# embedding.py
import os
import faiss
import pickle
import numpy as np
from typing import List, Tuple
from google.generativeai import configure, embeddings as gem_embeddings

# Configure Gemini API
configure(api_key=os.getenv("GOOGLE_API_KEY"))

class EmbeddingHandler:
    def __init__(
        self,
        embedding_model="models/embedding-001",
        db_path="vector_store/faiss_index",
    ):
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.index = None
        self.documents = []
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_embedding(self, text):
        """Get embedding from Gemini API"""
        try:
            result = gem_embeddings.embed_content(
                model=self.embedding_model, content=text
            )
            return result["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def create_embeddings(self, chunks: List[str]):
        """Convert text chunks into embeddings and store in FAISS"""
        vectors = []
        for chunk in chunks:
            emb = self.get_embedding(chunk)
            if emb:
                vectors.append(emb)
                self.documents.append(chunk)

        if not vectors:
            raise ValueError("No embeddings were created. Check API key or content.")

        dimension = len(vectors[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(vectors, dtype="float32"))
        self.save_index()

    def save_index(self):
        """Save FAISS index and documents"""
        faiss.write_index(self.index, f"{self.db_path}.faiss")
        with open(f"{self.db_path}_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self):
        """Load FAISS index and documents"""
        self.index = faiss.read_index(f"{self.db_path}.faiss")
        with open(f"{self.db_path}_docs.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query: str, top_k=5) -> List[Tuple[str, float]]:
        """Search similar chunks for a query"""
        query_emb = self.get_embedding(query)
        if query_emb is None:
            return []
        D, I = self.index.search(np.array([query_emb], dtype="float32"), top_k)
        return [(self.documents[i], float(D[0][j])) for j, i in enumerate(I[0])]