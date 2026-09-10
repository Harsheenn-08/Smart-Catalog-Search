import json
import numpy as np
import faiss
from database import get_database_connection

VECTOR_DIMENSION = 384  # SentenceTransformer("all-MiniLM-L6-v2") output size

class VectorIndex:
    def __init__(self, dimension=VECTOR_DIMENSION):
        self.dimension = dimension
        # Use IndexFlatIP (Inner Product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []  # Maps FAISS integer positions to SQLite Product IDs

    def build_index_from_db(self):
        """Loads all product vectors from SQLite into the FAISS index."""
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, embedding FROM products")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("No vectors found in database to index.")
            return

        vectors = []
        self.id_map = []

        for row in rows:
            vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
            # Normalize vector for inner product to equal cosine similarity
            faiss.normalize_L2(vec.reshape(1, -1))
            vectors.append(vec)
            self.id_map.append(row["id"])

        vectors_np = np.vstack(vectors).astype(np.float32)
        self.index.reset()
        self.index.add(vectors_np)
        print(f"FAISS index successfully populated with {self.index.ntotal} vectors.")

    def search(self, query_vector, top_k=5):
        """Searches FAISS index for nearest neighbor vector matches."""
        if self.index.ntotal == 0:
            return [], []

        query_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        # Retrieve top K matches
        distances, indices = self.index.search(query_vec, top_k)
        
        matched_ids = []
        scores = []

        # Iterate over zipped index and distance pairs
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.id_map):
                matched_ids.append(self.id_map[idx])
                scores.append(float(dist))

        return matched_ids, scores