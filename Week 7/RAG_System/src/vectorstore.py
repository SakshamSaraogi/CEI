"""
vectorstore.py
--------------
Initializes a vector database, stores chunk embeddings, and configures
them for fast similarity search.

Backend: FAISS (IndexFlatIP over L2-normalized vectors == cosine
similarity search). If FAISS is not installed, falls back automatically
to an exact NumPy brute-force cosine search so the pipeline still runs.

Also implements hybrid search (dense vector similarity + sparse TF-IDF
keyword overlap) and a lightweight re-ranking pass.
"""

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, embedder, chunks, dense_weight: float = 0.7):
        """
        embedder: an embeddings.LSAEmbedder / SentenceTransformerEmbedder instance
        chunks:   List[chunking.Chunk]
        dense_weight: weight given to dense/vector score in hybrid search
                      (1 - dense_weight is given to the sparse keyword score)
        """
        self.embedder = embedder
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.dense_weight = dense_weight

        # 1. Fit embedder + build dense vectors
        self.vectors = embedder.fit(self.texts).astype("float32")
        self.dim = self.vectors.shape[1]

        # 2. Build ANN index
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vectors)
            self.backend = "faiss.IndexFlatIP"
        else:
            self.index = None
            self.backend = "numpy-brute-force"

        # 3. Sparse TF-IDF matrix for keyword / hybrid scoring
        self._tfidf_matrix = (
            embedder.get_tfidf_matrix(self.texts) if hasattr(embedder, "get_tfidf_matrix") else None
        )

    # ---------- dense search ----------

    def _dense_search(self, query_vector: np.ndarray, top_k: int):
        query_vector = query_vector.astype("float32").reshape(1, -1)
        if self.index is not None:
            scores, idxs = self.index.search(query_vector, top_k)
            return idxs[0].tolist(), scores[0].tolist()
        # brute-force cosine (vectors are already L2-normalized)
        sims = self.vectors @ query_vector[0]
        top_idx = np.argsort(-sims)[:top_k]
        return top_idx.tolist(), sims[top_idx].tolist()

    # ---------- sparse keyword search ----------

    def _sparse_scores(self, query: str):
        if self._tfidf_matrix is None:
            return None
        q_vec = self.embedder.get_tfidf_query(query)
        scores = (self._tfidf_matrix @ q_vec.T).toarray().ravel()
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        return scores

    # ---------- public retrieval ----------

    def similarity_search(self, query: str, top_k: int = 5, hybrid: bool = True):
        """Returns list of (Chunk, score) sorted by relevance, descending."""
        query_vector = self.embedder.embed_query(query)
        candidate_k = max(top_k * 4, 20)
        idxs, dense_scores = self._dense_search(query_vector, min(candidate_k, len(self.chunks)))

        dense_score_map = {i: s for i, s in zip(idxs, dense_scores)}

        if hybrid and self._tfidf_matrix is not None:
            sparse_scores = self._sparse_scores(query)
            combined = {}
            for i in dense_score_map:
                combined[i] = (
                    self.dense_weight * dense_score_map[i]
                    + (1 - self.dense_weight) * sparse_scores[i]
                )
        else:
            combined = dense_score_map

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(score)) for i, score in ranked]
