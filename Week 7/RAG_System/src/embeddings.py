"""
embeddings.py
-------------
Maps chunked text strings into vector representations.

Two pluggable backends are provided:

1. "lsa" (default, fully OFFLINE, no model download required):
   TF-IDF vectorization followed by Truncated SVD (Latent Semantic
   Analysis). This gives genuine dense semantic vectors -- similar
   documents/chunks land close together in the reduced space -- without
   requiring any internet access to download pretrained weights. This
   is what is used for the demo run in this repo/sandbox.

2. "sentence-transformers" (OPTIONAL, requires internet the first time
   to download the pretrained model, e.g. 'all-MiniLM-L6-v2'):
   Produces higher-quality dense embeddings from a pretrained
   transformer encoder. Recommended for production use once you have
   unrestricted internet access (e.g. on your own machine or in
   GitHub Actions / Colab).

Both backends expose the same interface:
    embedder = get_embedder(name, **kwargs)
    embedder.fit(list_of_texts)          # build/train the space
    vectors = embedder.transform(texts)  # -> np.ndarray [n, dim]
    vector  = embedder.embed_query(text) # -> np.ndarray [dim]
"""

import numpy as np


class LSAEmbedder:
    """Offline embedding backend: TF-IDF + Truncated SVD (LSA)."""

    name = "tfidf-lsa"

    def __init__(self, dim: int = 128, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_df=0.9,
            min_df=1,
        )
        self.svd = None
        self._fitted = False

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD

        tfidf = self.vectorizer.fit_transform(texts)
        # SVD components cannot exceed min(n_samples, n_features) - 1
        n_components = max(2, min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = self.svd.fit_transform(tfidf)
        self._fitted = True
        self.dim = n_components
        return self._normalize(vectors)

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before transform().")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        return self._normalize(vectors)

    def embed_query(self, text: str):
        return self.transform([text])[0]

    def get_tfidf_query(self, text: str):
        """Raw sparse TF-IDF vector for keyword-based scoring (hybrid search)."""
        return self.vectorizer.transform([text])

    def get_tfidf_matrix(self, texts):
        return self.vectorizer.transform(texts)

    @staticmethod
    def _normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vectors / norms


class SentenceTransformerEmbedder:
    """
    Optional online backend using a pretrained sentence-transformers model.
    Requires internet access on first use to download model weights.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._fitted = False
        self._texts_cache = None

    def fit(self, texts):
        self._fitted = True
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors)

    def transform(self, texts):
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors)

    def embed_query(self, text: str):
        return self.transform([text])[0]


def get_embedder(name: str = "lsa", **kwargs):
    if name in ("lsa", "tfidf", "tfidf-lsa", "offline"):
        return LSAEmbedder(**kwargs)
    if name in ("sentence-transformers", "st", "minilm"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder backend: {name}")
