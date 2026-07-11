"""
rerank.py
---------
Re-ranking layer applied after the initial retrieval pass to improve
precision at the top of the result list.

Two backends:

1. "overlap" (default, OFFLINE): re-scores candidates using normalized
   token-overlap / BM25-style term-weighting between the query and
   each candidate chunk, combined with the original retrieval score.
   No model download required.

2. "cross-encoder" (OPTIONAL, requires internet to download a
   pretrained cross-encoder such as 'cross-encoder/ms-marco-MiniLM-L-6-v2').
   Scores each (query, chunk) pair jointly for higher-precision reranking.
"""

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def overlap_rerank(query: str, candidates, alpha: float = 0.5, top_n: int = 3):
    """
    candidates: list of (Chunk, retrieval_score)
    Returns top_n candidates re-scored by alpha * retrieval_score +
    (1 - alpha) * normalized term-overlap score.
    """
    q_tokens = Counter(_tokenize(query))
    if not q_tokens:
        return candidates[:top_n]

    rescored = []
    for chunk, score in candidates:
        c_tokens = Counter(_tokenize(chunk.text))
        overlap = sum((q_tokens & c_tokens).values())
        overlap_norm = overlap / max(len(q_tokens), 1)
        final_score = alpha * score + (1 - alpha) * overlap_norm
        rescored.append((chunk, final_score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]


def cross_encoder_rerank(query: str, candidates, top_n: int = 3,
                          model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Optional online reranker. Requires `pip install sentence-transformers`."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores = model.predict(pairs)
    rescored = list(zip([c for c, _ in candidates], scores))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]
