"""
chunking.py
-----------
Splits raw unstructured document text into smaller, overlapping chunks
so that retrieval can operate at a fine granularity while generation
still receives enough surrounding context.

Strategy: sentence-aware sliding window.
  1. Normalize whitespace.
  2. Split into sentences (simple punctuation-based splitter -- no
     heavy NLP dependency required).
  3. Greedily pack sentences into chunks up to `chunk_size` characters.
  4. Carry the last `overlap` characters of a chunk into the next chunk
     so context is not lost at chunk boundaries.
"""

import re
from dataclasses import dataclass, field
from typing import List

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Sentence-aware sliding-window chunking.

    chunk_size: target max characters per chunk.
    overlap: number of trailing characters repeated at the start of the
             next chunk to preserve context continuity.
    """
    text = _normalize(text)
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the end of the previous chunk
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(documents, chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """documents: List[ingestion.Document] -> List[Chunk]"""
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk{i}",
                    doc_id=doc.doc_id,
                    text=piece,
                    metadata={"source": doc.source, "chunk_index": i},
                )
            )
    return all_chunks
