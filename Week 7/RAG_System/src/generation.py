"""
generation.py
-------------
Connects retrieved context chunks + the original query into a unified
prompt and produces a grounded answer.

Backends:

1. "extractive" (default, fully OFFLINE): builds a grounded answer
   directly from the retrieved chunks using sentence-level relevance
   scoring (no LLM weights to download). Deterministic and fast --
   used for the demo run/logs in this repo.

2. "hf-local" (OPTIONAL, requires internet to download a small seq2seq
   model such as 'google/flan-t5-base' the first time): wraps a local
   HuggingFace transformers pipeline for real free-text generation.

3. "openai" / "cohere" (OPTIONAL, requires an API key + internet):
   thin wrappers so you can swap in a hosted LLM for production-quality
   answers. See README for how to enable.

All backends implement: generate(query, context_chunks) -> str
"""

import re
from src.chunking import Chunk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_prompt(query: str, context_chunks) -> str:
    context_block = "\n\n".join(
        f"[Source: {c.doc_id}]\n{c.text}" for c, _score in context_chunks
    )
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context below. If the answer is not contained in the context, say "
        "you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


class ExtractiveGenerator:
    """
    Offline generator: selects and lightly stitches together the most
    query-relevant sentences from the retrieved chunks. This keeps the
    answer strictly grounded (every word traces back to the source
    documents) without needing to download an LLM.
    """

    name = "extractive"

    def generate(self, query: str, context_chunks, max_sentences: int = 3) -> str:
        if not context_chunks:
            return "I don't know — no relevant context was found in the documents."

        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        candidates = []
        for chunk, score in context_chunks:
            for sentence in SENTENCE_SPLIT_RE.split(chunk.text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                s_tokens = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                overlap = len(q_tokens & s_tokens)
                relevance = overlap + 0.001 * score
                candidates.append((relevance, sentence, chunk.doc_id))

        if not candidates:
            return "I don't know — the retrieved context did not contain a clear answer."

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:max_sentences]
        # preserve original document order roughly by re-sorting on first appearance
        answer = " ".join(sent for _, sent, _ in top)
        sources = sorted(set(doc_id for _, _, doc_id in top))
        return f"{answer}\n\n(Grounded in: {', '.join(sources)})"


class HFLocalGenerator:
    """Optional local seq2seq LLM backend (requires internet on first run)."""

    name = "hf-local"

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, query: str, context_chunks, **kwargs) -> str:
        prompt = build_prompt(query, context_chunks)
        result = self.pipe(prompt, max_new_tokens=200)
        return result[0]["generated_text"]


class OpenAIGenerator:
    """Optional hosted LLM backend. Requires OPENAI_API_KEY + internet."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def generate(self, query: str, context_chunks, **kwargs) -> str:
        prompt = build_prompt(query, context_chunks)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


def get_generator(name: str = "extractive", **kwargs):
    if name == "extractive":
        return ExtractiveGenerator()
    if name == "hf-local":
        return HFLocalGenerator(**kwargs)
    if name == "openai":
        return OpenAIGenerator(**kwargs)
    raise ValueError(f"Unknown generator backend: {name}")
