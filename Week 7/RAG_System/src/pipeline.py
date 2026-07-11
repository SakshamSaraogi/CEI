"""
pipeline.py
-----------
Wires together ingestion -> chunking -> embedding -> vector store ->
retrieval -> (optional) reranking -> generation into a single
end-to-end RAG pipeline.
"""

import time
from src.ingestion import load_documents
from src.chunking import chunk_documents
from src.embeddings import get_embedder
from src.vectorstore import VectorStore
from src.rerank import overlap_rerank
from src.generation import get_generator


class RAGPipeline:
    def __init__(
        self,
        data_path: str = "data/sample_docs",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedder_backend: str = "lsa",
        generator_backend: str = "extractive",
        embedding_dim: int = 128,
        retrieve_top_k: int = 8,
        rerank_top_k: int = 3,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
    ):
        self.config = dict(
            data_path=data_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder_backend=embedder_backend,
            generator_backend=generator_backend,
            embedding_dim=embedding_dim,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
        )
        self.metrics = {}
        self._build(data_path, chunk_size, chunk_overlap, embedder_backend, embedding_dim)
        self.generator = get_generator(generator_backend)
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking

    def _build(self, data_path, chunk_size, chunk_overlap, embedder_backend, embedding_dim):
        t0 = time.time()
        documents = load_documents(data_path)
        t1 = time.time()

        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
        t2 = time.time()

        embedder = get_embedder(embedder_backend, dim=embedding_dim) if embedder_backend == "lsa" \
            else get_embedder(embedder_backend)
        store = VectorStore(embedder, chunks)
        t3 = time.time()

        self.documents = documents
        self.chunks = chunks
        self.embedder = embedder
        self.store = store

        self.metrics.update(
            num_documents=len(documents),
            num_chunks=len(chunks),
            avg_chunk_chars=round(sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
            embedding_dim=store.dim,
            vector_backend=store.backend,
            ingestion_seconds=round(t1 - t0, 4),
            chunking_seconds=round(t2 - t1, 4),
            indexing_seconds=round(t3 - t2, 4),
        )

    def ask(self, query: str, verbose: bool = False):
        t0 = time.time()
        candidates = self.store.similarity_search(
            query, top_k=self.retrieve_top_k, hybrid=self.use_hybrid_search
        )
        t1 = time.time()

        if self.use_reranking:
            top_chunks = overlap_rerank(query, candidates, top_n=self.rerank_top_k)
        else:
            top_chunks = candidates[: self.rerank_top_k]
        t2 = time.time()

        answer = self.generator.generate(query, top_chunks)
        t3 = time.time()

        result = {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": round(s, 4), "text": c.text}
                for c, s in top_chunks
            ],
            "timing": {
                "retrieval_seconds": round(t1 - t0, 4),
                "rerank_seconds": round(t2 - t1, 4),
                "generation_seconds": round(t3 - t2, 4),
                "total_seconds": round(t3 - t0, 4),
            },
        }

        if verbose:
            print(f"\nQ: {query}")
            print(f"A: {answer}")
            print("Top retrieved chunks:")
            for c, s in top_chunks:
                print(f"  - [{c.chunk_id}] score={s:.4f} :: {c.text[:100]}...")

        return result
