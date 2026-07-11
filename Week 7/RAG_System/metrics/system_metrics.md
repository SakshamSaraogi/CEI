# RAG System — Metrics Report

Generated: 2026-07-05T16:42:25.210044

## 1. Document Ingestion
- Source path: `data/sample_docs`
- Documents ingested: 3
- Ingestion time: 0.0001s

## 2. Chunking Profile
- Strategy: sentence-aware sliding window (see `src/chunking.py`)
- Chunk size (target chars): 800
- Chunk overlap (chars): 150
- Total chunks produced: 14
- Average chunk length: 676.0 characters
- Chunking time: 0.0004s

## 3. Embedding Model
- Backend: `lsa` (TF-IDF + Truncated SVD (LSA), fully offline)
- Embedding dimension: 13
- Indexing time (embed + index build): 0.9981s

## 4. Vector Store
- Backend: `faiss.IndexFlatIP`
- Similarity metric: cosine similarity (inner product over L2-normalized vectors)
- Hybrid search (dense + TF-IDF keyword): True
- Retrieval top-k (pre-rerank): 8

## 5. Re-ranking
- Enabled: True
- Method: lexical term-overlap re-scoring blended with retrieval score (`src/rerank.py`)
- Final context chunks passed to generator: 3

## 6. Language Model / Generation
- Backend: `extractive` (deterministic, grounded sentence-extraction — no LLM download required)
- Prompt construction: query + concatenated top-ranked context chunks (`src/generation.py::build_prompt`)

## 7. Evaluation Results (12 dynamic sample questions)
- Top-1 retrieval accuracy: 91.67%
- Top-k retrieval accuracy: 100.00%
- Average end-to-end latency per question: 0.0013s

## 8. Notes on Scaling to Production
- Swap `--embedder sentence-transformers` for higher-quality dense embeddings once internet access to Hugging Face is available.
- Swap `--generator hf-local` (e.g. `google/flan-t5-base`) or `--generator openai` for free-text generative answers instead of extractive answers.
- Swap the vector store backend to a managed service (Pinecone, Weaviate, Chroma Cloud) for multi-user, persistent, horizontally-scaled deployments.
