"""
evaluate.py
-----------
Runs a fixed set of dynamic sample questions against the pipeline,
checks whether retrieval pulled context from the expected source
document, and writes:

  - logs/validation_log.txt     -> full Q/A + retrieval trace, human-readable
  - metrics/system_metrics.md   -> chunking / embedding / vector-store / LLM config report
"""

import json
import time
from datetime import datetime
from src.pipeline import RAGPipeline

# Each test case names the doc_id we EXPECT to be the top retrieved source,
# based on the sample corpus in data/sample_docs/. This lets us log a simple
# retrieval-accuracy metric alongside the generated answers.
TEST_QUESTIONS = [
    {"question": "What is the main idea of the RAG project?", "expected_doc": "rag_project_notes"},
    {"question": "What are the stages of the RAG system architecture?", "expected_doc": "rag_project_notes"},
    {"question": "What is retrieval-augmented generation used for?", "expected_doc": "rag_project_notes"},
    {"question": "What is a vector database?", "expected_doc": "vector_databases"},
    {"question": "Why do vector databases use approximate nearest neighbor search?", "expected_doc": "vector_databases"},
    {"question": "What is the difference between FAISS and Pinecone?", "expected_doc": "vector_databases"},
    {"question": "What does hybrid search combine?", "expected_doc": "vector_databases"},
    {"question": "What is an embedding model?", "expected_doc": "embedding_models"},
    {"question": "How does TF-IDF and LSA work for embeddings?", "expected_doc": "embedding_models"},
    {"question": "How is embedding quality evaluated?", "expected_doc": "embedding_models"},
    {"question": "What improvements could be made to this RAG project?", "expected_doc": "rag_project_notes"},
    {"question": "What is a cross-encoder used for?", "expected_doc": "vector_databases"},
]


def run_evaluation(pipeline: RAGPipeline, log_path: str, metrics_path: str):
    log_lines = []
    log_lines.append("RAG SYSTEM — VALIDATION LOG")
    log_lines.append(f"Generated: {datetime.now().isoformat()}")
    log_lines.append(f"Config: {json.dumps(pipeline.config, indent=2)}")
    log_lines.append(f"Build metrics: {json.dumps(pipeline.metrics, indent=2)}")
    log_lines.append("=" * 90)

    correct_top1 = 0
    correct_topk = 0
    total_time = 0.0

    for i, case in enumerate(TEST_QUESTIONS, 1):
        q = case["question"]
        expected = case["expected_doc"]

        result = pipeline.ask(q)
        total_time += result["timing"]["total_seconds"]

        retrieved_docs = [c["doc_id"] for c in result["retrieved_chunks"]]
        top1_hit = retrieved_docs[0] == expected if retrieved_docs else False
        topk_hit = expected in retrieved_docs

        correct_top1 += int(top1_hit)
        correct_topk += int(topk_hit)

        log_lines.append(f"\n[Test {i}] Question: {q}")
        log_lines.append(f"Expected source doc: {expected}")
        log_lines.append(f"Retrieved source docs (ranked): {retrieved_docs}")
        log_lines.append(f"Top-1 match: {top1_hit} | Top-k match: {topk_hit}")
        log_lines.append(f"Answer:\n{result['answer']}")
        log_lines.append(f"Timing (s): {result['timing']}")
        log_lines.append("-" * 90)

    n = len(TEST_QUESTIONS)
    top1_acc = correct_top1 / n
    topk_acc = correct_topk / n
    avg_latency = total_time / n

    summary = (
        f"\nSUMMARY\n"
        f"Questions evaluated: {n}\n"
        f"Top-1 retrieval accuracy (correct doc ranked first): {top1_acc:.2%}\n"
        f"Top-k retrieval accuracy (correct doc in final context set): {topk_acc:.2%}\n"
        f"Average end-to-end latency per question: {avg_latency:.4f}s\n"
    )
    log_lines.append(summary)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    write_metrics_report(pipeline, metrics_path, top1_acc, topk_acc, avg_latency, n)
    print(summary)
    print(f"Full validation log written to: {log_path}")
    print(f"System metrics report written to: {metrics_path}")


def write_metrics_report(pipeline, path, top1_acc, topk_acc, avg_latency, n_questions):
    m = pipeline.metrics
    c = pipeline.config
    content = f"""# RAG System — Metrics Report

Generated: {datetime.now().isoformat()}

## 1. Document Ingestion
- Source path: `{c['data_path']}`
- Documents ingested: {m['num_documents']}
- Ingestion time: {m['ingestion_seconds']}s

## 2. Chunking Profile
- Strategy: sentence-aware sliding window (see `src/chunking.py`)
- Chunk size (target chars): {c['chunk_size']}
- Chunk overlap (chars): {c['chunk_overlap']}
- Total chunks produced: {m['num_chunks']}
- Average chunk length: {m['avg_chunk_chars']} characters
- Chunking time: {m['chunking_seconds']}s

## 3. Embedding Model
- Backend: `{c['embedder_backend']}` ({'TF-IDF + Truncated SVD (LSA), fully offline' if c['embedder_backend'] == 'lsa' else 'pretrained sentence-transformers model'})
- Embedding dimension: {m['embedding_dim']}
- Indexing time (embed + index build): {m['indexing_seconds']}s

## 4. Vector Store
- Backend: `{m['vector_backend']}`
- Similarity metric: cosine similarity (inner product over L2-normalized vectors)
- Hybrid search (dense + TF-IDF keyword): {c['use_hybrid_search']}
- Retrieval top-k (pre-rerank): {c['retrieve_top_k']}

## 5. Re-ranking
- Enabled: {c['use_reranking']}
- Method: lexical term-overlap re-scoring blended with retrieval score (`src/rerank.py`)
- Final context chunks passed to generator: {c['rerank_top_k']}

## 6. Language Model / Generation
- Backend: `{c['generator_backend']}` ({'deterministic, grounded sentence-extraction — no LLM download required' if c['generator_backend'] == 'extractive' else 'transformer-based generative model'})
- Prompt construction: query + concatenated top-ranked context chunks (`src/generation.py::build_prompt`)

## 7. Evaluation Results ({n_questions} dynamic sample questions)
- Top-1 retrieval accuracy: {top1_acc:.2%}
- Top-k retrieval accuracy: {topk_acc:.2%}
- Average end-to-end latency per question: {avg_latency:.4f}s

## 8. Notes on Scaling to Production
- Swap `--embedder sentence-transformers` for higher-quality dense embeddings once internet access to Hugging Face is available.
- Swap `--generator hf-local` (e.g. `google/flan-t5-base`) or `--generator openai` for free-text generative answers instead of extractive answers.
- Swap the vector store backend to a managed service (Pinecone, Weaviate, Chroma Cloud) for multi-user, persistent, horizontally-scaled deployments.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    print("Building pipeline for evaluation...")
    pipeline = RAGPipeline(data_path="data/sample_docs")
    run_evaluation(pipeline, "logs/validation_log.txt", "metrics/system_metrics.md")
