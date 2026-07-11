"""
main.py
-------
CLI entry point for the RAG Document Question Answering system.

Usage:
    python main.py                          # interactive Q&A over data/sample_docs
    python main.py --data path/to/folder     # use a custom document folder
    python main.py --ask "What is RAG?"      # ask a single question and exit
    python main.py --embedder sentence-transformers --generator hf-local
                                              # opt into the online, higher-quality backends
"""

import argparse
from src.pipeline import RAGPipeline


def parse_args():
    p = argparse.ArgumentParser(description="RAG Document Question Answering System")
    p.add_argument("--data", default="data/sample_docs", help="Path to a document or folder of documents")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--embedder", default="lsa", choices=["lsa", "sentence-transformers"])
    p.add_argument("--generator", default="extractive", choices=["extractive", "hf-local", "openai"])
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8, help="Chunks fetched from the vector store")
    p.add_argument("--rerank-k", type=int, default=3, help="Chunks kept after re-ranking")
    p.add_argument("--no-hybrid", action="store_true", help="Disable hybrid keyword+vector search")
    p.add_argument("--no-rerank", action="store_true", help="Disable the re-ranking pass")
    p.add_argument("--ask", type=str, default=None, help="Ask a single question and exit")
    return p.parse_args()


def main():
    args = parse_args()

    print("Building RAG pipeline...")
    pipeline = RAGPipeline(
        data_path=args.data,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder_backend=args.embedder,
        generator_backend=args.generator,
        embedding_dim=args.embedding_dim,
        retrieve_top_k=args.top_k,
        rerank_top_k=args.rerank_k,
        use_hybrid_search=not args.no_hybrid,
        use_reranking=not args.no_rerank,
    )
    print("Pipeline ready.")
    print(pipeline.metrics)

    if args.ask:
        pipeline.ask(args.ask, verbose=True)
        return

    print("\nType a question (or 'exit' to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        pipeline.ask(query, verbose=True)


if __name__ == "__main__":
    main()
