# RAG Document Question Answering System 🤖📄

A simple, fully-open-source Retrieval-Augmented Generation (RAG) pipeline that answers questions
from custom documents (PDF, TXT, DOCX, or JSON corpora such as Hugging Face RAG datasets).

Unlike the [reference implementation](https://github.com/VivekChauhan05/RAG_Document_Question_Answering)
this uses no paid/cloud-only APIs by default (no Cohere key, no Pinecone key required) — it runs
completely **offline out of the box**, and has optional pluggable hooks to swap in
sentence-transformers, a local HF LLM, or OpenAI/Cohere when you have full internet access.

## Why offline-by-default?

This project was built in a sandboxed environment that could not reach `huggingface.co` to download
pretrained model weights (only PyPI/npm/GitHub were reachable). Rather than ship something that only
*claims* to work, the default backends were chosen to be genuinely runnable anywhere, with **no model
downloads required**:

- **Embeddings:** TF-IDF → Truncated SVD (Latent Semantic Analysis) — real dense vectors, no download.
- **Generation:** Deterministic grounded sentence-extraction from the retrieved context — no LLM download.

Both are swappable with one CLI flag once you have full internet access (see [Upgrading](#upgrading-to-online-backends) below).

## Pipeline Architecture

```
Documents (pdf/txt/docx/json)
        │  src/ingestion.py
        ▼
   Raw text per doc
        │  src/chunking.py   (sentence-aware sliding window, configurable size/overlap)
        ▼
   Text chunks
        │  src/embeddings.py (TF-IDF + LSA, or sentence-transformers)
        ▼
   Chunk vectors ──────────────► src/vectorstore.py (FAISS IndexFlatIP, cosine sim,
        │                         + TF-IDF hybrid keyword scoring, auto-falls back to
        │                         NumPy brute-force search if FAISS unavailable)
        ▼
   User query ─── embed ───► similarity_search(query, top_k) ─── candidates
                                         │
                                         ▼
                              src/rerank.py (lexical overlap re-ranking)
                                         │
                                         ▼
                              src/generation.py (extractive / hf-local / openai)
                                         │
                                         ▼
                                 Grounded answer + cited sources
```

All stages are orchestrated by `src/pipeline.py`.

## Project Structure

```
RAG_System/
├── data/sample_docs/         # demo corpus (3 original docs on RAG/vector DBs/embeddings)
├── src/
│   ├── ingestion.py           # Step 1: load pdf/txt/docx/json documents
│   ├── chunking.py            # Step 2: sentence-aware chunking with overlap
│   ├── embeddings.py          # Step 3: TF-IDF+LSA or sentence-transformers backends
│   ├── vectorstore.py         # Step 4: FAISS/NumPy vector store + hybrid search
│   ├── rerank.py              # Step 8: overlap-based / cross-encoder re-ranking
│   ├── generation.py          # Step 7: prompt building + answer generation backends
│   └── pipeline.py            # Orchestrates all of the above end-to-end
├── main.py                    # CLI: interactive or single-question mode
├── evaluate.py                # Runs sample questions, writes validation log + metrics report
├── logs/validation_log.txt    # generated: per-question retrieval + answer trace
├── metrics/system_metrics.md  # generated: chunking/embedding/vector-store/LLM config report
└── requirements.txt
```

## Quickstart

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Ask a single question against the bundled sample corpus
python main.py --ask "What is the main idea of the document?"

# Or start an interactive Q&A session
python main.py

# Run the full evaluation suite (writes logs/validation_log.txt + metrics/system_metrics.md)
python evaluate.py
```

### Using your own documents

```bash
python main.py --data path/to/your/folder --ask "Your question here"
```

Supported file types in that folder: `.pdf`, `.txt`, `.md`, `.docx`, `.json`.


## Upgrading to online backends

Once you have unrestricted internet (your own machine, Colab, GitHub Actions, etc.):

```bash
pip install sentence-transformers transformers torch

# Better embeddings (downloads all-MiniLM-L6-v2 once, ~90MB)
python main.py --embedder sentence-transformers --ask "your question"

# Real generative answers instead of extractive ones (downloads flan-t5-base once)
python main.py --generator hf-local --ask "your question"

# Or plug in OpenAI (needs OPENAI_API_KEY env var + `pip install openai`)
python main.py --generator openai --ask "your question"
```

You can mix and match: `python main.py --embedder sentence-transformers --generator hf-local`.

## Configuration knobs (Instruction 8: optimization experiments)

| Flag | Default | Purpose |
|---|---|---|
| `--chunk-size` | 800 | Target characters per chunk |
| `--chunk-overlap` | 150 | Overlap carried between adjacent chunks |
| `--top-k` | 8 | Chunks fetched from the vector store before re-ranking |
| `--rerank-k` | 3 | Chunks kept after re-ranking, sent to the generator |
| `--no-hybrid` | off | Disable TF-IDF keyword blending, use pure dense search |
| `--no-rerank` | off | Disable the re-ranking pass |
| `--embedder` | `lsa` | `lsa` (offline) or `sentence-transformers` (online) |
| `--generator` | `extractive` | `extractive` (offline), `hf-local`, or `openai` |

Try, e.g.:
```bash
python main.py --chunk-size 400 --chunk-overlap 50 --no-hybrid --ask "What is a vector database?"
```
and compare the retrieved chunks/scores against the defaults to see the effect of each knob —
this is the "system optimization" experimentation instruction 8 asked for.

## Evaluation results (this repo's bundled sample corpus, offline backends)

From the latest `python evaluate.py` run (see `logs/validation_log.txt` and `metrics/system_metrics.md`
for full detail):

- 12 dynamic sample questions across 3 source documents
- **Top-1 retrieval accuracy: 91.67%** (correct source document ranked first)
- **Top-k retrieval accuracy: 100%** (correct source document present in the final context set)
- Average end-to-end latency per question: ~1.4ms (offline backends, tiny demo corpus)

## Key Concepts (recap)

- **Retrieval**: find the most relevant chunks via embeddings + vector similarity search.
- **Augmentation**: inject retrieved chunks into the model's input as context.
- **Generation**: produce an answer grounded in that retrieved context, not just the model's own memorized knowledge.

## License

MIT — feel free to reuse, extend, and adapt.
