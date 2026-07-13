import os
import time
import json
import re
import pickle
import uuid
from pathlib import Path
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# Resolve directories relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Ensure we import our local modules
import sys
sys.path.append(str(SCRIPT_DIR))
from extractor import extract_page_data, STATS
from chunker import create_chunks

DATA_DIR = PROJECT_ROOT / "data" / "brochures"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"
BM25_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"

def tokenize_text(text):
    # Basic word tokenizer for BM25
    return re.findall(r"\w+", text.lower())

def main():
    start_time = time.time()
    
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} directory does not exist.")
        return
        
    print("=============================================================")
    print("DriveWise-RAG: Starting Full-Corpus Ingestion")
    print("=============================================================")
    
    all_chunks = []
    failed_pages = []
    brand_stats = {}
    file_chunks_stats = {}
    
    brands = [d for d in os.listdir(DATA_DIR) if (DATA_DIR / d).is_dir()]
    brands.sort()
    
    for brand in brands:
        brand_path = DATA_DIR / brand
        brand_stats[brand] = 0
        pdf_files = [f for f in os.listdir(brand_path) if f.lower().endswith(".pdf")]
        pdf_files.sort()
        
        print(f"\nProcessing brand: {brand.upper()} ({len(pdf_files)} files)")
        
        for f in pdf_files:
            pdf_path = brand_path / f
            file_chunks_count = 0
            
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
            except Exception as e:
                print(f"  [ERROR] Failed to open {brand}/{f}: {e}")
                failed_pages.append((str(pdf_path), 0, f"Failed to open PDF: {e}"))
                continue
                
            print(f"  Ingesting {f} ({total_pages} pages)...")
            
            for page_num in range(1, total_pages + 1):
                try:
                    # Run extraction & chunking
                    page_data = extract_page_data(str(pdf_path), page_num)
                    chunks = create_chunks(page_data)
                    all_chunks.extend(chunks)
                    file_chunks_count += len(chunks)
                    brand_stats[brand] += len(chunks)
                except Exception as page_err:
                    print(f"    [ERROR] Page {page_num} failed: {page_err}")
                    failed_pages.append((str(pdf_path), page_num, str(page_err)))
                    
            file_chunks_stats[f"{brand}/{f}"] = file_chunks_count
            print(f"    Completed {f}: Generated {file_chunks_count} chunks.")
            
    print("\nExtraction & Chunking phase finished.")
    print(f"Total chunks generated: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks generated. Ingestion aborted.")
        return
        
    # --- Generate Embeddings ---
    print("\n-------------------------------------------------------------")
    print("Generating local embeddings using BAAI/bge-large-en-v1.5 via FastEmbed...")
    print("-------------------------------------------------------------")
    embed_start = time.time()
    
    # Initialize FastEmbed text embedding client
    try:
        # This will automatically download and load the BAAI/bge-small-en-v1.5 ONNX model
        embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding model: {e}")
        return
        
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    print(f"Embedding {len(chunk_texts)} chunks...")
    
    try:
        # Use a small batch size to avoid ONNX Runtime memory allocation errors
        embeddings = list(embed_model.embed(chunk_texts, batch_size=32))
        print(f"Embedding generation completed in {time.time() - embed_start:.2f}s.")
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        return
        
    # --- Store in Qdrant ---
    print("\n-------------------------------------------------------------")
    print("Storing in local Qdrant database...")
    print("-------------------------------------------------------------")
    qdrant_start = time.time()
    
    try:
        # Create parent directory for qdrant
        QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
        qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        
        collection_name = "car_brochures"
        
        # Recreate collection to ensure a clean ingest
        if qdrant_client.collection_exists(collection_name):
            print(f"Recreating existing Qdrant collection: '{collection_name}'...")
            qdrant_client.delete_collection(collection_name)
            
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        points = []
        for idx, chunk in enumerate(all_chunks):
            # Convert numpy array embedding to list of floats
            vector = embeddings[idx].tolist()
            # Generate deterministic UUID based on unique chunk_id to avoid duplication
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_text": chunk["chunk_text"],
                    "metadata": chunk["metadata"]
                }
            ))
            
        # Bulk upsert points in batches
        batch_size = 100
        print(f"Writing {len(points)} points to Qdrant...")
        for i in range(0, len(points), batch_size):
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size]
            )
            
        print(f"Qdrant storage completed in {time.time() - qdrant_start:.2f}s.")
    except Exception as e:
        print(f"[ERROR] Failed to store in Qdrant: {e}")
        return
        
    # --- Build BM25 Index ---
    print("\n-------------------------------------------------------------")
    print("Building BM25 Sparse Index...")
    print("-------------------------------------------------------------")
    bm25_start = time.time()
    
    try:
        tokenized_corpus = [tokenize_text(c["chunk_text"]) for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_data = {
            "bm25": bm25,
            "chunks": all_chunks
        }
        
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25_data, f)
            
        print(f"BM25 index built and pickled at {BM25_PATH} in {time.time() - bm25_start:.2f}s.")
    except Exception as e:
        print(f"[ERROR] Failed to build BM25 index: {e}")
        return
        
    total_time = time.time() - start_time
    
    # --- Final Ingestion Summary Report ---
    print("\n=============================================================")
    print("INGESTION COMPLETE SUMMARY REPORT")
    print("=============================================================")
    print(f"Total Chunk Count: {len(all_chunks)}")
    print(f"Total Ingestion Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\n--- Chunks Per Brand ---")
    for b, count in brand_stats.items():
        print(f"  {b.upper()}: {count} chunks")
        
    print("\n--- Gemini Vision Ingestion Stats ---")
    print(f"  Total Pages Routed to Vision: {STATS['vision_calls'] + STATS['vision_cached']}")
    print(f"  Uncached Vision API Calls Executed: {STATS['vision_calls']}")
    print(f"  Cached Vision Pages Loaded (skipped calls): {STATS['vision_cached']}")
    print(f"  Vision API Call Retries triggered: {STATS['vision_retries']}")
    print(f"  Vision API Successes: {STATS['vision_successes']}")
    print(f"  Vision API Failures (fell back to text): {STATS['vision_failures']}")
    
    print("\n--- Flagged Files (Unexpected Chunk Counts) ---")
    # Identify files with unusually low/high chunks
    flagged_files = []
    for fn, count in file_chunks_stats.items():
        if count == 0:
            flagged_files.append(f"  {fn} produced 0 chunks (Extraction/Format Issue)")
        elif count > 50:
            flagged_files.append(f"  {fn} produced abnormally high chunks ({count} chunks) - Check Layout")
    if flagged_files:
        for fline in flagged_files:
            print(fline)
    else:
        print("  No anomalies detected. All files produced reasonable chunk distributions.")
        
    print("\n--- Failed Pages (Failed to extract or parse) ---")
    if failed_pages:
        print(f"  Total Failed Pages: {len(failed_pages)}")
        for path, page, err in failed_pages[:10]:
            print(f"  - File: {path}, Page: {page}, Error: {err}")
        if len(failed_pages) > 10:
            print(f"  - ... and {len(failed_pages) - 10} more failures.")
    else:
        print("  0 pages failed. Complete extraction coverage!")
    print("=============================================================")

if __name__ == "__main__":
    main()
