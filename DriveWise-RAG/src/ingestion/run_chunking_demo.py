import sys
import json
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR))

from extractor import extract_page_data
from chunker import create_chunks

# Ensure stdout is in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

CRETA_PDF = str(PROJECT_ROOT / "data" / "brochures" / "hyundai" / "creta.pdf")

def main():
    print("=============================================================")
    print("DEMO: Narrative Chunking (creta.pdf Page 2)")
    print("=============================================================")
    page2_extracted = extract_page_data(CRETA_PDF, 2)
    page2_chunks = create_chunks(page2_extracted)
    print(f"Total chunks generated: {len(page2_chunks)}")
    for i, chunk in enumerate(page2_chunks):
        print(f"\n--- Chunk {i+1} [ID: {chunk['metadata']['chunk_id']}] ---")
        print(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}")
        print("Content:")
        print(chunk['chunk_text'])
    print("=============================================================\n")

    print("=============================================================")
    print("DEMO: Structure-Aware Table Chunking (creta.pdf Page 17)")
    print("=============================================================")
    page17_extracted = extract_page_data(CRETA_PDF, 17)
    page17_chunks = create_chunks(page17_extracted)
    print(f"Total chunks generated: {len(page17_chunks)}")
    
    # We will show the first 3 chunks (Dimensions, Engine, Transmission etc.)
    for i, chunk in enumerate(page17_chunks[:3]):
        print(f"\n--- Chunk {i+1} [ID: {chunk['metadata']['chunk_id']}] ---")
        print(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}")
        print("Content:")
        print(chunk['chunk_text'])
        print("-" * 50)
    print("=============================================================")

if __name__ == "__main__":
    main()
