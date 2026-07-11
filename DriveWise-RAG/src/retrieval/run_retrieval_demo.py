import os
import sys
from pathlib import Path

# Resolve directories relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR))

from retriever import DriveWiseRetriever

def main():
    print("=============================================================")
    print("DriveWise-RAG Retrieval Pipeline Verification Demo")
    print("=============================================================")
    
    # Initialize retriever
    try:
        retriever = DriveWiseRetriever()
        print("Retriever successfully initialized!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize retriever: {e}")
        return
        
    # List of test cases (query, history)
    test_cases = [
        {
            "id": 1,
            "query": "What is the boot space of the Tata Nexon?",
            "history": []
        },
        {
            "id": 2,
            "query": "Does the Hyundai Creta have a sunroof?",
            "history": []
        },
        {
            "id": 3,
            "query": "Which has a bigger engine, the Tata Nexon or the Mahindra XUV3XO?",
            "history": []
        },
        {
            "id": 4,
            "query": "What about its mileage?",
            "history": [
                {"role": "user", "content": "What is the boot space of the Tata Nexon?"},
                {"role": "assistant", "content": "The Tata Nexon boot space is 350 liters."}
            ]
        }
    ]
    
    for case in test_cases:
        print("\n" + "="*80)
        print(f"TEST CASE {case['id']}: '{case['query']}'")
        print("="*80)
        if case['history']:
            print("Conversation History:")
            for turn in case['history']:
                print(f"  - {turn['role'].upper()}: {turn['content']}")
        else:
            print("Conversation History: None")
            
        print("\nExecuting query understanding & retrieval...")
        res = retriever.query(case['query'], case['history'])
        
        print(f"\n[QUERY UNDERSTANDING RESULTS]")
        print(f"  - Rewritten Query: '{res['rewritten_query']}'")
        print(f"  - Comparison Detected: {res['is_comparison']}")
        
        understanding = res["raw_understanding"]
        if res['is_comparison']:
            print(f"  - Comparison Targets: {[t.model_dump() for t in understanding.comparison_targets]}")
        else:
            print(f"  - Primary Target: {understanding.primary_target.model_dump() if understanding.primary_target else 'None'}")
            
        print(f"\n[RETRIEVED RESULTS PER SOURCE MODEL]")
        for model_key, chunks in res["retrieved_results"].items():
            print(f"\n---> Source Model: {model_key.upper()}")
            if not chunks:
                print("     [WARNING] No data found in corpus for this model target!")
                continue
                
            print(f"     Retrieved {len(chunks)} chunks (Showing top 3 with scores):")
            for idx, chunk in enumerate(chunks[:3]):
                print(f"\n     {idx+1}. Chunk ID: {chunk['metadata']['chunk_id']}")
                print(f"        Section Type: {chunk['metadata']['section_type']}")
                print(f"        Page: {chunk['metadata']['page_number']}")
                print(f"        Dense Rank: {chunk['dense_rank']} | Sparse Rank: {chunk['sparse_rank']}")
                print(f"        RRF Score: {chunk['rrf_score']:.6f} | Rerank Score: {chunk['rerank_score']:.4f}")
                print(f"        Grounding Confidence: {chunk['grounding_confidence']*100:.2f}%")
                
                # Truncate text snippet for readable CLI output
                text_snippet = chunk['chunk_text'].strip().replace('\n', ' ')
                if len(text_snippet) > 250:
                    text_snippet = text_snippet[:250] + "..."
                print(f"        Snippet: {text_snippet}")
                
    print("\n" + "="*80)
    print("End of retrieval pipeline demonstration.")
    print("="*80)

if __name__ == "__main__":
    main()
