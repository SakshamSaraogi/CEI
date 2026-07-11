import os
import sys
import json
from pathlib import Path

# Setup system path to resolve relative imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.append(str(PROJECT_ROOT / "src" / "generation"))

from retriever import DriveWiseRetriever
from generator import DriveWiseGenerator

def main():
    print("=============================================================")
    print("DriveWise-RAG End-to-End Generation Verification Demo")
    print("=============================================================")
    
    # 1. Initialize retriever & generator
    try:
        retriever = DriveWiseRetriever()
        generator = DriveWiseGenerator(retriever)
        print("Retriever and Generator successfully initialized!")
        print(f"Active Provider: {generator.provider.upper()}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline components: {e}")
        return
        
    # 2. Define test cases
    test_cases = [
        {
            "id": 1,
            "query": "What is the boot space of the Tata Nexon?",
            "history": [],
            "description": "Tests single-model variant disambiguation (321L CNG vs 382L Petrol/Diesel)"
        },
        {
            "id": 2,
            "query": "Does the Hyundai Creta have a sunroof?",
            "history": [],
            "description": "Tests standard high-confidence factual retrieval and citation mapping"
        },
        {
            "id": 3,
            "query": "Which has a bigger engine, the Tata Nexon or the Mahindra XUV3XO?",
            "history": [],
            "description": "Tests cross-model comparison with missing corpus data targets"
        },
        {
            "id": 4,
            "query": "What is the mileage of the Tata Nexon?",
            "history": [],
            "description": "Tests low-confidence retrieval handling (factual info missing in brochure)"
        }
    ]
    
    # 3. Execute queries and print structured output
    for case in test_cases:
        print("\n" + "="*80)
        print(f"TEST CASE {case['id']}: '{case['query']}'")
        print(f"Description: {case['description']}")
        print("="*80)
        
        print("Running end-to-end retrieval and generation...")
        try:
            res = generator.generate(case["query"], case["history"])
            
            # Print structured JSON output
            print("\nStructured JSON Response:")
            print(json.dumps(res.model_dump(), indent=2))
            
            # --- Dynamic Grounding Validation Sanity Check ---
            import re
            citations = res.citations
            answer = res.answer
            numbers = re.findall(r"\d{3,}", answer)
            
            print("\nGrounding & Citation Sanity Check:")
            if not numbers:
                print("  [INFO] No numeric specifications found in answer to validate.")
            elif not citations:
                print("  [WARNING] Numeric specifications found in answer, but citation list is empty!")
            else:
                chunk_map = {c["metadata"]["chunk_id"]: c["chunk_text"] for c in retriever.all_chunks}
                for num in numbers:
                    grounded = False
                    matching_cited = []
                    for cit in citations:
                        cid = cit.chunk_id
                        if cid in chunk_map:
                            if num in chunk_map[cid]:
                                grounded = True
                                matching_cited.append(cid)
                    if grounded:
                        print(f"  [SUCCESS] Numeric spec '{num}' is grounded in cited chunk(s): {matching_cited}")
                    else:
                        print(f"  [FAILURE] Numeric spec '{num}' in answer is NOT verbatim present in any cited chunk! Cited chunk IDs: {[c.chunk_id for c in citations]}")
        except Exception as e:
            print(f"[ERROR] Failed to generate response for test case {case['id']}: {e}")
            
    print("\n" + "="*80)
    print("End of generation layer verification.")
    print("="*80)

if __name__ == "__main__":
    main()
