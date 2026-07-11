import sys
import types
import os
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# 1. Disable background live evaluation during offline run
os.environ["OFFLINE_EVAL"] = "True"

# 2. Register vertexai mock before any other imports
mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
class DummyChatVertexAI:
    pass
mock_vertex.ChatVertexAI = DummyChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

# 3. Initialize nest_asyncio to prevent asyncio deadlocks
import nest_asyncio
nest_asyncio.apply()

# 4. Load env
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Add src to python path
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "retrieval"))
sys.path.append(str(PROJECT_ROOT / "src" / "generation"))

from retriever import DriveWiseRetriever
from generation.generator import DriveWiseGenerator

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_correctness, context_precision
from ragas import evaluate

def run_with_retry(func, *args, max_retries=10, initial_delay=5, **kwargs):
    """
    Executes a function, catching rate-limiting and quota exhaustion errors,
    retrying with exponential backoff or the suggested retry time if provided.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            # Catch RESOURCE_EXHAUSTED, 429, 500, 503, INTERNAL, or UNAVAILABLE errors
            if ("RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "Quota exceeded" in err_msg 
                or "INTERNAL" in err_msg or "500" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg):
                # Try to parse the suggested retry time
                retry_match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
                wait_time = float(retry_match.group(1)) + 1.0 if retry_match else delay
                print(f"  [WARNING] Rate limit or transient error encountered ({err_msg[:60]}...). Waiting {wait_time:.2f}s before retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                delay = max(delay * 2, wait_time + 1.0)
            else:
                # Re-raise any other exception immediately
                raise e
    raise RuntimeError("Failed after maximum retries due to persistent rate limiting.")

def main():
    print("=============================================================")
    print("DriveWise-RAG Offline Evaluation Runner (RAGAS)")
    print("=============================================================")
    
    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not found in .env.")
        sys.exit(1)
        
    # Create evaluation directory
    eval_dir = PROJECT_ROOT / "data" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_file = PROJECT_ROOT / "src" / "evaluation" / "eval_dataset.json"
    if not dataset_file.exists():
        print(f"[ERROR] Evaluation dataset not found at {dataset_file}")
        sys.exit(1)
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
        
    print(f"Loaded {len(eval_cases)} evaluation cases.")
    
    # Initialize RAG components
    print("Initializing retriever and generator...")
    retriever = DriveWiseRetriever()
    generator = DriveWiseGenerator(retriever)
    
    responses = []
    contexts_list = []
    
    # Loop and run end-to-end pipeline
    print("\nRunning test queries through RAG pipeline...")
    for idx, item in enumerate(eval_cases):
        qid = item["id"]
        query = item["query"]
        history = item["history"]
        category = item["category"]
        
        print(f"[{idx+1}/{len(eval_cases)}] Category: {category} | QID: {qid} | Query: '{query}'")
        
        # 1. Retrieve contexts with rate-limit retries
        try:
            retrieval_res = run_with_retry(retriever.query, query, history)
            contexts = []
            for target, chunks in retrieval_res["retrieved_results"].items():
                for chunk in chunks:
                    contexts.append(chunk["chunk_text"])
            if not contexts:
                contexts = ["No contexts retrieved."]
        except Exception as ret_err:
            print(f"  [ERROR] Retrieval failed for QID {qid}: {ret_err}")
            contexts = ["Retrieval failed."]
            
        # 2. Generate answer with rate-limit retries
        try:
            gen_res = run_with_retry(generator.generate, query, history)
            answer = gen_res.answer
        except Exception as gen_err:
            print(f"  [ERROR] Generation failed for QID {qid}: {gen_err}")
            answer = "Generation failed."
            
        responses.append(answer)
        contexts_list.append(contexts)
        
        # Base sleep to spacing out requests comfortably
        time.sleep(10)
        
    # Prepare RAGAS evaluation dataset list
    print("\nPreparing RAGAS evaluation data...")
    ragas_data = []
    for item, ans, ctxs in zip(eval_cases, responses, contexts_list):
        ragas_data.append({
            "user_input": item["query"],
            "response": ans,
            "retrieved_contexts": ctxs,
            "reference": item["reference"]
        })
    
    # Initialize Evaluators
    print("Initializing RAGAS evaluator LLM and Embeddings...")
    from config import EVALUATION_MODEL
    evaluator_llm = ChatGoogleGenerativeAI(model=EVALUATION_MODEL, google_api_key=api_key)
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # Execute RAGAS evaluation row-by-row with rate-limit retries
    scores_list = []
    print("\nExecuting RAGAS evaluation metrics row-by-row...")
    for idx, item in enumerate(ragas_data):
        print(f"  [{idx+1}/{len(ragas_data)}] Evaluating QID {eval_cases[idx]['id']}...")
        single_dataset = EvaluationDataset.from_list([item])
        
        try:
            result = run_with_retry(
                evaluate,
                dataset=single_dataset,
                metrics=[faithfulness, answer_correctness, context_precision],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=True
            )
            scores_list.append(result.scores[0])
        except Exception as eval_err:
            print(f"  [ERROR] Ragas evaluation failed for QID {eval_cases[idx]['id']}: {eval_err}")
            # Fallback to zero score
            scores_list.append({"faithfulness": 0.0, "answer_correctness": 0.0, "context_precision": 0.0})
            
        time.sleep(5)
        
    # Calculate aggregate scores manually
    f_total, ac_total, cp_total = 0.0, 0.0, 0.0
    f_count, ac_count, cp_count = 0, 0, 0
    
    for s in scores_list:
        if s.get("faithfulness") is not None:
            f_total += s["faithfulness"]
            f_count += 1
        if s.get("answer_correctness") is not None:
            ac_total += s["answer_correctness"]
            ac_count += 1
        if s.get("context_precision") is not None:
            cp_total += s["context_precision"]
            cp_count += 1
            
    avg_faithfulness = f_total / f_count if f_count > 0 else 0.0
    avg_correctness = ac_total / ac_count if ac_count > 0 else 0.0
    avg_precision = cp_total / cp_count if cp_count > 0 else 0.0
    
    # Prepare report details
    report_items = []
    low_scoring_items = []
    
    for idx, item in enumerate(eval_cases):
        eval_scores = scores_list[idx]
        f_score = eval_scores.get("faithfulness", 0.0) if eval_scores.get("faithfulness") is not None else 0.0
        ac_score = eval_scores.get("answer_correctness", 0.0) if eval_scores.get("answer_correctness") is not None else 0.0
        cp_score = eval_scores.get("context_precision", 0.0) if eval_scores.get("context_precision") is not None else 0.0
        
        report_item = {
            "id": item["id"],
            "category": item["category"],
            "query": item["query"],
            "generated_answer": responses[idx],
            "reference": item["reference"],
            "scores": {
                "faithfulness": f_score,
                "answer_correctness": ac_score,
                "context_precision": cp_score
            }
        }
        report_items.append(report_item)
        
        if f_score < 0.6 or ac_score < 0.6 or cp_score < 0.6:
            low_scoring_items.append(report_item)
            
    # Compile JSON report
    json_report = {
        "summary": {
            "total_questions": len(eval_cases),
            "aggregate_scores": {
                "faithfulness": avg_faithfulness,
                "answer_correctness": avg_correctness,
                "context_precision": avg_precision
            }
        },
        "details": report_items
    }
    
    # Save JSON report
    with open(eval_dir / "eval_report.json", "w", encoding="utf-8") as jf:
        json.dump(json_report, jf, indent=2)
        
    # Compile Markdown report
    md_content = f"""# DriveWise-RAG Offline Evaluation Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 1. Summary Metrics
* **Total Questions Evaluated**: {len(eval_cases)}
* **Average Faithfulness**: {avg_faithfulness:.4f}
* **Average Answer Correctness**: {avg_correctness:.4f}
* **Average Context Precision**: {avg_precision:.4f}

---

## 2. Low-Scoring Questions (Score < 0.60)
"""
    if not low_scoring_items:
        md_content += "No questions scored below the 0.60 threshold! All pipeline components are performing optimally.\n"
    else:
        md_content += "| QID | Category | Query | Faithfulness | Answer Correctness | Context Precision |\n"
        md_content += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for item in low_scoring_items:
            md_content += f"| {item['id']} | {item['category']} | {item['query']} | {item['scores']['faithfulness']:.2f} | {item['scores']['answer_correctness']:.2f} | {item['scores']['context_precision']:.2f} |\n"
            
    md_content += "\n---\n\n## 3. Full Question Details\n"
    md_content += "| QID | Category | Query | Faithfulness | Answer Correctness | Context Precision |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for item in report_items:
        md_content += f"| {item['id']} | {item['category']} | {item['query']} | {item['scores']['faithfulness']:.2f} | {item['scores']['answer_correctness']:.2f} | {item['scores']['context_precision']:.2f} |\n"

    # Save Markdown report
    with open(eval_dir / "eval_report.md", "w", encoding="utf-8") as mf:
        mf.write(md_content)
        
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION RUN COMPLETE")
    print("="*50)
    print(f"Total Questions: {len(eval_cases)}")
    print(f"Aggregate Faithfulness:  {avg_faithfulness:.4f}")
    print(f"Aggregate Correctness:   {avg_correctness:.4f}")
    print(f"Aggregate Precision:     {avg_precision:.4f}")
    print("="*50)
    
    if low_scoring_items:
        print(f"\n[WARNING] Found {len(low_scoring_items)} low-scoring questions (< 0.60):")
        for item in low_scoring_items:
            print(f"  - QID {item['id']} ({item['category']}): '{item['query']}'")
            print(f"    Scores -> Faithfulness: {item['scores']['faithfulness']:.2f} | Correctness: {item['scores']['answer_correctness']:.2f} | Precision: {item['scores']['context_precision']:.2f}")
    else:
        print("\n[SUCCESS] All questions scored >= 0.60 threshold!")
        
    print(f"\nScored reports saved at:\n  - Markdown: {eval_dir / 'eval_report.md'}\n  - JSON: {eval_dir / 'eval_report.json'}")

if __name__ == "__main__":
    main()
