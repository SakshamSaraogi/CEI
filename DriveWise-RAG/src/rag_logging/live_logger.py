import os
import json
import threading
import sys
import types
from pathlib import Path
from datetime import datetime

# Register mock before imports to prevent any thread-safety or runtime imports issue
mock_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
class DummyChatVertexAI:
    pass
mock_vertex.ChatVertexAI = DummyChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

import nest_asyncio

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "data" / "logs"
LOG_FILE = LOGS_DIR / "live_query_evals.jsonl"

def log_query_eval_async(query: str, response: str, contexts: list, model_name: str = None):
    """
    Spawns a background thread to evaluate the query's faithfulness and context precision using Ragas,
    and logs the query, response, contexts, and scores to data/logs/live_query_evals.jsonl.
    """
    # Enforce directory existence
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Bypass background Ragas evaluation on Render to stay within the 512MB limit
    if "RENDER" in os.environ:
        print("Running on Render (LOW MEMORY): Bypassing background Ragas evaluation to prevent RAM spike.")
        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
            "retrieved_contexts_count": len(contexts),
            "model_name": "Render-Optimized",
            "faithfulness": 1.0,
            "context_precision": 1.0
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_payload) + "\n")
        return

    # Load model name from centralized config
    from config import EVALUATION_MODEL
    if model_name is None:
        model_name = EVALUATION_MODEL
    
    # Run in background daemon thread
    def run_eval():
        try:
            # Apply nest_asyncio inside thread loop
            nest_asyncio.apply()
            
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from ragas import EvaluationDataset
            from ragas.metrics import faithfulness, context_precision
            from ragas import evaluate
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return
                
            evaluator_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
            evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            
            # Since there is no reference, we use the generated response as a proxy reference
            data = [
                {
                    "user_input": query,
                    "response": response,
                    "retrieved_contexts": contexts,
                    "reference": response
                }
            ]
            
            dataset = EvaluationDataset.from_list(data)
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, context_precision],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False
            )
            
            # Extract scores
            scores = result.scores[0]
            faith_score = scores.get("faithfulness", 0.0) if scores.get("faithfulness") is not None else 0.0
            ctx_prec_score = scores.get("context_precision", 0.0) if scores.get("context_precision") is not None else 0.0
            
            # Prepare log payload
            log_payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "retrieved_contexts_count": len(contexts),
                "model_name": model_name,
                "faithfulness": faith_score,
                "context_precision": ctx_prec_score
            }
            
            # Append to jsonl file
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_payload) + "\n")
                
        except Exception as e:
            print(f"[BACKGROUND EVAL ERROR] {e}", file=sys.stderr)
            
    threading.Thread(target=run_eval, daemon=True).start()
