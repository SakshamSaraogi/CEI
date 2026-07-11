import os
import re
import math
import pickle
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Load env variables (api key is in .env at project root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# --- Path Configurations ---
BM25_PATH = PROJECT_ROOT / "data" / "bm25_index.pkl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"

# --- Tokenizer Helper for BM25 ---
def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

# --- Sigmoid Helper for Confidence ---
def compute_sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))

def is_empty_page_chunk(text: str) -> bool:
    """
    Detects if a chunk contains description text for a scanned/image page
    that does not contain any specifications or tabular data.
    """
    low = text.lower()
    empty_phrases = [
        "there is no tabular or specification data",
        "no tabular or specification data present",
        "promotional advertisement",
        "promotional marketing graphic",
        "promotional graphic",
        "marketing graphic"
    ]
    return any(p in low for p in empty_phrases)

def get_disclaimer_demotion(text: str) -> float:
    """
    Applies a score penalty to chunks that represent fine print footnotes,
    disclaimers, warranty conditions, or generic market segment definitions.
    """
    low = text.lower()
    
    # 1. Check for footers/footnotes and terms boilerplate
    is_footnote = (low.strip().startswith("*") or "disclaimer:" in low or "terms & conditions" in low) and len(text) < 300
    
    # 2. Check for generic segment definition text (misattributing other cars' specs)
    is_segment_def = "segment is defined by" in low or "comparable" in low or "suvs whose length" in low or "engine capacity from" in low or "displacement from" in low
    
    if is_footnote or is_segment_def:
        return -15.0  # Apply score demotion penalty
    return 0.0

def get_lexical_boost(query: str, chunk_text: str, search_keywords: Optional[List[str]] = None) -> float:
    """
    Computes a lexical boost based on exact query keyword matches to help
    overcome cross-encoder prose bias on structured/specification tables.
    Uses dynamic keywords if provided, else falls back to query text content words.
    Excludes brand/model names to focus boost strictly on attribute match.
    """
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    
    stop_words = {
        "what", "is", "the", "of", "does", "have", "a", "which", "bigger", "smaller", 
        "or", "about", "its", "are", "than", "in", "on", "for", "with"
    }
    
    brand_model_terms = {
        "tata", "nexon", "hyundai", "creta", "maruti", "suzuki", "swift", 
        "mahindra", "xuv3xo", "harrier", "safari", "punch", "tiago", "altroz",
        "exter", "alcazar", "fronx", "baleno", "brezza", "dzire", "ertiga", "jimny"
    }
    
    if search_keywords:
        keywords = [kw.lower() for kw in search_keywords if kw.lower() not in stop_words and kw.lower() not in brand_model_terms]
    else:
        keywords = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stop_words and w.lower() not in brand_model_terms]
        
    matched_keywords = [kw for kw in keywords if kw in chunk_words]
    
    # Base match boost: 1.0 per matching keyword
    boost = len(matched_keywords) * 1.0
    
    # If the user query has multiple target keywords (like 'boot' and 'space'),
    # we want to heavily reward chunks that contain ALL of them
    if len(keywords) > 1 and len(matched_keywords) == len(keywords):
        boost += 10.0
    elif len(matched_keywords) > 0:
        boost += 8.0 * len(matched_keywords)
        
    return boost



# --- Pydantic Schemas for Query Understanding ---
class Target(BaseModel):
    brand: Optional[str] = Field(None, description="The normalized brand name, e.g. tata, hyundai, maruti-suzuki, mahindra")
    model: Optional[str] = Field(None, description="The normalized model name, e.g. nexon, creta, swift, xuv3xo")

class QueryUnderstandingResult(BaseModel):
    rewritten_query: str = Field(..., description="The rewritten standalone query string resolving references/pronouns")
    is_comparison: bool = Field(..., description="Whether the query compares multiple models or brands")
    primary_target: Optional[Target] = Field(None, description="The primary brand/model target for single-model queries")
    comparison_targets: List[Target] = Field(default_factory=list, description="List of brand/model targets being compared")
    search_keywords: List[str] = Field(default_factory=list, description="Extract 2-4 important noun keywords/attributes being searched from the query, e.g. ['boot', 'space'], ['sunroof'], ['mileage'], ['engine'], ['torque'], ['ground', 'clearance'], ['dimensions']")


# --- Pipeline Initialization Class ---
class DriveWiseRetriever:
    def __init__(self):
        # 1. Initialize Gemini Client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
        self.genai_client = genai.Client(api_key=api_key)
        
        # Load model name from centralized config
        from config import QUERY_UNDERSTANDING_MODEL
        self.model_name = QUERY_UNDERSTANDING_MODEL
        
        # 2. Load BM25 Index
        if not BM25_PATH.exists():
            raise FileNotFoundError(f"BM25 index file not found at: {BM25_PATH}. Please run ingest_corpus.py first.")
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.all_chunks = bm25_data["chunks"]
        
        # 3. Load Qdrant Client
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(f"Qdrant DB directory not found at: {QDRANT_PATH}. Please run ingest_corpus.py first.")
        self.qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        self.collection_name = "car_brochures"
        
        # 4. Initialize Local Embedding & Reranker Models
        print("Initializing Dense Embedding Model (BAAI/bge-large-en-v1.5)...")
        self.embed_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
        print("Initializing Cross-Encoder Reranker Model (Xenova/ms-marco-MiniLM-L-6-v2)...")
        self.rerank_model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        
        # 5. Dynamically build brand_model_map from indexed chunks
        self.brand_model_map = {}
        for chunk in self.all_chunks:
            b = chunk["metadata"]["brand"].lower()
            m = chunk["metadata"]["model"].lower()
            # Standard model key
            self.brand_model_map[m] = (b, m)
            # Keys with spaces replacing dashes (e.g. "alto-k10" -> "alto k10")
            m_space = m.replace("-", " ")
            self.brand_model_map[m_space] = (b, m)
            # Keys with dashes removed (e.g. "alto-k10" -> "altok10")
            m_nospace = m.replace("-", "")
            self.brand_model_map[m_nospace] = (b, m)
            # Brand model combinations
            self.brand_model_map[f"{b} {m}"] = (b, m)
            self.brand_model_map[f"{b} {m_space}"] = (b, m)
            
        # Add custom brand-model aliases/typos
        self.brand_model_map["tigog"] = ("tata", "tiago")
        self.brand_model_map["vitara"] = ("maruti-suzuki", "grand-vitara")
        
    def rewrite_and_understand_query(self, query: str, history: List[Dict[str, str]] = None) -> QueryUnderstandingResult:
        """
        Sends the query and conversation history to Gemini to resolve pronouns/references,
        normalize brands/models, and detect comparisons.
        """
        history = history or []
        hist_text = ""
        for turn in history:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            hist_text += f"{role}: {content}\n"
            
        prompt = f"""
        Analyze the current user query along with the conversation history. Perform these tasks:
        1. Rewrite the current query to resolve any pronouns or implicit references (like "its mileage", "what about dimensions?") to make it a standalone search query.
        2. Detect if the user query is a comparison between multiple car models/brands.
        3. Identify the target brands and models mentioned in the query. Normalize brands to lowercase (e.g. "tata", "hyundai", "maruti-suzuki", "mahindra") and models to lowercase (e.g. "nexon", "creta", "swift").
        4. Extract 2-4 key noun attribute keywords from the rewritten query that represent the core features or attributes being searched (e.g., 'boot', 'space', 'sunroof', 'mileage', 'engine', 'torque', 'dimensions', 'clearance') and output them in "search_keywords".
        
        Rules for targets:
        - Normalization: Hyundai -> hyundai, Maruti Suzuki -> maruti-suzuki, Tata -> tata, Mahindra -> mahindra.
        - If no brand is explicitly mentioned but the model is known (e.g. "nexon"), resolve its brand (e.g. "tata") based on general knowledge or conversation history.
        - For non-comparison queries, fill in "primary_target". Set "comparison_targets" to empty list.
        - For comparison queries, fill in "comparison_targets". Set "primary_target" to null.
        
        Conversation History:
        {hist_text if hist_text else "None"}
        
        Current Query:
        {query}
        """
        
        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryUnderstandingResult
            )
        )
        
        # Robust JSON cleaning
        raw_text = response.text.strip() if response.text else "{}"
        first_idx = raw_text.find('{')
        last_idx = raw_text.rfind('}')
        if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
            clean_text = raw_text[first_idx:last_idx+1]
        else:
            clean_text = raw_text
            
        return QueryUnderstandingResult.model_validate_json(clean_text)
        
    def retrieve_for_target(self, query: str, brand: Optional[str], model: Optional[str], search_keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top 5 chunks for a specific normalized brand & model target.
        Applies hard pre-filters to both Qdrant and BM25.
        """
        import time
        t_start = time.time()
        
        # --- Handle empty target bounds (corpus verification) ---
        # If brand or model is specified but not in our corpus folders, return early empty
        brand_normalized = brand.lower().strip() if brand else None
        model_normalized = model.lower().strip() if model else None
        
        # Check if the brand exists under our corpus folders
        if brand_normalized and model_normalized:
            # Check if we have chunks matching this exact brand/model
            has_matching_chunks = any(
                c["metadata"]["brand"].lower() == brand_normalized
                and c["metadata"]["model"].lower() == model_normalized
                for c in self.all_chunks
            )
            if not has_matching_chunks:
                # Return empty list immediately to signal missing data gracefully
                return []
                
        # --- 1. Dense Retrieval (Qdrant) ---
        t_dense_start = time.time()
        query_vector = list(self.embed_model.embed([query]))[0].tolist()
        
        qdrant_filter = None
        if brand_normalized and model_normalized:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key="metadata.brand", match=MatchValue(value=brand_normalized)),
                    FieldCondition(key="metadata.model", match=MatchValue(value=model_normalized))
                ]
            )
            
        qdrant_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=50  # Fetch more to allow room for filtering
        )
        qdrant_results = [
            p for p in qdrant_response.points
            if not is_empty_page_chunk(p.payload["chunk_text"])
        ][:20]
        t_dense_end = time.time()
        
        # --- 2. Sparse Retrieval (BM25) ---
        t_sparse_start = time.time()
        query_tokens = tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        scored_chunks = []
        for idx, chunk in enumerate(self.all_chunks):
            scored_chunks.append({
                "chunk": chunk,
                "bm25_score": float(bm25_scores[idx])
            })
            
        if brand_normalized and model_normalized:
            filtered_scored = [
                sc for sc in scored_chunks
                if sc["chunk"]["metadata"]["brand"].lower() == brand_normalized
                and sc["chunk"]["metadata"]["model"].lower() == model_normalized
                and not is_empty_page_chunk(sc["chunk"]["chunk_text"])
            ]
        else:
            filtered_scored = [
                sc for sc in scored_chunks
                if not is_empty_page_chunk(sc["chunk"]["chunk_text"])
            ]
            
        filtered_scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        bm25_results = filtered_scored[:20]
        t_sparse_end = time.time()
        
        # --- 3. Reciprocal Rank Fusion (RRF) ---
        t_rrf_start = time.time()
        dense_ranked = [r.payload["metadata"]["chunk_id"] for r in qdrant_results]
        sparse_ranked = [r["chunk"]["metadata"]["chunk_id"] for r in bm25_results]
        
        chunk_lookup = {}
        for r in qdrant_results:
            cid = r.payload["metadata"]["chunk_id"]
            chunk_lookup[cid] = {
                "chunk_text": r.payload["chunk_text"],
                "metadata": r.payload["metadata"],
                "dense_score": r.score
            }
        for r in bm25_results:
            cid = r["chunk"]["metadata"]["chunk_id"]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = {
                    "chunk_text": r["chunk"]["chunk_text"],
                    "metadata": r["chunk"]["metadata"],
                    "dense_score": None
                }
            chunk_lookup[cid]["sparse_score"] = r["bm25_score"]
            
        rrf_scores = {}
        k = 60
        for cid in chunk_lookup:
            dense_rank = dense_ranked.index(cid) + 1 if cid in dense_ranked else None
            sparse_rank = sparse_ranked.index(cid) + 1 if cid in sparse_ranked else None
            
            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += 1.0 / (k + dense_rank)
            if sparse_rank is not None:
                rrf_score += 1.0 / (k + sparse_rank)
                
            rrf_scores[cid] = rrf_score
            chunk_lookup[cid]["dense_rank"] = dense_rank
            chunk_lookup[cid]["sparse_rank"] = sparse_rank
            chunk_lookup[cid]["rrf_score"] = rrf_score
            
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_rrf_chunks = [chunk_lookup[cid] for cid in sorted_cids[:20]]
        t_rrf_end = time.time()
        
        # --- 4. Cross-Encoder Re-Ranking ---
        t_rerank_start = time.time()
        if not top_rrf_chunks:
            return []
            
        texts_to_score = [c["chunk_text"] for c in top_rrf_chunks]
        rerank_scores = list(self.rerank_model.rerank(query, texts_to_score))
        
        for idx, chunk in enumerate(top_rrf_chunks):
            raw_score = float(rerank_scores[idx])
            boost = get_lexical_boost(query, chunk["chunk_text"], search_keywords)
            
            # Structure boost: prioritize structured specification tables over narrative paragraphs
            structure_boost = 0.0
            sec_type = chunk["metadata"].get("section_type", "narrative")
            if sec_type != "narrative" and sec_type != "scanned":
                structure_boost = 3.5
                
            demotion = get_disclaimer_demotion(chunk["chunk_text"])
            final_score = raw_score + boost + structure_boost + demotion
            chunk["rerank_score"] = final_score
            chunk["grounding_confidence"] = compute_sigmoid(final_score)
            
        top_rrf_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        t_rerank_end = time.time()
        
        print(f"[TIMING] retrieve_for_target ({brand_normalized or 'all'}/{model_normalized or 'all'}):")
        print(f"  - Dense Vector: {(t_dense_end - t_dense_start)*1000:.1f}ms")
        print(f"  - Sparse BM25: {(t_sparse_end - t_sparse_start)*1000:.1f}ms")
        print(f"  - RRF Fusion: {(t_rrf_end - t_rrf_start)*1000:.1f}ms")
        print(f"  - Cross-Encoder: {(t_rerank_end - t_rerank_start)*1000:.1f}ms")
        print(f"  - Total retrieve_for_target: {(time.time() - t_start)*1000:.1f}ms")
        
        return top_rrf_chunks[:5]
        
    def extract_targets_locally(self, query: str, history: List[Dict[str, str]] = None) -> List[Tuple[str, str]]:
        query_lower = query.lower()
        
        # If it is a cross-corpus general query, do not fall back to history target
        is_cross_corpus = any(p in query_lower for p in ["which car", "which model", "which brand", "which vehicle", "which tata", "which hyundai", "which maruti", "which of the", "which of these"])
        
        targets = []
        
        # Check current query first against dynamic brand_model_map
        for model_key, (brand, mdl) in self.brand_model_map.items():
            # Check for exact word boundary matches to avoid false positives (e.g. "exter" in "external")
            pattern = rf"\b{re.escape(model_key)}\b"
            if re.search(pattern, query_lower):
                targets.append((brand, mdl))
                
        # If no targets found in current query, check conversation history (most recent first)
        if not targets and history and not is_cross_corpus:
            for turn in reversed(history):
                content_lower = turn.get("content", "").lower()
                for model_key, (brand, mdl) in self.brand_model_map.items():
                    pattern = rf"\b{re.escape(model_key)}\b"
                    if re.search(pattern, content_lower):
                        targets.append((brand, mdl))
                        break
                if targets:
                    break
                    
        # Remove duplicate target tuples while preserving order
        unique_targets = []
        for t in targets:
            if t not in unique_targets:
                unique_targets.append(t)
        return unique_targets

    def extract_keywords_locally(self, query: str) -> List[str]:
        # Simple stopword filtering to get key search terms
        stopwords = {
            "what", "is", "the", "of", "a", "does", "have", "in", "on", "for", "with", 
            "about", "its", "how", "compare", "and", "or", "to", "than", "between", "details",
            "show", "tell", "me", "find", "get", "has", "do", "any", "are", "by", "from", "at"
        }
        # Remove punctuation
        clean_query = re.sub(r"[^\w\s]", "", query.lower())
        words = clean_query.split()
        keywords = [w for w in words if w not in stopwords]
        
        # Expand keywords for common variants across all major spec categories
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            
            # Dimensions & size
            if any(term in kw for term in ["dimension", "size", "big", "large", "length", "width", "height", "wheelbase", "clearance", "lwh", "l*w*h"]):
                expanded.extend(["dimension", "dimensions", "l*w*h", "lwh", "length", "width", "height", "wheelbase", "clearance", "overall", "mm", "unladen"])
            
            # Safety
            elif any(term in kw for term in ["safety", "safe", "airbag", "abs", "ebd", "esc", "esp", "camera", "sensor", "adas"]):
                expanded.extend(["safety", "safe", "airbag", "airbags", "abs", "ebd", "esc", "esp", "camera", "sensor", "sensors", "adas", "assist", "isofix"])
                
            # Infotainment & screens & audio
            elif any(term in kw for term in ["infotainment", "screen", "audio", "sound", "music", "speakers", "display", "touchscreen", "bose", "harman"]):
                expanded.extend(["infotainment", "screen", "audio", "sound", "display", "touchscreen", "speakers", "harman", "bose", "android", "apple", "carplay", "system", "cm", "inch", "connectivity"])
                
            # Transmission & gearbox
            elif any(term in kw for term in ["transmission", "gearbox", "gear", "manual", "automatic", "amt", "ags", "dct", "dca", "cvt", "ivt", "clutch"]):
                expanded.extend(["transmission", "gearbox", "gear", "gears", "manual", "automatic", "amt", "ags", "dct", "dca", "cvt", "ivt", "clutch", "speed"])
                
            # Tyres & wheels
            elif any(term in kw for term in ["tyre", "tire", "wheel", "alloy", "rim"]):
                expanded.extend(["tyre", "tyres", "tire", "tires", "wheel", "wheels", "alloy", "rim", "size", "r14", "r15", "r16", "r17"])
                
            # Brakes & suspension
            elif any(term in kw for term in ["brakes", "brake", "disc", "drum", "suspension", "strut", "spring", "shock"]):
                expanded.extend(["brakes", "brake", "disc", "drum", "suspension", "strut", "spring", "shock", "steering", "turning", "radius", "front", "rear"])
                
            # Engine & power
            elif any(term in kw for term in ["engine", "power", "torque", "displacement", "cc", "cylinder", "ps", "kw", "nm", "turbo", "gdi"]):
                expanded.extend(["engine", "power", "torque", "displacement", "cc", "cylinder", "cylinders", "ps", "rpm", "nm", "turbo", "gdi", "max", "capacity"])
                
            # Boot space & capacities
            elif any(term in kw for term in ["boot", "space", "luggage", "cargo", "capacity", "volume", "litres", "litre", "fuel", "tank"]):
                expanded.extend(["boot", "space", "luggage", "cargo", "capacity", "volume", "litres", "litre", "fuel", "tank", "capacity"])
                
        # Remove duplicates preserving order
        seen = set()
        final_keywords = []
        for kw in expanded:
            if kw not in seen:
                seen.add(kw)
                final_keywords.append(kw)
                
        return final_keywords[:10]

    def query(self, user_query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        The main orchestrator. Performs fast local target resolution and retrieves chunks.
        """
        import time
        t_query_start = time.time()
        
        history = history or []
        
        t_ext_start = time.time()
        targets = self.extract_targets_locally(user_query, history)
        search_keywords = self.extract_keywords_locally(user_query)
        t_ext_end = time.time()
        
        is_comparison = len(targets) > 1
        
        results = {
            "original_query": user_query,
            "is_comparison": is_comparison,
            "retrieved_results": {},
            "search_keywords": search_keywords
        }
        
        # Step 2: Retrieve based on comparison routing
        t_ret_start = time.time()
        if is_comparison:
            for brand, model in targets:
                key = f"{brand}/{model}"
                chunks = self.retrieve_for_target(
                    query=user_query,
                    brand=brand,
                    model=model,
                    search_keywords=search_keywords
                )
                results["retrieved_results"][key] = chunks
        else:
            if targets:
                brand, model = targets[0]
                key = f"{brand}/{model}"
            else:
                brand, model = None, None
                key = "global"
                
            chunks = self.retrieve_for_target(
                query=user_query,
                brand=brand,
                model=model,
                search_keywords=search_keywords
            )
            results["retrieved_results"][key] = chunks
        t_ret_end = time.time()
        
        print(f"[TIMING] DriveWiseRetriever.query Total:")
        print(f"  - Local Resolution (targets + terms): {(t_ext_end - t_ext_start)*1000:.1f}ms")
        print(f"  - Chunks Retrieval (all targets): {(t_ret_end - t_ret_start)*1000:.1f}ms")
        print(f"  - Total Retriever Query: {(time.time() - t_query_start)*1000:.1f}ms")
        
        return results
