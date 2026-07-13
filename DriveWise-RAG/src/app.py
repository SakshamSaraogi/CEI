import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz  # PyMuPDF

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.append(str(PROJECT_ROOT / "retrieval"))
    sys.path.append(str(PROJECT_ROOT / "generation"))

from rag_logging.sqlite_logger import SQLiteLogger
from config import GENERATION_MODEL

app = FastAPI(title="DriveWise-RAG API", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Serves cover images statically
IMAGES_DIR = PROJECT_ROOT.parent / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serves brand logos statically
LOGOS_DIR = PROJECT_ROOT.parent / "data" / "logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/logos", StaticFiles(directory=str(LOGOS_DIR)), name="logos")

# Serves 3D models statically
MODEL_DIR = PROJECT_ROOT.parent / "data" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/model", StaticFiles(directory=str(MODEL_DIR)), name="model")

# Serves frontend static files
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from fastapi.responses import FileResponse
@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

# Logger instance
logger = SQLiteLogger()

@app.on_event("startup")
def startup_event():
    # Eagerly initialize RAG components on boot if not in mock mode
    if os.environ.get("MOCK_LLM", "true").lower() != "true":
        print("Eagerly initializing RAG components on startup...")
        try:
            get_rag_components()
            print("RAG components initialized successfully!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during eager startup initialization: {e}")

# Pydantic models for chat request/response
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    brand: Optional[str] = None
    model: Optional[str] = None

class CitationSchema(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    document_version: str
    section_type: str

class DisambiguationSchema(BaseModel):
    variant: str
    value: str
    chunk_id: str

class ComparisonSchema(BaseModel):
    models: List[str]
    per_model_answers: Dict[str, str]

class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationSchema]
    confidence: str
    comparison: Optional[ComparisonSchema] = None
    variant_disambiguation: Optional[List[DisambiguationSchema]] = None
    rewritten_query: str
    is_comparison: bool
    latency_ms: float

# Import retriever/generator lazily so mock mode doesn't load heavy modules on startup
_retriever = None
_generator = None

def get_rag_components():
    global _retriever, _generator
    if _retriever is None:
        from retrieval.retriever import DriveWiseRetriever
        from generation.generator import DriveWiseGenerator
        _retriever = DriveWiseRetriever()
        _generator = DriveWiseGenerator(_retriever)
    return _retriever, _generator

# Pre-baked responses for MOCK_LLM mode
MOCK_RESPONSES = {
    "creta_sunroof": ChatResponse(
        answer="According to the Hyundai Creta brochure, the Hyundai Creta is equipped with a Smart Panoramic Sunroof starting from the SX variant. It features voice control commands such as 'Open Sunroof' for easy operation.",
        citations=[
            CitationSchema(
                chunk_id="creta_p12_c2",
                source_file="hyundai_creta_2026.pdf",
                page_number=12,
                document_version="2026",
                section_type="features"
            )
        ],
        confidence="high",
        rewritten_query="Does the Hyundai Creta have a sunroof?",
        is_comparison=False,
        latency_ms=320.0
    ),
    "creta_engine": ChatResponse(
        answer="According to the Hyundai Creta brochure, it is offered with three engine options:\n1. **1.5L MPi Petrol**: 1497 cc, producing 115 PS at 6300 rpm.\n2. **1.5L U2 CRDi Diesel**: 1493 cc, producing 116 PS at 4000 rpm.\n3. **1.5L Turbo GDi Petrol**: 1482 cc, producing 160 PS at 5500 rpm.",
        citations=[
            CitationSchema(
                chunk_id="creta_p31_c1",
                source_file="hyundai_creta_2026.pdf",
                page_number=31,
                document_version="2026",
                section_type="engine"
            )
        ],
        confidence="high",
        rewritten_query="What cc engine is there in Creta?",
        is_comparison=False,
        latency_ms=310.0
    ),
    "nexon_boot": ChatResponse(
        answer="The boot space of the Tata Nexon varies across variants:\n- **Petrol/Diesel variants**: Offers a spacious 382 litres of boot capacity.\n- **CNG variant**: Offers 321 litres of boot capacity, featuring Tata's innovative twin-cylinder technology placed under the luggage area to maximize usable space.",
        citations=[
            CitationSchema(
                chunk_id="nexon_p26_c1",
                source_file="tata_nexon_2026.pdf",
                page_number=26,
                document_version="2026",
                section_type="specifications"
            ),
            CitationSchema(
                chunk_id="nexon_p26_c2",
                source_file="tata_nexon_2026.pdf",
                page_number=26,
                document_version="2026",
                section_type="specifications"
            )
        ],
        confidence="high",
        variant_disambiguation=[
            DisambiguationSchema(variant="Petrol/Diesel", value="382 litres", chunk_id="nexon_p26_c1"),
            DisambiguationSchema(variant="CNG", value="321 litres", chunk_id="nexon_p26_c2")
        ],
        rewritten_query="What is the boot space of the Tata Nexon?",
        is_comparison=False,
        latency_ms=280.0
    ),
    "nexon_engine": ChatResponse(
        answer="According to the Tata Nexon brochure, the vehicle is equipped with:\n- **1.2L Turbocharged Revotron Petrol/CNG engine**: 1199 cc, 3-cylinder engine.\n- **1.5L Turbocharged Revotorq Diesel engine**: 1497 cc, 4-cylinder engine.",
        citations=[
            CitationSchema(
                chunk_id="nexon_p44_c1",
                source_file="tata_nexon_2026.pdf",
                page_number=44,
                document_version="2026",
                section_type="engine"
            )
        ],
        confidence="high",
        rewritten_query="What cc engine is in Tata Nexon?",
        is_comparison=False,
        latency_ms=290.0
    ),
    "compare_nexon_xuv3xo": ChatResponse(
        answer="According to the Tata Nexon brochure, the Nexon is equipped with a 1.2L Turbocharged Revotron engine (1199 cc, 3-cylinder) for Petrol/CNG or a 1.5L Turbocharged Revotorq engine (1497 cc, 4-cylinder) for Diesel. However, there is no brochure or specification data available in our database for the Mahindra XUV3XO.",
        citations=[
            CitationSchema(
                chunk_id="nexon_p44_c1",
                source_file="tata_nexon_2026.pdf",
                page_number=44,
                document_version="2026",
                section_type="engine"
            )
        ],
        confidence="medium",
        comparison=ComparisonSchema(
            models=["nexon", "xuv3xo"],
            per_model_answers={
                "nexon": "1199 cc (Petrol/CNG) / 1497 cc (Diesel)",
                "xuv3xo": "No data found"
            }
        ),
        rewritten_query="Which has a bigger engine, the Tata Nexon or the Mahindra XUV3XO?",
        is_comparison=True,
        latency_ms=450.0
    ),
    "nexon_mileage": ChatResponse(
        answer="I was unable to find specific fuel efficiency or ARAI mileage figures for the Tata Nexon in the retrieved brochure chunks.",
        citations=[],
        confidence="not_found",
        rewritten_query="What is the mileage of the Tata Nexon?",
        is_comparison=False,
        latency_ms=190.0
    )
}

@app.get("/api/brands")
def get_brands():
    brochures_dir = PROJECT_ROOT.parent / "data" / "brochures"
    if not brochures_dir.exists():
        return []
    brands = []
    for brand_dir in brochures_dir.iterdir():
        if brand_dir.is_dir():
            brands.append(brand_dir.name)
    return sorted(brands)

@app.get("/api/models")
def get_models(brand: str):
    brand_dir = PROJECT_ROOT.parent / "data" / "brochures" / brand
    if not brand_dir.exists():
        raise HTTPException(status_code=404, detail=f"Brand {brand} not found.")
    
    models = []
    for pdf_path in brand_dir.glob("*.pdf"):
        model_name = pdf_path.stem
        # Get page count dynamically
        try:
            doc = fitz.open(pdf_path)
            pages = len(doc)
            doc.close()
        except:
            pages = 0
            
        models.append({
            "name": model_name,
            "pages": pages,
            "year": 2026, # Standard brochure version year
            "cover_url": f"/images/{brand}/{model_name}.png"
        })
    return sorted(models, key=lambda x: x["name"])

def is_catalog_query(query: str) -> bool:
    low = query.lower().strip().replace("?", "").replace(".", "")
    patterns = [
        r"what\s+(all\s+)?(cars|models|brands|vehicles|brochures|brochure)\s+(are\s+)?(in|have|do you|exist|supported|available|registered)",
        r"what\s+(is|are|details\s+is|info\s+is)\s+in\s+(your|the|this)\s+database",
        r"what\s+do\s+you\s+have\s+in\s+(your\s+)?database",
        r"what\s+can\s+i\s+ask\s+(you\s+)?about",
        r"list\s+(all\s+)?(the\s+)?(cars|models|brands|vehicles|brochures|database)",
        r"which\s+(cars|models|brands|vehicles)\s+do\s+you\s+(know|have|support)",
        r"tell\s+me\s+(about\s+)?what\s+(cars|models|brands|brochures)\s+you\s+(have|know|support)",
        r"show\s+me\s+the\s+(list\s+of\s+)?(cars|models|brands|vehicles|brochures)",
        r"what\s+cars\s+are\s+there",
        r"what\s+models\s+are\s+there",
        r"what\s+brands\s+are\s+there",
        r"what\s+is\s+your\s+catalog",
        r"show\s+catalog"
    ]
    for p in patterns:
        if re.search(p, low):
            return True
            
    if "database" in low and ("what" in low or "list" in low or "show" in low or "tell" in low or "in" in low):
        return True
    if ("cars" in low or "models" in low or "brands" in low or "brochures" in low) and ("do you have" in low or "are available" in low or "list of" in low or "what all" in low):
        return True
    return False

def get_database_catalog() -> Dict[str, List[str]]:
    brochures_dir = PROJECT_ROOT.parent / "data" / "brochures"
    catalog = {}
    if brochures_dir.exists():
        for brand_dir in brochures_dir.iterdir():
            if brand_dir.is_dir():
                brand_name = brand_dir.name
                brand_display = brand_name.title()
                if brand_display.lower() == "maruti-suzuki":
                    brand_display = "Maruti Suzuki"
                
                models = []
                for pdf_path in brand_dir.glob("*.pdf"):
                    model_display = pdf_path.stem.replace("-", " ").title()
                    model_display = model_display.replace("Ev", "EV").replace("Cng", "CNG")
                    models.append(model_display)
                if models:
                    catalog[brand_display] = sorted(models)
    return catalog

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()
    
    # 0. Intercept catalog/meta queries immediately
    if is_catalog_query(request.query):
        catalog = get_database_catalog()
        lines = ["Our database contains the following brochures across Maruti Suzuki, Hyundai, and Tata:\n"]
        for brand, models in sorted(catalog.items()):
            lines.append(f"- **{brand}**: {', '.join(models)}")
        answer = "\n".join(lines)
        
        resp = ChatResponse(
            answer=answer,
            citations=[],
            confidence="high",
            comparison=None,
            variant_disambiguation=None,
            rewritten_query=request.query,
            is_comparison=False,
            latency_ms=(time.time() - start_time) * 1000.0
        )
        
        # Log to DB
        logger.log_query(
            query=request.query,
            rewritten_query=resp.rewritten_query,
            brand=request.brand,
            model=request.model,
            is_comparison=resp.is_comparison,
            confidence=resp.confidence,
            latency_ms=resp.latency_ms,
            tokens_used=50,
            response_json=resp.model_dump_json()
        )
        return resp
        
    mock_llm = os.environ.get("MOCK_LLM", "true").lower() == "true"
    
    # 1. Format history list
    history_list = [{"role": m.role, "content": m.content} for m in request.history]
    
    if mock_llm:
        # Determine query type and filters for mock results
        q_lower = request.query.lower()
        model_filter = (request.model or "").lower()
        
        # Scenario A: Comparison query containing both Nexon and XUV3XO
        if ("nexon" in q_lower and "xuv3xo" in q_lower) or ("nexon" in q_lower and "mahindra" in q_lower):
            resp = MOCK_RESPONSES["compare_nexon_xuv3xo"]
            
        # Scenario B: Creta specific queries
        elif model_filter == "creta" or "creta" in q_lower:
            if "sunroof" in q_lower:
                resp = MOCK_RESPONSES["creta_sunroof"]
            elif "engine" in q_lower or "cc" in q_lower:
                resp = MOCK_RESPONSES["creta_engine"]
            else:
                resp = ChatResponse(
                    answer=f"This is a mock response: No pre-baked scenario matches this Creta query: '{request.query}'. Please ask about 'sunroof' or 'engine/cc'.",
                    citations=[],
                    confidence="not_found",
                    rewritten_query=request.query,
                    is_comparison=False,
                    latency_ms=100.0
                )
                
        # Scenario C: Nexon specific queries
        elif model_filter == "nexon" or "nexon" in q_lower:
            if "boot" in q_lower or "space" in q_lower or "luggage" in q_lower:
                resp = MOCK_RESPONSES["nexon_boot"]
            elif "engine" in q_lower or "cc" in q_lower:
                resp = MOCK_RESPONSES["nexon_engine"]
            elif "mileage" in q_lower or "efficiency" in q_lower or "kmpl" in q_lower:
                resp = MOCK_RESPONSES["nexon_mileage"]
            else:
                resp = ChatResponse(
                    answer=f"This is a mock response: No pre-baked scenario matches this Nexon query: '{request.query}'. Please ask about 'boot space', 'engine/cc', or 'mileage'.",
                    citations=[],
                    confidence="not_found",
                    rewritten_query=request.query,
                    is_comparison=False,
                    latency_ms=100.0
                )
                
        # Scenario D: Fallback for any other query or unrecognized model
        else:
            resp = ChatResponse(
                answer=f"This is a mock response — no matching demo scenario was found for query: '{request.query}' and model filter: '{request.brand or 'All'}/{request.model or 'All'}'.",
                citations=[],
                confidence="not_found",
                rewritten_query=request.query,
                is_comparison=False,
                latency_ms=100.0
            )
            
        # Log to DB
        logger.log_query(
            query=request.query,
            rewritten_query=resp.rewritten_query,
            brand=request.brand,
            model=request.model,
            is_comparison=resp.is_comparison,
            confidence=resp.confidence,
            latency_ms=resp.latency_ms,
            tokens_used=120,
            response_json=resp.model_dump_json()
        )
        return resp
    else:
        # Check if API Key is missing for real RAG mode
        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="GEMINI_API_KEY is missing in your .env file. Please add your key, or toggle 'Mock Generation Mode' to ON in Settings to explore the dashboard offline."
            )
        elif provider == "groq" and not os.environ.get("GROQ_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY is missing in your .env file. Please add your key, or toggle 'Mock Generation Mode' to ON in Settings to explore the dashboard offline."
            )
        elif provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OPENROUTER_API_KEY is missing in your .env file. Please add your key, or toggle 'Mock Generation Mode' to ON in Settings to explore the dashboard offline."
            )
        # Real RAG Pipeline execution
        try:
            retriever, generator = get_rag_components()
            result = generator.generate(request.query, history_list)
            latency = (time.time() - start_time) * 1000.0
            
            # Map comparison
            comp = None
            if result.comparison:
                comp = ComparisonSchema(
                    models=result.comparison.models,
                    per_model_answers=result.comparison.per_model_answers
                )
            
            # Map variant disambiguation
            v_disambig = None
            if result.variant_disambiguation:
                v_disambig = [
                    DisambiguationSchema(
                        variant=v.variant,
                        value=v.value,
                        chunk_id=v.chunk_id
                    ) for v in result.variant_disambiguation
                ]
                
            resp = ChatResponse(
                answer=result.answer,
                citations=[
                    CitationSchema(
                        chunk_id=c.chunk_id,
                        source_file=c.source_file,
                        page_number=c.page_number,
                        document_version=c.document_version,
                        section_type=c.section_type
                    ) for c in result.citations
                ],
                confidence=result.confidence,
                comparison=comp,
                variant_disambiguation=v_disambig,
                rewritten_query=result.rewritten_query or request.query,
                is_comparison=result.is_comparison or False,
                latency_ms=latency
            )
            
            # Log real query
            logger.log_query(
                query=request.query,
                rewritten_query=resp.rewritten_query,
                brand=request.brand,
                model=request.model,
                is_comparison=resp.is_comparison,
                confidence=resp.confidence,
                latency_ms=latency,
                tokens_used=450,
                response_json=resp.model_dump_json()
            )
            return resp
        except Exception as e:
            import traceback
            traceback.print_exc() # Force full traceback print in Render logs
            logger.log_query(
                query=request.query,
                brand=request.brand,
                model=request.model,
                error_msg=str(e)
            )
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
def get_analytics():
    return logger.get_analytics()

@app.get("/api/logs")
def get_logs(limit: int = Query(50, ge=1, le=100)):
    return logger.get_logs(limit)

@app.get("/api/settings")
def get_settings():
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "groq":
        from config import GROQ_MODEL
        model_name = GROQ_MODEL
    elif provider == "openrouter":
        from config import OPENROUTER_MODEL
        model_name = OPENROUTER_MODEL
    elif provider == "ollama":
        model_name = os.environ.get("OLLAMA_MODEL", "llama3")
    else:
        model_name = GENERATION_MODEL
        
    return {
        "mock_llm": os.environ.get("MOCK_LLM", "true").lower() == "true",
        "model_name": f"{provider.upper()} ({model_name})"
    }

@app.post("/api/settings")
def set_settings(settings: Dict[str, Any]):
    mock_val = settings.get("mock_llm", True)
    os.environ["MOCK_LLM"] = "true" if mock_val else "false"
    return {"status": "success", "mock_llm": mock_val}

@app.post("/api/logs/clear")
def clear_logs():
    with logger.conn:
        logger.conn.execute("DELETE FROM ratings")
        logger.conn.execute("DELETE FROM query_logs")
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    # Make sure env is set
    if "MOCK_LLM" not in os.environ:
        os.environ["MOCK_LLM"] = "true"
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
