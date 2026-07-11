import os
import hashlib
import json
import re
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Resolve directories relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # src/ingestion/extractor.py -> src/ingestion -> src -> PROJECT_ROOT

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Prompt for Vision extraction
EXTRACTION_PROMPT = """Extract the tabular/specification data from this car brochure page.
Reconstruct the page's tabular or specification content as a clean Markdown table (or structured key-value pairs where no clear table exists).
You must explicitly preserve which value belongs to which row, column, and variant.
Do not summarize, skip, or drop any rows or columns. Extract everything with full fidelity.
Provide only the markdown table/data. Do not include markdown code block backticks (like ```markdown) or introductory text in your final response. Just output the clean extracted content.
"""

# Known scanned brochures
SCANNED_FILES = {
    "alto-k10.pdf",
    "eeco.pdf",
    "ertiga.pdf",
    "invicto.pdf",
    "s-presso.pdf",
    "xi6.pdf"
}

# Special symbols highlighting grids/options
GRID_SYMBOLS = {"✓", "✔", "", "●", "■", "◆", "▲"}

# Key specification terms
SPEC_KEYWORDS = {
    "specifications", "dimensions", "technical specifications", 
    "variants", "features list", "powertrain", "transmission", 
    "engine", "key features"
}

# Global statistics tracker
STATS = {
    "vision_calls": 0,
    "vision_retries": 0,
    "vision_successes": 0,
    "vision_failures": 0,
    "vision_cached": 0
}

def classify_page(page, filename: str) -> str:
    """
    Classifies a PDF page into NARRATIVE, SPEC/TABLE, or SCANNED.
    """
    # 1. Check if the entire file is known to be scanned
    if filename.lower() in SCANNED_FILES:
        return "SCANNED"
        
    text = page.get_text()
    total_chars = len(text.strip())
    
    # 2. Check if near-zero text (functionally scanned page)
    if total_chars < 50:
        if len(page.get_images()) > 0:
            return "SCANNED"
        else:
            return "NARRATIVE"  # empty page
            
    # 3. Analyze text characteristics
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    short_lines = [l for l in lines if len(l) < 15]
    short_line_ratio = len(short_lines) / len(lines) if lines else 0
    
    digit_count = sum(c.isdigit() for c in text)
    digit_ratio = digit_count / total_chars if total_chars else 0
    
    # Check for table indicators
    has_spec_kws = any(kw in text.lower() for kw in SPEC_KEYWORDS)
    has_grid_symbols = any(sym in text for sym in GRID_SYMBOLS)
    
    # 4. Heuristics for SPEC/TABLE
    if has_spec_kws:
        return "SPEC/TABLE"
    if digit_ratio > 0.15 and short_line_ratio > 0.5:
        return "SPEC/TABLE"
    if has_grid_symbols and short_line_ratio > 0.4:
        return "SPEC/TABLE"
        
    return "NARRATIVE"

def extract_year_metadata(text: str, filename: str, doc_metadata: dict) -> str:
    """
    Extracts the document version/year using cues from name, metadata, and text.
    Returns None/unknown for jimny.pdf to satisfy safety constraints.
    """
    if "jimny" in filename.lower():
        return None
        
    candidates = set()
    # Check filename
    fn_years = re.findall(r"\b(201[5-9]|202[0-9]|2030)\b", filename)
    candidates.update(fn_years)
    
    # Check document metadata
    meta_str = json.dumps(doc_metadata)
    meta_years = re.findall(r"\b(201[5-9]|202[0-9]|2030)\b", meta_str)
    candidates.update(meta_years)
    
    for key in ['creationDate', 'modDate']:
        val = doc_metadata.get(key)
        if val and isinstance(val, str):
            m = re.search(r"\b(201[5-9]|202[0-9]|2030)", val)
            if m:
                candidates.add(m.group(1))
                
    # Check body text
    my_matches = re.findall(r"\bMY\s*([12][0-9])\b", text, re.IGNORECASE)
    for m in my_matches:
        candidates.add("20" + m)
    text_years = re.findall(r"\b(201[5-9]|202[0-9]|2030)\b", text)
    candidates.update(text_years)
    
    years = sorted(list(candidates))
    return years[0] if years else None

def extract_page_data(pdf_path: str, page_number: int) -> dict:
    """
    Orchestrates extraction of a single page:
    1. Opens PDF.
    2. Runs standard text extraction and classification.
    3. Routes to standard text or Gemini Vision API.
    4. Caches vision results locally.
    """
    pdf_path_obj = Path(pdf_path)
    filename = pdf_path_obj.name
    
    # Get relative file path to project root for metadata and cache key
    try:
        relative_filepath = pdf_path_obj.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_filepath = pdf_path_obj.as_posix()
        
    # Get brand name (assumes folder structure data/brochures/{brand}/{model}.pdf)
    brand = pdf_path_obj.parent.name
    model = pdf_path_obj.stem
    
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    
    classification = classify_page(page, filename)
    doc_metadata = doc.metadata
    
    # Read standard text
    raw_text = page.get_text()
    
    # Extract version metadata
    doc_version = extract_year_metadata(raw_text, filename, doc_metadata)
    
    doc.close()
    
    # Prepare base metadata
    meta = {
        "brand": brand,
        "model": model,
        "variant": None,
        "section_type": classification.lower(),
        "page_number": page_number,
        "document_version": doc_version,
        "source_file": relative_filepath,
        "extraction_method": "text"
    }
    
    if classification == "NARRATIVE":
        return {
            "extracted_text": raw_text.strip(),
            "metadata": meta
        }
        
    # For SPEC/TABLE and SCANNED, we call Gemini Vision
    meta["extraction_method"] = "vision"
    
    # Check prompt-sensitive cache
    cache_dir = PROJECT_ROOT / ".cache" / "vision_extraction"
    os.makedirs(cache_dir, exist_ok=True)
    
    prompt_hash = hashlib.sha256(EXTRACTION_PROMPT.encode("utf-8")).hexdigest()
    combined_key = f"{relative_filepath}_page_{page_number}_{prompt_hash}"
    cache_key = hashlib.sha256(combined_key.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                STATS["vision_cached"] += 1
                return {
                    "extracted_text": cached_data["extracted_markdown"],
                    "metadata": meta
                }
        except Exception:
            pass
            
    # Perform Vision Call
    print(f"Calling Gemini Vision for {brand}/{model}.pdf Page {page_number}...")
    STATS["vision_calls"] += 1
    
    # Render page to PNG bytes
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    
    # Initialize client (uses GEMINI_API_KEY from environment)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
        
    client = genai.Client(api_key=api_key)
    
    # Try models in sequence of preference (using centralized config as primary)
    from config import INGESTION_VISION_MODEL
    models_to_try = [INGESTION_VISION_MODEL, "gemini-3.1-flash-lite", "gemini-flash-latest"]
    # De-duplicate while preserving order
    models_to_try = list(dict.fromkeys(models_to_try))
    extracted_markdown = None
    last_error = None
    used_model = None
    
    import time
    
    for model_name in models_to_try:
        retries = 3
        backoff = 2
        for attempt in range(retries):
            try:
                if attempt > 0:
                    STATS["vision_retries"] += 1
                print(f"Attempting Vision call with {model_name} (Attempt {attempt+1}/{retries})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                        EXTRACTION_PROMPT
                    ]
                )
                extracted_markdown = response.text
                if extracted_markdown is not None:
                    used_model = model_name
                    break
            except Exception as e:
                err_msg = str(e)
                is_transient = "429" in err_msg or "resource_exhausted" in err_msg.lower() or "503" in err_msg or "unavailable" in err_msg.lower()
                if is_transient and attempt < retries - 1:
                    sleep_time = backoff ** (attempt + 1)
                    print(f"Model {model_name} transient error: {err_msg}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Model {model_name} failed: {err_msg}")
                    last_error = e
                    break
        if extracted_markdown is not None:
            break
            
    if extracted_markdown is not None:
        STATS["vision_successes"] += 1
        # Clean up any potential markdown block wrapped responses
        extracted_markdown = extracted_markdown.strip()
        if extracted_markdown.startswith("```markdown"):
            extracted_markdown = extracted_markdown[11:].strip()
        if extracted_markdown.startswith("```"):
            extracted_markdown = extracted_markdown[3:].strip()
        if extracted_markdown.endswith("```"):
            extracted_markdown = extracted_markdown[:-3].strip()
            
        # Write to cache
        cache_data = {
            "relative_filepath": relative_filepath,
            "page_number": page_number,
            "prompt_text": EXTRACTION_PROMPT,
            "extracted_markdown": extracted_markdown,
            "used_model": used_model
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
            
        meta["used_model"] = used_model
        return {
            "extracted_text": extracted_markdown,
            "metadata": meta
        }
    else:
        print(f"Gemini API invocation failed completely. Last error: {last_error}")
        STATS["vision_failures"] += 1
        # Fallback to standard extracted text on failure to keep pipeline moving
        meta["extraction_method"] = "text_fallback"
        return {
            "extracted_text": raw_text.strip(),
            "metadata": meta
        }
