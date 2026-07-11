import sys
from pathlib import Path
from extractor import extract_page_data

# Ensure stdout is in UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CRETA_PDF = str(PROJECT_ROOT / "data" / "brochures" / "hyundai" / "creta.pdf")

def test_extraction():
    print("=============================================================")
    print("VERIFICATION TASK (a): NARRATIVE PAGE EXTRACTION")
    print("=============================================================")
    # Page 2 of Creta should be classified as NARRATIVE
    narrative_result = extract_page_data(CRETA_PDF, 2)
    print(f"Classification / Section Type: {narrative_result['metadata']['section_type']}")
    print(f"Extraction Method: {narrative_result['metadata']['extraction_method']}")
    print(f"Metadata: {narrative_result['metadata']}")
    print("--- Extracted Text ---")
    print(narrative_result['extracted_text'])
    print("=============================================================\n")

    print("=============================================================")
    print("VERIFICATION TASK (b): SPEC/TABLE PAGE EXTRACTION (VISION)")
    print("=============================================================")
    # Page 17 of Creta is a spec page
    spec_result = extract_page_data(CRETA_PDF, 17)
    print(f"Classification / Section Type: {spec_result['metadata']['section_type']}")
    print(f"Extraction Method: {spec_result['metadata']['extraction_method']}")
    print(f"Metadata: {spec_result['metadata']}")
    print("--- Extracted Markdown Table ---")
    print(spec_result['extracted_text'])
    print("=============================================================")

if __name__ == "__main__":
    test_extraction()
