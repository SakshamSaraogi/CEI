import os
from pathlib import Path
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BROCHURES_DIR = PROJECT_ROOT / "data" / "brochures"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

def main():
    print("Starting cover extraction...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    # Iterate through brands
    for brand_dir in BROCHURES_DIR.iterdir():
        if not brand_dir.is_dir():
            continue
        
        brand_name = brand_dir.name
        brand_images_dir = IMAGES_DIR / brand_name
        brand_images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing brand: {brand_name}")
        
        # Iterate through brochure PDFs
        for pdf_path in brand_dir.glob("*.pdf"):
            model_name = pdf_path.stem
            output_path = brand_images_dir / f"{model_name}.png"
            
            try:
                # Open PDF and render first page (page 0)
                doc = fitz.open(pdf_path)
                if len(doc) > 0:
                    page = doc[0]
                    zoom = 150 / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    pix.save(str(output_path))
                    extracted_count += 1
                    print(f"  Extracted cover for {brand_name}/{model_name}")
                doc.close()
            except Exception as e:
                print(f"  [ERROR] Failed to extract cover for {pdf_path}: {e}")
                
    print(f"\nExtraction completed. Saved {extracted_count} cover images to {IMAGES_DIR}")

if __name__ == "__main__":
    main()
