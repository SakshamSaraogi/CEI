import re
from pathlib import Path

def is_subcategory_header(row: str) -> bool:
    """
    Detects if a row is a sub-category header in a markdown table.
    e.g. | **Engine** | | | | or | **Dimensions** | | | |
    """
    parts = [p.strip() for p in row.split("|")]
    if len(parts) >= 3:
        col1 = parts[1]
        # Check if first column has bold marker and the rest of the cells are empty or dashes
        is_bold = col1.startswith("**") and col1.endswith("**")
        rest_empty = all(p == "" or p == "-" or p == "None" for p in parts[2:-1])
        if is_bold or (col1.isupper() and len(col1) < 25 and rest_empty):
            return True
    return False

def chunk_markdown_table(table_text: str) -> list[tuple[str, str]]:
    """
    Chunks a markdown table by its logical bold sub-sections.
    Each chunk retains the core table headers (first 2 rows) to keep variant column context.
    Returns a list of tuples: (chunk_text, category_name)
    """
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    
    # Extract table header lines (typically the first two lines containing '|')
    header_lines = []
    data_lines = []
    
    for line in lines:
        if line.startswith("|"):
            if len(header_lines) < 2:
                header_lines.append(line)
            else:
                data_lines.append(line)
        else:
            # Keep non-table lines (disclaimers/footers) in a separate place or attach to the last chunk
            data_lines.append(line)
            
    # If we couldn't parse a proper markdown table, fall back
    if len(header_lines) < 2:
        # Return fallback with category=None
        paragraphs = [p.strip() for p in table_text.split("\n\n") if p.strip()]
        return [(p, None) for p in paragraphs]
        
    sections = []
    current_section = None
    
    for line in data_lines:
        if line.startswith("|") and is_subcategory_header(line):
            # Start a new sub-section
            parts = [p.strip() for p in line.split("|")]
            # Clean bold text: "**Engine**" -> "engine"
            category_raw = parts[1].replace("**", "").strip().lower()
            
            current_section = {
                "header": line,
                "category": category_raw,
                "rows": []
            }
            sections.append(current_section)
        elif line.startswith("|"):
            # It's a standard data row
            if current_section is None:
                # Default section if rows appear before any sub-category header
                current_section = {
                    "header": None,
                    "category": None,
                    "rows": []
                }
                sections.append(current_section)
            current_section["rows"].append(line)
        else:
            # Non-table line (e.g. footnotes, disclaimers)
            if current_section is None:
                current_section = {
                    "header": None,
                    "category": None,
                    "rows": []
                }
                sections.append(current_section)
            current_section["rows"].append(line)
            
    # Assemble chunks
    chunks = []
    header_part = "\n".join(header_lines)
    
    for sec in sections:
        chunk_lines = [header_part]
        if sec["header"]:
            chunk_lines.append(sec["header"])
        if sec["rows"]:
            chunk_lines.extend(sec["rows"])
        chunk_text = "\n".join(chunk_lines)
        chunks.append((chunk_text, sec["category"]))
        
    # If no sections were built, return the original text
    if not chunks:
        return [(table_text, None)]
        
    return chunks

def chunk_narrative_text(text: str, target_size: int = 800) -> list[str]:
    """
    Chunks narrative text by logical paragraphs, avoiding splitting mid-paragraph.
    Groups multiple short paragraphs up to target_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        p_len = len(p)
        # If adding this paragraph would exceed target_size, yield the current chunk first
        if current_chunk and (current_length + p_len + 2 > target_size):
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_length = p_len
        else:
            current_chunk.append(p)
            current_length += p_len + (2 if current_length > 0 else 0)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def create_chunks(extracted_data: dict) -> list[dict]:
    """
    Takes an extraction result dict from extractor.py and returns a list of chunk dicts
    fully tagged with metadata.
    """
    text = extracted_data["extracted_text"]
    metadata = extracted_data["metadata"]
    
    section_type = metadata["section_type"]
    model_name = metadata.get("model", "unknown")
    page_num = metadata.get("page_number", 0)
    
    chunks_output = []
    
    # Perform chunking based on page type
    if section_type in ("specification", "spec/table", "scanned"):
        raw_chunks = chunk_markdown_table(text)
        for i, (raw_chunk, category) in enumerate(raw_chunks):
            chunk_id = f"{model_name}_p{page_num}_c{i+1}"
            chunk_meta = metadata.copy()
            chunk_meta["chunk_id"] = chunk_id
            
            # Map section_type specifically
            if category:
                chunk_meta["section_type"] = category
            elif section_type == "scanned":
                chunk_meta["section_type"] = "scanned"
            else:
                chunk_meta["section_type"] = "specification"  # fallback label
                
            chunks_output.append({
                "chunk_text": raw_chunk,
                "metadata": chunk_meta
            })
    else:
        raw_chunks = chunk_narrative_text(text)
        for i, raw_chunk in enumerate(raw_chunks):
            chunk_id = f"{model_name}_p{page_num}_c{i+1}"
            chunk_meta = metadata.copy()
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["section_type"] = "narrative"
            
            chunks_output.append({
                "chunk_text": raw_chunk,
                "metadata": chunk_meta
            })
        
    return chunks_output
