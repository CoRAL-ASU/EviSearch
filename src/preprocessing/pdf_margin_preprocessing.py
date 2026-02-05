# src/preprocessing/preprocess_pdf.py
# Functions for PDF text extraction and cleaning (header/footer removal)
import fitz  # PyMuPDF
import re
from collections import Counter
from ..config.config import PATTERN_SAMPLE_PAGES, TOP_MARGIN, BOTTOM_MARGIN, TOP_THRESHOLD_RATIO, BOTTOM_THRESHOLD_RATIO
from ..chunking.utils_chunking import is_header_or_footer_by_heuristics
from ..utils.logging_utils import setup_logger

logger = setup_logger("preprocessing")


def extract_text_blocks_with_position(page):
    """
    Extract text blocks with their position information.
    Returns list of dicts: [{"text": str, "bbox": (x0, y0, x1, y1), "block_no": int}]
    """
    blocks = page.get_text("dict")["blocks"]
    text_blocks = []
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            
            if block_text.strip():
                text_blocks.append({
                    "text": block_text.strip(),
                    "bbox": block["bbox"],  # (x0, y0, x1, y1)
                    "block_no": block.get("number", 0)
                })
    
    return text_blocks


def detect_repeating_patterns(pdf_path, sample_pages=PATTERN_SAMPLE_PAGES):
    """
    Analyze first few pages to detect repeating headers/footers.
    Returns dict with common patterns found at top/bottom of pages.
    """
    try:
        doc = fitz.open(pdf_path)
        top_texts = []
        bottom_texts = []
        
        num_pages = min(sample_pages, len(doc))
        
        for page_num in range(num_pages):
            page = doc[page_num]
            page_height = page.rect.height
            blocks = extract_text_blocks_with_position(page)
            
            if not blocks:
                continue
            
            # Identify top and bottom regions (top/bottom X% of page)
            top_threshold = page_height * TOP_THRESHOLD_RATIO
            bottom_threshold = page_height * BOTTOM_THRESHOLD_RATIO
            
            for block in blocks:
                y0, y1 = block["bbox"][1], block["bbox"][3]
                text = block["text"]
                
                # Top region
                if y0 < top_threshold:
                    # Normalize text for pattern matching
                    normalized = re.sub(r'\d+', 'NUM', text.lower())
                    top_texts.append(normalized)
                
                # Bottom region
                elif y1 > bottom_threshold:
                    normalized = re.sub(r'\d+', 'NUM', text.lower())
                    bottom_texts.append(normalized)
        
        doc.close()
        
        # Find patterns that repeat across pages
        top_counter = Counter(top_texts)
        bottom_counter = Counter(bottom_texts)
        
        # Patterns that appear on multiple pages are likely headers/footers
        common_top = [pattern for pattern, count in top_counter.items() if count >= 2]
        common_bottom = [pattern for pattern, count in bottom_counter.items() if count >= 2]
        
        return {
            "top_patterns": common_top,
            "bottom_patterns": common_bottom
        }
    
    except Exception as e:
        logger.warning(f"⚠️ Pattern detection failed: {e}")
        return {"top_patterns": [], "bottom_patterns": []}


def is_header_or_footer_by_position(block, page_height, top_margin=TOP_MARGIN, bottom_margin=BOTTOM_MARGIN):
    """Check if text block is in header/footer region based on position."""
    y0, y1 = block["bbox"][1], block["bbox"][3]
    
    # Top margin (header region)
    if y0 < top_margin:
        return True
    
    # Bottom margin (footer region)
    if y1 > page_height - bottom_margin:
        return True
    
    return False


def is_header_or_footer_by_pattern(text, patterns):
    """Check if text matches known header/footer patterns."""
    if not patterns:
        return False
    
    # Normalize text for comparison
    normalized = re.sub(r'\d+', 'NUM', text.lower())
    
    return normalized in patterns["top_patterns"] or normalized in patterns["bottom_patterns"]


def clean_page_text_advanced(page, page_height, patterns):
    """
    Extract and clean text from a page using multiple strategies.
    Returns cleaned text with headers/footers removed.
    """
    blocks = extract_text_blocks_with_position(page)
    
    cleaned_blocks = []
    
    for block in blocks:
        text = block["text"]
        
        # Strategy 1: Position-based filtering
        if is_header_or_footer_by_position(block, page_height, top_margin=TOP_MARGIN, bottom_margin=BOTTOM_MARGIN):
            continue
        
        # Strategy 2: Pattern-based filtering (learned from PDF)
        if is_header_or_footer_by_pattern(text, patterns):
            continue
        
        # Strategy 3: Heuristic-based filtering
        if is_header_or_footer_by_heuristics(text):
            continue
        
        # This block passed all filters - keep it
        cleaned_blocks.append(text)
    
    return "\n".join(cleaned_blocks)

