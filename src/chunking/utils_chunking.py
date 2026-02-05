# src/chunking/utils_chunking.py
# Utility functions for text processing, embeddings, and external API calls
import re
import json
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pdfplumber
from collections import Counter
import fitz 

# Import config
from ..config.config import (
    EMBEDDING_MODEL_NAME,
    TEXT_CHUNK_MIN_SIZE,
    HEURISTIC_MAX_LENGTH,
    CHUNKING_PROVIDER,
    CHUNKING_MODEL,
    TEXT_CHUNK_OVERLAP,
    CHUNKING_MODE
)
from ..utils.logging_utils import setup_logger
from ..LLMProvider import LLMProvider

# Load models (shared across the codebase)
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

logger = setup_logger("utils")

# Lazy-loaded chunking provider
_chunking_provider = None

def _get_chunking_provider():
    """Get or initialize the chunking LLM provider."""
    global _chunking_provider
    if _chunking_provider is None:
        _chunking_provider = LLMProvider(provider=CHUNKING_PROVIDER, model=CHUNKING_MODEL)
        logger.info(f"Initialized chunking provider: {CHUNKING_PROVIDER}/{CHUNKING_MODEL}")
    return _chunking_provider


def looks_like_inline_table(text):
    """Check if text resembles an inline table based on digit and pattern density."""
    lines = text.split("\n")
    if len(lines) < 3:
        return False
    digit_lines = sum(1 for line in lines if re.search(r'\d', line) and re.search(r'\(.+\)', line))
    return digit_lines / len(lines) > 0.5


def is_table_caption_or_footnote(text):
    """Check if text is a table caption or footnote based on regex keywords."""
    return bool(
        re.search(r'^\s*(Table|Fig|Figure)\s+\d+', text, re.IGNORECASE)
        or 'TD$FIG' in text
        or re.search(r'\b[A-Z]{2,}\s*=', text)
    )


def text_chunking(text, max_size=TEXT_CHUNK_MIN_SIZE, overlap=0, mode=None):
    """
    Paragraph-based chunking that creates large chunks (4-5 per document).
    The text is collected and then divided into equal-sized large chunks.
    Args:
        text: Input text to chunk
        max_size: Maximum size of each chunk in characters (used as target chunk size)
        overlap: Deprecated parameter (kept for backward compatibility, but not used)
        mode: 'paragraph' (default), 'sentence' (legacy)
    Returns:
        List of text chunks (typically 4-5 chunks with large content)
    """
    if mode is None:
        mode = CHUNKING_MODE if 'CHUNKING_MODE' in globals() else 'paragraph'

    if mode == 'sentence':
        # Legacy sentence-based chunking
        doc = nlp(text)
        chunks = []
        current_chunk = ""
        for sent in doc.sents:
            sentence = sent.text.strip()
            if looks_like_inline_table(sentence) or is_table_caption_or_footnote(sentence):
                continue
            if is_header_or_footer_by_heuristics(sentence):
                continue
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    # New: Large chunk creation (4-5 chunks per document)
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    filtered_paragraphs = []
    for para in paragraphs:
        # Remove paragraphs that are tables, captions, or headers/footers
        if looks_like_inline_table(para) or is_table_caption_or_footnote(para):
            continue
        if is_header_or_footer_by_heuristics(para):
            continue
        filtered_paragraphs.append(para)

    # Combine all filtered paragraphs into one text
    all_text = "\n\n".join(filtered_paragraphs)
    
    if not all_text.strip():
        return []
    
    # Calculate target number of chunks (4-5 per document)
    # Use a large target chunk size to ensure we get only 4-5 chunks
    total_length = len(all_text)
    target_num_chunks = 5  # Target 5 chunks per document
    target_chunk_size = max(max_size * 2, total_length // target_num_chunks)  # At least double the max_size
    
    # Create chunks by distributing paragraphs
    chunks = []
    current_chunk = ""
    
    for para in filtered_paragraphs:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 2 <= target_chunk_size:
            current_chunk += "\n\n" + para
        else:
            # Only create a new chunk if we haven't reached target number of chunks
            # or if current chunk is already very large
            if len(chunks) < target_num_chunks or len(current_chunk) > target_chunk_size * 1.5:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Keep adding to current chunk to maintain 4-5 chunks total
                current_chunk += "\n\n" + para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_caption_from_gemini(text: str) -> str:
    """Extract caption from LLM output."""
    if not text:
        return "Table"

    match = re.search(r"##\s*(Table|Figure)\s*\d*", text, re.IGNORECASE)
    if match:
        start = match.end()
        rest = text[start:].lstrip()
        sentence = rest.split("\n")[0].strip()
        if sentence.endswith(('.', '!', '?')):
            return sentence
        else:
            return sentence + "."

    first_sentence = re.split(r'[.!?]\s*', text)[0].strip()
    return first_sentence + "." if first_sentence else "Table"


def parse_table_extraction_response(llm_response: str) -> dict:
    """
    Parse LLM response from table extraction prompt.
    
    Expected format:
    ##Markdown Table##
    
    [Markdown table content here]
    
    
    ##Caption##
    
    [Extracted and enriched caption here]
    
    Args:
        llm_response: Raw LLM response text
    
    Returns:
        dict with keys: 'markdown_table' and 'caption'
        Returns None values if parsing fails
    """
    if not llm_response:
        return {"markdown_table": None, "caption": None}
    
    result = {"markdown_table": None, "caption": None}
    
    # Try to extract markdown table section
    table_match = re.search(r"##\s*Markdown\s+Table\s*##\s*\n(.*?)(?=\n\s*##|$)", llm_response, re.DOTALL | re.IGNORECASE)
    if table_match:
        table_content = table_match.group(1).strip()
        # Remove any trailing markdown table markers that might be in the content
        table_content = re.sub(r"\s*##\s*Markdown\s+Table\s*##\s*$", "", table_content, flags=re.IGNORECASE)
        if table_content:
            result["markdown_table"] = table_content
    
    # Try to extract caption section
    caption_match = re.search(r"##\s*Caption\s*##\s*\n(.*?)(?=\n\s*##|$)", llm_response, re.DOTALL | re.IGNORECASE)
    if caption_match:
        caption_content = caption_match.group(1).strip()
        if caption_content:
            result["caption"] = caption_content
    
    # Fallback: if format doesn't match, try to extract caption using old method
    if not result["caption"]:
        result["caption"] = extract_caption_from_gemini(llm_response)
    
    return result


def extract_tables_pdfplumber(page) -> list:
    """Extract tables from a pdfplumber page and return as markdown strings."""
    try:
        tables = page.extract_tables()
        md_tables = []
        for tbl in tables:
            if not tbl or len(tbl) < 2:
                continue
            df = pd.DataFrame(tbl[1:], columns=tbl[0])
            md = df.to_markdown(index=False)
            md_tables.append(md.strip())
        return md_tables
    except Exception as e:
        logger.warning(f"[pdfplumber] Error: {e}")
        return []


def extract_images_fitz(page, page_num) -> list:
    """Extract embedded images from a PDF page using PyMuPDF."""
    
    img_chunks = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base = page.parent.extract_image(xref)
            b64 = base64.b64encode(base["image"]).decode()
            img_chunks.append({
                "type": "image",
                "content": f"Image of size {len(b64)} characters (Base64)",
                "page": page_num,
                "length": len(b64),
                "source": "image",
                "image_base64": b64
            })
        except Exception as e:
            logger.warning(f"[Image] Failed XREF {xref}: {e}")
    return img_chunks


def analyze_image_with_llm(image_pil, prompt_text=None):
    """
    Analyze image using configured LLM provider (multimodal).
    
    Uses CHUNKING_PROVIDER and CHUNKING_MODEL from config.
    Supports: Gemini (default), GPT-4V
    
    Args:
        image_pil: PIL Image object
        prompt_text: Optional custom prompt
    
    Returns:
        tuple: (text_response, input_tokens, output_tokens)
    """
    if not prompt_text:
        prompt_text = (
            "Analyze this image from a research paper.\n"
            "If it's a table, return:\n"
            "## Table X\n"
            "Caption sentence.\n"
            "| Header | ...\n\n"
            "If it's a figure, return:\n"
            "## Figure X\n"
            "Description in 2-3 sentences.\n"
            "Return ONLY the content, no JSON."
        )
    
    provider = _get_chunking_provider()
    response = provider.generate_with_image(prompt_text, image_pil)
    
    if response.success:
        return response.text, response.input_tokens, response.output_tokens
    else:
        logger.warning(f"[LLM] Failed: {response.error}")
        return None, 0, 0


# Keep old function name as alias for backward compatibility
def ask_gemini_with_image(image_pil, prompt_text=None):
    """
    Legacy function name. Use analyze_image_with_llm instead.
    
    Returns:
        str: Text response (for backward compatibility, does not return tokens)
    """
    text, _, _ = analyze_image_with_llm(image_pil, prompt_text)
    return text


def save_chunks_to_json(chunks, output_path):
    """Save chunks to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    logger.info(f"✅ Chunks saved to {output_path}")


def is_header_or_footer_by_heuristics(text):
    """
    Use general heuristics to detect headers/footers.
    This catches common patterns across different journals.
    """
    text_lower = text.lower().strip()
    
    # Common characteristics of headers/footers:
    # 1. Very short (< HEURISTIC_MAX_LENGTH chars)
    if len(text.strip()) > HEURISTIC_MAX_LENGTH:
        return False
    
    # 2. Contains common header/footer keywords
    header_footer_keywords = [
        'copyright', 'downloaded from', 'all rights reserved',
        'massachusetts medical society', 'nejm.org', 'doi:',
        'page', 'vol', 'volume', 'issue', 'published',
        'elsevier', 'wiley', 'springer', 'nature',
        'journal of', 'american', 'society',
        'training, and similar technologies'  # Common copyright clause
    ]
    
    if any(keyword in text_lower for keyword in header_footer_keywords):
        return True
    
    # 3. Pattern: Journal abbreviation + volume/issue + page numbers
    # e.g., "n engl j med 377;4  nejm.org  July 27, 2017"
    if re.search(r'[a-z\s]+\d+[;:]\d+', text_lower):
        return True
    
    # 4. Mostly numbers, dates, or page markers
    # e.g., "339", "July 27, 2017"
    if re.search(r'^\s*\d+\s*$', text):  # Standalone numbers
        return True
    
    # 5. Date patterns
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+,\s+\d{4}\b', text_lower):
        return True
    
    # 6. URL patterns
    if re.search(r'\b[a-z]+\.(org|com|edu|gov)\b', text_lower):
        return True
    
    # 7. Copyright symbols and years
    if re.search(r'©\s*\d{4}|copyright.*\d{4}', text_lower):
        return True
    
    return False
