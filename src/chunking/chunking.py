# src/chunking/chunking.py
# Main chunking logic for PDF processing
import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from PIL import Image
import pdfplumber
from io import BytesIO
from ..config.config import PIXMAP_RESOLUTION
from ..preprocessing.pdf_margin_preprocessing import (
    detect_repeating_patterns,
    clean_page_text_advanced
)
from ..chunking.utils_chunking import (
    text_chunking,
    extract_tables_pdfplumber,
    analyze_image_with_llm,
    extract_caption_from_gemini,
    parse_table_extraction_response,
    save_chunks_to_json
)
from ..utils.logging_utils import setup_logger

logger = setup_logger("chunking")


class PDFChunker:
    """
    Class for chunking PDF content into text, tables, and figures.
    Orchestrates preprocessing, extraction, and chunk generation.
    Supports targeted processing based on page metadata.
    """
    
    def __init__(self, pdf_path, page_metadata=None):
        self.pdf_path = pdf_path
        self.chunks = []
        self.accumulated_text = []  # Accumulate all text from pages
        
        # Build lookup sets for fast page checking (targeted processing)
        if page_metadata:
            self.table_pages = set(t["page"] for t in page_metadata.get("tables", []))
            self.figure_pages = set(f["page"] for f in page_metadata.get("figures", []))
            logger.info(f"🎯 Targeted mode: {len(self.table_pages)} table pages, {len(self.figure_pages)} figure pages")
        else:
            # If no metadata, process all pages (backwards compatible)
            self.table_pages = None
            self.figure_pages = None
            logger.info("📖 Standard mode: processing all pages")
        # Token usage for costing (image LLM calls only; page classification tracked separately)
        self.usage_input_tokens = 0
        self.usage_output_tokens = 0
    
    def _process_page_text(self, page, page_num, patterns):
        """Process text content for a single page and accumulate it."""
        page_height = page.rect.height
        raw_text = clean_page_text_advanced(page, page_height, patterns)
        
        # Stop at References
        ref_match = re.search(r'(?i)\b(references|bibliography)\b', raw_text)
        if ref_match:
            raw_text = raw_text[:ref_match.start()].strip()
        
        # Accumulate text from this page (will be chunked later)
        if raw_text.strip():
            self.accumulated_text.append({
                "text": raw_text,
                "page": page_num + 1
            })
    
    def _process_tables(self, page, page_num, pdf_path):
        """Extract and process tables for a single page using LLM with retry logic."""
        try:
            # Use page screenshot for LLM analysis
            pix = page.get_pixmap(matrix=fitz.Matrix(PIXMAP_RESOLUTION, PIXMAP_RESOLUTION))
            img_bytes = pix.tobytes("png")
            img_pil = Image.open(BytesIO(img_bytes))
            
            # Load table extraction prompt
            from pathlib import Path
            prompt_path = Path(__file__).parent / "table_extraction.txt"
            if not prompt_path.exists():
                logger.warning(f"[Table] Prompt file not found: {prompt_path}, using default")
                prompt_text = None
            else:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
            
            # Retry logic: try up to 3 times to get valid table extraction
            max_retries = 3
            markdown_table = None
            caption = "Table"
            
            for attempt in range(1, max_retries + 1):
                # Call LLM to extract table and caption
                llm_response, in_tok, out_tok = analyze_image_with_llm(img_pil, prompt_text=prompt_text)
                self.usage_input_tokens += in_tok
                self.usage_output_tokens += out_tok

                if not llm_response:
                    logger.warning(f"[Table] Page {page_num + 1}, attempt {attempt}/{max_retries}: No LLM response")
                    if attempt < max_retries:
                        continue
                    else:
                        break
                
                # Parse LLM response to extract markdown table and caption
                parsed = parse_table_extraction_response(llm_response)
                markdown_table = parsed.get("markdown_table")
                caption = parsed.get("caption") or "Table"
                
                if markdown_table:
                    logger.info(f"[Table] Page {page_num + 1}: Successfully extracted table on attempt {attempt}")
                    break
                else:
                    logger.warning(f"[Table] Page {page_num + 1}, attempt {attempt}/{max_retries}: Could not extract table from LLM response")
                    if attempt < max_retries:
                        logger.info(f"[Table] Page {page_num + 1}: Retrying...")
            
            # If all retries failed, fallback to pdfplumber
            if not markdown_table:
                logger.warning(f"[Table] Page {page_num + 1}: All {max_retries} LLM attempts failed, trying pdfplumber fallback")
                try:
                    with pdfplumber.open(pdf_path) as plumber:
                        pl_page = plumber.pages[page_num]
                        md_tables = extract_tables_pdfplumber(pl_page)
                        if md_tables:
                            markdown_table = md_tables[0]
                            logger.info(f"[Table] Page {page_num + 1}: Using pdfplumber fallback")
                        else:
                            logger.warning(f"[Table] Page {page_num + 1}: pdfplumber found no tables")
                            return
                except Exception as e:
                    logger.warning(f"[Table] Page {page_num + 1}: pdfplumber fallback failed: {e}")
                    return
            
            # Create table chunk with LLM-generated content (or pdfplumber fallback)
            self.chunks.append({
                "type": "table",
                "content": caption,
                "page": page_num + 1,
                "length": len(img_bytes),
                "source": "image",
                "table_content": f"##Markdown Table##\n\n{markdown_table}\n\n##Caption##\n\n{caption}"
            })
            
        except Exception as e:
            logger.warning(f"[Table] Page {page_num + 1} failed: {e}")
    
    def _process_figures(self, raw_text, page, page_num):
        """Extract and process figures if detected on the page."""
        if re.search(r"\bfig(?:ure)?s?[ .:-]*\d+", raw_text, re.IGNORECASE):
            pix = page.get_pixmap(matrix=fitz.Matrix(PIXMAP_RESOLUTION, PIXMAP_RESOLUTION))
            img_bytes = pix.tobytes("png")
            img_pil = Image.open(BytesIO(img_bytes))
            gemini_raw, in_tok, out_tok = analyze_image_with_llm(img_pil)
            self.usage_input_tokens += in_tok
            self.usage_output_tokens += out_tok
            description = gemini_raw.strip() if gemini_raw else "Figure"
            
            block = f"```\n##Figure Descriptions##\n\n{description}\n```"
            self.chunks.append({
                "type": "figure",
                "content": block,
                "page": page_num + 1,
                "length": len(img_bytes),
                "source": "image",
                "figure_content": block
            })
    
    
    def _create_large_text_chunks(self):
        """Create 4-5 large text chunks from all accumulated text."""
        if not self.accumulated_text:
            return
        
        # Create page-delimited text segments with markers
        page_texts = []
        for item in self.accumulated_text:
            page_texts.append({
                "text": item["text"],
                "page": item["page"]
            })
        
        # Combine all text from all pages
        all_text = "\n\n".join([item["text"] for item in self.accumulated_text])
        
        # Get the overall page range for logging
        page_numbers = [item["page"] for item in self.accumulated_text]
        min_page = min(page_numbers) if page_numbers else 1
        max_page = max(page_numbers) if page_numbers else 1
        
        # Generate large text chunks (4-5 chunks total)
        text_chunks = text_chunking(all_text)
        
        logger.info(f"📝 Created {len(text_chunks)} large text chunks from {len(self.accumulated_text)} pages")
        
        # Estimate page ranges for each chunk based on character distribution
        total_chars = len(all_text)
        cumulative_chars = 0
        chunk_page_ranges = []
        
        for txt in text_chunks:
            chunk_len = len(txt)
            # Calculate which pages this chunk approximately covers
            start_ratio = cumulative_chars / total_chars if total_chars > 0 else 0
            end_ratio = (cumulative_chars + chunk_len) / total_chars if total_chars > 0 else 1
            
            # Map ratios to page numbers
            total_pages = max_page - min_page + 1
            start_page = min_page + int(start_ratio * total_pages)
            end_page = min_page + int(end_ratio * total_pages)
            
            # Ensure at least one page and valid range
            start_page = max(min_page, start_page)
            end_page = min(max_page, max(start_page, end_page))
            
            chunk_page_ranges.append((start_page, end_page))
            cumulative_chars += chunk_len
        
        # Add text chunks to the chunks list with estimated page ranges
        for idx, (txt, (start_page, end_page)) in enumerate(zip(text_chunks, chunk_page_ranges), 1):
            page_str = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
            self.chunks.append({
                "type": "text",
                "content": txt,
                "page": page_str,
                "chunk_number": idx,
                "length": len(txt)
            })
    
    def chunk(self):
        """Main method to process the entire PDF and generate chunks."""
        try:
            # Step 0: Learn header/footer patterns
            logger.info("🔍 Analyzing PDF for header/footer patterns...")
            patterns = detect_repeating_patterns(self.pdf_path)
            logger.info(f"   Found {len(patterns['top_patterns'])} top patterns and {len(patterns['bottom_patterns'])} bottom patterns")
            
            doc = fitz.open(self.pdf_path)
            stop_processing = False

            for page_num in range(len(doc)):
                if stop_processing:
                    break

                page = doc[page_num]
                page_number = page_num + 1
                
                # ALWAYS: Accumulate text (will be chunked later)
                raw_text = clean_page_text_advanced(page, page.rect.height, patterns)
                self._process_page_text(page, page_num, patterns)
                
                # CONDITIONAL: Process tables (only if on table page OR no metadata)
                if self.table_pages is None or page_number in self.table_pages:
                    self._process_tables(page, page_num, self.pdf_path)
                
                # CONDITIONAL: Process figures (only if on figure page OR no metadata)
                if self.figure_pages is None or page_number in self.figure_pages:
                    self._process_figures(raw_text, page, page_num)

            doc.close()
            
            # After processing all pages, create large text chunks (4-5 per PDF)
            self._create_large_text_chunks()
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
        
        return self.chunks


def process_pdf(pdf_path, output_path="pdf_chunks.json", use_llm_classification=True):
    """
    Entry point function to process PDF with optional LLM-based page classification.

    Args:
        pdf_path: Path to PDF file
        output_path: Path to save chunks JSON (e.g., "output_dir/pdf_chunked.json")
        use_llm_classification: If True, use Gemini to classify pages first (default: True)

    Returns:
        tuple: (chunks list, usage_dict for costing)
        usage_dict has keys: input_tokens, output_tokens, cost_usd, provider, model, breakdown (list of per-sub-call usage)
    """
    from ..config.config import CHUNKING_PROVIDER, CHUNKING_MODEL
    from ..utils.costing import usage_to_cost_dict
    from ..LLMProvider.google_genai_client import vertex_auth_error_message

    output_dir = Path(output_path).parent
    page_metadata = None
    classification_usage = None

    if use_llm_classification:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 STEP 1: Page Classification with LLM")
        logger.info("=" * 80 + "\n")
        
        try:
            from .page_classifier import PageClassifier
            from ..config.config import (
                STRUCTURER_MODEL, 
                STRUCTURER_BASE_URL
            )

            classifier = PageClassifier(
                pdf_path=pdf_path,
                output_dir=output_dir,
                structurer_model=STRUCTURER_MODEL,
                structurer_base_url=STRUCTURER_BASE_URL
            )
            page_metadata, classification_usage = classifier.classify()

            # Log summary
            table_pages = sorted(set(t["page"] for t in page_metadata["tables"]))
            figure_pages = sorted(set(f["page"] for f in page_metadata["figures"]))
            
            logger.info(f"\n✅ Classification complete:")
            logger.info(f"   Tables:  {len(page_metadata['tables'])} found on {len(table_pages)} pages: {table_pages}")
            logger.info(f"   Figures: {len(page_metadata['figures'])} found on {len(figure_pages)} pages: {figure_pages}\n")
        
        except Exception as e:
            logger.error(f"\n❌ Page classification failed: {e}")
            logger.error(f"   Vertex AI setup hint: {vertex_auth_error_message()}")
            logger.error("   Please check logs and retry manually\n")
            raise  # Fail fast
    
    logger.info("=" * 80)
    logger.info("🔄 STEP 2: Targeted PDF Chunking")
    logger.info("=" * 80 + "\n")
    
    chunker = PDFChunker(pdf_path, page_metadata=page_metadata)
    chunks = chunker.chunk()
    
    logger.info(f"\n✅ Extracted {len(chunks)} total chunks")
    
    # Count chunk types
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk.get("type", "unknown")
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    logger.info("   Chunk breakdown:")
    for ctype, count in sorted(chunk_types.items()):
        logger.info(f"   - {ctype}: {count}")
    
    save_chunks_to_json(chunks, output_path)
    logger.info(f"\n💾 Saved chunks to: {output_path}")

    if chunks:
        logger.info("\n📄 Sample Chunk:")
        logger.info(json.dumps(chunks[0], indent=4, ensure_ascii=False)[:500] + "...")

    # Build usage for costing: breakdown (page classification + image LLM) and totals
    breakdown = []
    if classification_usage:
        d = usage_to_cost_dict(
            classification_usage["provider"],
            classification_usage["model"],
            classification_usage["input_tokens"],
            classification_usage["output_tokens"],
        )
        breakdown.append(d)
    if chunker.usage_input_tokens or chunker.usage_output_tokens:
        d = usage_to_cost_dict(
            CHUNKING_PROVIDER,
            CHUNKING_MODEL,
            chunker.usage_input_tokens,
            chunker.usage_output_tokens,
        )
        breakdown.append(d)
    total_in = sum(b["input_tokens"] for b in breakdown)
    total_out = sum(b["output_tokens"] for b in breakdown)
    total_cost = sum(b["cost_usd"] for b in breakdown)
    usage_dict = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(total_cost, 6),
        "provider": breakdown[0]["provider"] if len(breakdown) == 1 else "mixed",
        "model": breakdown[0]["model"] if len(breakdown) == 1 else "mixed",
        "breakdown": breakdown,
    }
    return chunks, usage_dict
