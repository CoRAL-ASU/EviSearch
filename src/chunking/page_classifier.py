# src/chunking/page_classifier.py
"""
Page classifier for identifying table and figure pages in PDFs using LLM.
Uses Gemini for initial classification and local LLM for structuring.
"""
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

from ..LLMProvider.google_genai_client import create_vertex_genai_client, get_genai_types
from ..LLMProvider.structurer import OutputStructurer
from ..utils.logging_utils import setup_logger

logger = setup_logger("page_classifier")


# Pydantic schemas
class TableInfo(BaseModel):
    """Information about a table in the PDF."""
    page: int = Field(..., description="Page number where table is located")
    name: str = Field(..., description="Table name/number (e.g., 'Table 1')")
    description: str = Field(default="", description="Brief caption or description of table content")


class FigureInfo(BaseModel):
    """Information about a figure in the PDF."""
    page: int = Field(..., description="Page number where figure is located")
    name: str = Field(..., description="Figure name/number (e.g., 'Figure 1')")
    description: str = Field(default="", description="Brief description (e.g., 'survival curve', 'forest plot')")


class TablesResponse(BaseModel):
    """Structured response for tables classification."""
    tables: List[TableInfo] = Field(default_factory=list, description="List of tables found in PDF")


class FiguresResponse(BaseModel):
    """Structured response for figures classification."""
    figures: List[FigureInfo] = Field(default_factory=list, description="List of figures found in PDF")


class PageClassifier:
    """
    Classifies PDF pages to identify which contain tables and figures.
    Uses Gemini for classification and local LLM for response structuring.
    """
    
    def __init__(self, pdf_path: str, output_dir: str,
                 structurer_model: str = "Qwen/Qwen3-8B",
                 structurer_base_url: str = "http://localhost:8001/v1"):
        """
        Initialize page classifier.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save metadata artifacts
            Uses Vertex AI Gemini auth from environment
            structurer_model: Local model for structuring responses
            structurer_base_url: Base URL for local LLM API
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir / "chunking_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = create_vertex_genai_client()
        self.types = get_genai_types()
        self.structurer = OutputStructurer(
            base_url=structurer_base_url,
            model=structurer_model,
            enable_thinking=False  # Disable thinking for cleaner JSON output
        )
    
    def classify(self) -> dict:
        """
        Main classification method.
        
        Returns:
            dict: {
                "tables": [{"page": 5, "name": "Table 1", "description": "..."}, ...],
                "figures": [{"page": 3, "name": "Figure 1", "description": "..."}, ...]
            }
        
        Raises:
            Exception: If classification fails (fail fast strategy)
        """
        try:
            logger.info(f"📄 Loading PDF: {Path(self.pdf_path).name}")
            
            # Upload PDF to Gemini
            pdf_bytes = Path(self.pdf_path).read_bytes()
            pdf_part = self.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            logger.info(f"   Size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
            
            # Classify tables
            logger.info("📋 Classifying table pages...")
            tables_raw, tables_structured, tables_usage = self._classify_tables(pdf_part)
            logger.info(f"   Found {len(tables_structured)} tables")

            # Classify figures
            logger.info("📊 Classifying figure pages...")
            figures_raw, figures_structured, figures_usage = self._classify_figures(pdf_part)
            logger.info(f"   Found {len(figures_structured)} figures")

            # Save all artifacts
            self._save_artifacts(tables_raw, tables_structured, figures_raw, figures_structured)

            # Usage for costing (Gemini gemini-2.5-flash)
            class_usage = {
                "input_tokens": tables_usage[0] + figures_usage[0],
                "output_tokens": tables_usage[1] + figures_usage[1],
                "provider": "gemini",
                "model": "gemini-2.5-flash",
            }

            # Return combined metadata and usage
            return {
                "tables": tables_structured,
                "figures": figures_structured
            }, class_usage
        
        except Exception as e:
            logger.error(f"❌ Page classification failed: {e}")
            raise  # Fail fast

    def _classify_tables(self, pdf_part):
        """
        Classify table pages using Gemini and structure response.
        
        Returns:
            tuple: (raw_text, structured_list)
        """
        prompt = """Analyze this clinical trial PDF and identify ALL pages that contain data tables.

For each table you find, provide:
- Page number (exact integer)
- Table name or number (e.g., "Table 1", "Table 2")
- Brief caption or description of what data the table contains

List them clearly, one per line, in this format:
Page X: Table Y - [brief description]

If no tables are found, respond with "No tables found".

Be thorough - check all pages."""
        
        config = self.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2000
        )
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pdf_part, prompt],
            config=config
        )
        usage = getattr(response, "usage_metadata", None)
        tables_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        tables_out = getattr(usage, "candidates_token_count", 0) if usage else 0

        raw_text = response.text.strip()
        logger.info(f"   Raw response length: {len(raw_text)} chars")

        # Structure with local LLM
        structured_result = self.structurer.structure(
            text=raw_text,
            schema=TablesResponse,
            max_retries=3,
            return_dict=True
        )
        
        if not structured_result.success:
            raise ValueError(f"Failed to structure tables response: {structured_result.error}")

        return raw_text, structured_result.data.get("tables", []), (tables_in, tables_out)

    def _classify_figures(self, pdf_part):
        """
        Classify figure pages using Gemini and structure response.
        
        Returns:
            tuple: (raw_text, structured_list)
        """
        prompt = """Now, identify ALL pages that contain figures, graphs, or diagrams.

For each figure you find, provide:
- Page number (exact integer)
- Figure name or number (e.g., "Figure 1", "Figure 2")
- Brief description of the figure type (e.g., "Kaplan-Meier survival curve", "Forest plot", "CONSORT diagram")

List them clearly, one per line, in this format:
Page X: Figure Y - [brief description]

If no figures are found, respond with "No figures found".

Be thorough - check all pages."""
        
        config = self.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2000
        )
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pdf_part, prompt],
            config=config
        )
        usage = getattr(response, "usage_metadata", None)
        figures_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        figures_out = getattr(usage, "candidates_token_count", 0) if usage else 0

        raw_text = response.text.strip()
        logger.info(f"   Raw response length: {len(raw_text)} chars")

        # Structure with local LLM
        structured_result = self.structurer.structure(
            text=raw_text,
            schema=FiguresResponse,
            max_retries=3,
            return_dict=True
        )

        if not structured_result.success:
            raise ValueError(f"Failed to structure figures response: {structured_result.error}")

        return raw_text, structured_result.data.get("figures", []), (figures_in, figures_out)
    
    def _save_artifacts(self, tables_raw, tables_structured, figures_raw, figures_structured):
        """Save all classification artifacts to chunking_metadata/ directory."""
        # Save raw responses
        (self.metadata_dir / "tables_raw.txt").write_text(tables_raw, encoding='utf-8')
        (self.metadata_dir / "figures_raw.txt").write_text(figures_raw, encoding='utf-8')
        
        # Save structured responses
        with open(self.metadata_dir / "tables_structured.json", 'w', encoding='utf-8') as f:
            json.dump({"tables": tables_structured}, f, indent=2)
        
        with open(self.metadata_dir / "figures_structured.json", 'w', encoding='utf-8') as f:
            json.dump({"figures": figures_structured}, f, indent=2)
        
        # Save combined metadata
        combined = {
            "tables": tables_structured,
            "figures": figures_structured
        }
        with open(self.metadata_dir / "page_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)
        
        logger.info(f"💾 Saved metadata to {self.metadata_dir}")
