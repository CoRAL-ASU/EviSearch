"""
extraction_service.py

Service module that wraps the baseline_file_search_gemini_native.py functionality
for use in the web interface.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, OrderedDict

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "experiment-scripts"))

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

load_dotenv()


class ExtractionService:
    """Service for extracting clinical trial data from PDFs using Gemini."""
    
    def __init__(self, model: str = "gemini-2.0-flash-001"):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google.genai is required. Install with: pip install google-genai")
        
        self.model = model
        self.client = None  # Lazy initialization
        self._pdf_part = None
        self.current_pdf_path = None
        
        # Load column definitions
        self.definitions = self._load_definitions()
    
    def _ensure_client(self):
        """Ensure Gemini client is initialized (lazy initialization)."""
        if self.client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not set in environment")
            self.client = genai.Client(api_key=api_key)
        
    def _load_definitions(self) -> Dict[str, Dict[str, str]]:
        """Load column definitions from CSV using the definitions.py approach."""
        # Try Definitions_open_ended.csv first (from config), fallback to Definitions_with_eval_category.csv
        definitions_path = repo_root / "src" / "table_definitions" / "Definitions_open_ended.csv"
        
        if not definitions_path.exists():
            definitions_path = repo_root / "src" / "table_definitions" / "Definitions_with_eval_category.csv"
        
        if not definitions_path.exists():
            raise FileNotFoundError(f"No definitions CSV found in src/table_definitions/")
        
        definitions = {}
        
        with open(definitions_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            header = lines[0].strip().split(',')
            
            # Find column indices
            col_name_idx = next((i for i, h in enumerate(header) if 'Column Name' in h), 0)
            def_idx = next((i for i, h in enumerate(header) if 'Definition' in h), 1)
            label_idx = next((i for i, h in enumerate(header) if 'Label' in h), 2)
            
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',', max(col_name_idx, def_idx, label_idx) + 1)
                if len(parts) > max(col_name_idx, def_idx, label_idx):
                    col_name = parts[col_name_idx].strip().strip('"')
                    definition = parts[def_idx].strip().strip('"')
                    label = parts[label_idx].strip().strip('"') if len(parts) > label_idx else ""
                    
                    if col_name and definition:
                        definitions[col_name] = {
                            "definition": definition,
                            "label": label
                        }
        
        return definitions
    
    def upload_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Upload a PDF file for extraction."""
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            return {"success": False, "error": f"PDF file not found: {pdf_path}"}
        
        try:
            # Ensure client is initialized before using it
            self._ensure_client()
            
            pdf_bytes = pdf_path_obj.read_bytes()
            self._pdf_part = genai_types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
            self.current_pdf_path = pdf_path
            
            return {
                "success": True,
                "message": f"PDF loaded successfully ({len(pdf_bytes)} bytes)",
                "filename": pdf_path_obj.name
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to load PDF: {str(e)}"}
    
    def _build_prompt_for_columns(self, columns: List[Dict[str, str]]) -> str:
        """Build extraction prompt for specific columns."""
        lines = ["Extract values for the following columns:\n"]
        
        for i, col in enumerate(columns, 1):
            lines.append(
                f"{i}. {col['column']}: {col['definition']}\n"
                "   If not present, use value: 'not found' and reasoning: 'not found'."
            )
        
        lines.append("\n" + "=" * 60)
        lines.append(
            "For each column, provide:\n"
            "- 'value': the extracted value (or 'not found')\n"
            "- 'page_number': the page number as a string where the value was found (or 'unknown' if not found)\n"
            "- 'modality': the type of content where the value was found (e.g., 'text', 'table', 'figure') or 'unknown' if not found\n"
            "- 'reasoning': brief explanation of where and how you found the value (or 'not found')"
        )
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _build_json_schema(self, columns: List[str]) -> Dict[str, Any]:
        """Build JSON schema for extraction with location information."""
        properties = {}
        
        for col in columns:
            properties[col] = {
                "type": "OBJECT",
                "properties": {
                    "value": {
                        "type": "STRING",
                        "description": "The extracted value exactly as in the document; use 'not found' if not reported."
                    },
                    "page_number": {
                        "type": "STRING",
                        "description": "The page number where the value was found (as string); use 'unknown' if not found."
                    },
                    "modality": {
                        "type": "STRING",
                        "description": "The type of content: 'text', 'table', 'figure', or 'unknown' if not found."
                    },
                    "reasoning": {
                        "type": "STRING",
                        "description": "Brief explanation of where and how the value was found; use 'not found' if not found."
                    }
                },
                "required": ["value", "page_number", "modality", "reasoning"]
            }
        
        return {
            "type": "OBJECT",
            "properties": properties,
            "required": list(columns)
        }
    
    def _query_gemini(self, prompt: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Query Gemini with the PDF and prompt."""
        if self._pdf_part is None:
            raise ValueError("No PDF uploaded. Please upload a PDF first.")
        
        # Ensure client is initialized
        self._ensure_client()
        
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=json_schema,
        )
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, self._pdf_part],
            config=config,
        )
        
        response_text = (response.text or "").strip()
        
        try:
            return json.loads(response_text) if response_text else {}
        except json.JSONDecodeError:
            return {"_error": "JSON decode failed", "_raw": response_text}
    
    def extract_single_column(self, column_name: str, definition: Optional[str] = None) -> Dict[str, Any]:
        """Extract a single column value."""
        if not definition:
            # Use definition from loaded definitions
            if column_name not in self.definitions:
                return {
                    "success": False,
                    "error": f"Column '{column_name}' not found in definitions"
                }
            definition = self.definitions[column_name]["definition"]
        
        columns = [{"column": column_name, "definition": definition}]
        prompt = self._build_prompt_for_columns(columns)
        schema = self._build_json_schema([column_name])
        
        try:
            result = self._query_gemini(prompt, schema)
            
            if "_error" in result:
                return {"success": False, "error": result.get("_error"), "raw": result.get("_raw")}
            
            column_data = result.get(column_name, {})
            
            return {
                "success": True,
                "column": column_name,
                "value": column_data.get("value", "not found"),
                "page_number": column_data.get("page_number"),
                "modality": column_data.get("modality"),
                "evidence": column_data.get("reasoning", ""),
                "definition": definition
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_all_columns(self) -> Dict[str, Any]:
        """Extract all 133 columns."""
        # Group columns by label for efficient extraction
        label_groups = defaultdict(list)
        for col_name, col_info in self.definitions.items():
            label_groups[col_info["label"]].append({
                "column": col_name,
                "definition": col_info["definition"]
            })
        
        all_results = {}
        errors = []
        
        for label, columns in label_groups.items():
            column_names = [c["column"] for c in columns]
            prompt = self._build_prompt_for_columns(columns)
            schema = self._build_json_schema(column_names)
            
            try:
                result = self._query_gemini(prompt, schema)
                
                if "_error" in result:
                    errors.append(f"Error in label group '{label}': {result.get('_error')}")
                    continue
                
                # Process results for this group
                for col_name in column_names:
                    col_data = result.get(col_name, {})
                    all_results[col_name] = {
                        "value": col_data.get("value", "not found"),
                        "page_number": col_data.get("page_number"),
                        "modality": col_data.get("modality"),
                        "evidence": col_data.get("reasoning", ""),
                        "definition": self.definitions[col_name]["definition"]
                    }
                    
            except Exception as e:
                errors.append(f"Exception in label group '{label}': {str(e)}")
        
        return {
            "success": len(all_results) > 0,
            "results": all_results,
            "total_columns": len(all_results),
            "errors": errors if errors else None
        }
    
    def extract_from_csv(self, csv_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract values for columns specified in CSV format."""
        results = {}
        errors = []
        
        for row in csv_data:
            column_name = row.get("column_name") or row.get("Column Name")
            definition = row.get("definition") or row.get("Definition")
            
            if not column_name or not definition:
                errors.append(f"Invalid row: missing column_name or definition")
                continue
            
            try:
                result = self.extract_single_column(column_name, definition)
                if result.get("success"):
                    results[column_name] = {
                        "value": result.get("value"),
                        "page_number": result.get("page_number"),
                        "modality": result.get("modality"),
                        "evidence": result.get("evidence"),
                        "definition": definition
                    }
                else:
                    errors.append(f"Failed to extract '{column_name}': {result.get('error')}")
            except Exception as e:
                errors.append(f"Exception extracting '{column_name}': {str(e)}")
        
        return {
            "success": len(results) > 0,
            "results": results,
            "total_columns": len(results),
            "errors": errors if errors else None
        }
    
    def get_available_columns(self) -> List[Dict[str, str]]:
        """Get list of all available columns and their definitions."""
        return [
            {
                "column_name": col_name,
                "definition": col_info["definition"],
                "label": col_info["label"]
            }
            for col_name, col_info in self.definitions.items()
        ]
    
    def cleanup(self):
        """Clean up resources."""
        self._pdf_part = None
        self.current_pdf_path = None
