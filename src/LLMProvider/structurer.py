# src/LLMProvider/structurer.py
"""
Output Structurer for converting free-form LLM reasoning into structured JSON.
Uses a local model (localhost:8001) with OpenAI-compatible API.
"""
import json
import re
from typing import Type, TypeVar, Union, Optional
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)


@dataclass
class StructurerResponse:
    """Response from the output structurer."""
    data: Union[BaseModel, dict]
    success: bool
    attempts: int
    error: Optional[str] = None


class OutputStructurer:
    """
    Structures free-form text into JSON using a local LLM.
    
    Usage:
        structurer = OutputStructurer(base_url="http://localhost:8001")
        result = structurer.structure(
            text="Patient is 65 years old with prostate cancer...",
            schema=PatientDataModel,
            max_retries=3
        )
        
        if result.success:
            patient_data = result.data  # Pydantic model instance
            print(patient_data.age)
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-8B",  # Full HuggingFace path
        api_key: str = "not-needed",  # Local models don't need real API keys
        debug_file: Optional[str] = None,  # Path to debug file for logging
        enable_thinking: bool = False  # Enable <think> tags in model output
    ):
        """
        Initialize the output structurer.
        
        Args:
            base_url: Base URL for the local model API (default: http://localhost:8001/v1)
            model: Model name (default: Qwen/Qwen3-8B)
            api_key: API key (not needed for local, but required by OpenAI SDK)
            debug_file: Optional path to file for logging raw LLM responses
            enable_thinking: Enable model thinking/reasoning output (default: False)
        """
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.debug_file = debug_file
        self.enable_thinking = enable_thinking
    
    def structure(
        self,
        text: str,
        schema: Type[T],
        max_retries: int = 3,
        temperature: float = 0.0,
        return_dict: bool = False
    ) -> StructurerResponse:
        """
        Structure free-form text into JSON matching the provided schema.
        
        Args:
            text: Free-form reasoning text from LLM
            schema: Pydantic BaseModel class defining the expected structure
            max_retries: Number of retry attempts if validation fails (default: 3)
            temperature: Sampling temperature (default: 0.0 for deterministic)
            return_dict: If True, return dict instead of Pydantic instance (default: False)
        
        Returns:
            StructurerResponse with structured data (Pydantic model or dict)
        """
        # Generate JSON schema from Pydantic model
        json_schema = schema.model_json_schema()
        
        # Create prompt for structuring
        prompt = self._create_structuring_prompt(text, json_schema)
        
        # Attempt to structure with retries
        for attempt in range(1, max_retries + 1):
            try:
                # Call local model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON structuring assistant. Extract and structure information from the given text into valid JSON matching the provided schema. Output ONLY valid JSON, no explanations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    }
                )
                
                # Extract response text
                response_text = response.choices[0].message.content.strip()
                
                # DEBUG: Log raw LLM response
                if self.debug_file:
                    with open(self.debug_file, 'a') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"RAW LOCAL LLM RESPONSE (Attempt {attempt}):\n")
                        f.write(f"{'='*80}\n")
                        f.write(response_text)
                        f.write("\n")
                
                # Try to parse JSON (returns list or dict)
                structured_json = self._extract_json(response_text)
                
                # DEBUG: Log what we extracted
                if self.debug_file:
                    with open(self.debug_file, 'a') as f:
                        f.write(f"\n📦 Extracted JSON type: {type(structured_json).__name__}\n")
                        f.write(f"📦 Extracted JSON: {json.dumps(structured_json, indent=2)[:200]}...\n")
                
                # Auto-wrap if needed:
                # Option 1: LLM returned array → wrap with schema's array field key
                # Option 2: LLM returned dict with correct key → use as-is
                if isinstance(structured_json, list):
                    array_field = self._find_array_field(schema)
                    if array_field:
                        if self.debug_file:
                            with open(self.debug_file, 'a') as f:
                                f.write(f"⚠️  Auto-wrapping bare array with key '{array_field}'\n")
                        structured_json = {array_field: structured_json}
                    else:
                        raise ValueError(f"Schema has no array field, but LLM returned array")
                
                # Validate against Pydantic schema
                structured_data = schema.model_validate(structured_json)
                
                # Success!
                return StructurerResponse(
                    data=structured_data.model_dump() if return_dict else structured_data,
                    success=True,
                    attempts=attempt
                )
            
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries:
                    # Final attempt failed
                    return StructurerResponse(
                        data={},
                        success=False,
                        attempts=attempt,
                        error=f"Failed after {max_retries} attempts. Last error: {str(e)}"
                    )
                # Continue to next retry
                continue
            
            except Exception as e:
                # Unexpected error (network, API, etc.)
                return StructurerResponse(
                    data={},
                    success=False,
                    attempts=attempt,
                    error=f"Unexpected error: {str(e)}"
                )
        
        # Should not reach here, but just in case
        return StructurerResponse(
            data={},
            success=False,
            attempts=max_retries,
            error="Unknown error occurred"
        )
    
    def _find_array_field(self, schema: Type[T]) -> Optional[str]:
        """
        Find the first List field name in the Pydantic schema.
        Used to auto-wrap bare arrays when LLM outputs [...] instead of {"field": [...]}.
        
        Args:
            schema: Pydantic BaseModel class
        
        Returns:
            Field name (str) or None if no List field found
        """
        for field_name, field_info in schema.model_fields.items():
            # Check if field annotation is a List type
            annotation = field_info.annotation
            
            # Handle typing.List, list, and Optional[List] cases
            if hasattr(annotation, '__origin__'):
                origin = annotation.__origin__
                # Check for list or List
                if origin is list:
                    return field_name
                # Handle Union types (e.g., Optional[List[...]] = Union[List, None])
                if origin is Union:
                    for arg in annotation.__args__:
                        if hasattr(arg, '__origin__') and arg.__origin__ is list:
                            return field_name
        
        return None
    
    def _create_structuring_prompt(self, text: str, json_schema: dict) -> str:
        """Create the structuring prompt with text and schema."""
        # Extract schema information for prompt
        schema_description = self._format_schema_for_prompt(json_schema)
        
        return f"""Extract information from the following text and structure it as JSON matching the schema below.

SCHEMA:
{schema_description}

TEXT TO STRUCTURE:
{text}

Output ONLY valid JSON matching the schema above. No explanations, no markdown formatting, no extra text."""
    
    def _format_schema_for_prompt(self, json_schema: dict) -> str:
        """Format JSON schema into a readable prompt description."""
        lines = []
        
        # Add title/description if present
        if "title" in json_schema:
            lines.append(f"Object: {json_schema['title']}")
        if "description" in json_schema:
            lines.append(f"Description: {json_schema['description']}")
        
        # Add properties
        if "properties" in json_schema:
            lines.append("\nFields:")
            for field_name, field_info in json_schema["properties"].items():
                field_type = field_info.get("type", "unknown")
                field_desc = field_info.get("description", "")
                
                # Handle array types
                if field_type == "array" and "items" in field_info:
                    items_info = field_info["items"]
                    if "$ref" in items_info:
                        # Reference to another schema - just show as array
                        lines.append(f"  - {field_name}: array of objects")
                        # Try to extract item properties if in definitions
                        if "definitions" in json_schema or "$defs" in json_schema:
                            defs = json_schema.get("definitions", json_schema.get("$defs", {}))
                            ref_name = items_info["$ref"].split("/")[-1]
                            if ref_name in defs:
                                item_props = defs[ref_name].get("properties", {})
                                lines.append(f"    Each item has:")
                                for item_field, item_info in item_props.items():
                                    item_type = item_info.get("type", "unknown")
                                    item_desc = item_info.get("description", "")
                                    lines.append(f"      • {item_field} ({item_type}): {item_desc}")
                    else:
                        lines.append(f"  - {field_name}: {field_type} - {field_desc}")
                else:
                    lines.append(f"  - {field_name} ({field_type}): {field_desc}")
        
        return "\n".join(lines)
    
    def _strip_think_tags(self, text: str) -> str:
        """
        Remove <think>...</think> blocks from LLM response.
        Some models output reasoning in think tags before the actual response.
        """
        # Remove everything between <think> and </think> (including tags)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
    
    def _extract_json(self, text: str):
        """
        Extract and parse JSON from response text.
        After stripping think tags, this should be clean JSON (array or object).
        Returns dict or list.
        """
        # STEP 1: Strip <think> tags
        text = self._strip_think_tags(text)
        
        # STEP 2: Parse the JSON directly - should be clean after stripping tags
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # If it fails, log for debugging and re-raise
            if self.debug_file:
                with open(self.debug_file, 'a') as f:
                    f.write(f"\n❌ JSON Parse Error: {e}\n")
                    f.write(f"Text after think tag stripping:\n{text[:500]}\n")
            raise
    
    
    def __repr__(self) -> str:
        return f"OutputStructurer(base_url='{self.base_url}', model='{self.model}', enable_thinking={self.enable_thinking})"
