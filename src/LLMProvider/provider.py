# src/LLMProvider/provider.py
"""
Unified LLM Provider for all inference tasks.
Supports: Gemini (Vertex AI), OpenAI, Novita, Groq, DeepInfra
"""
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
from pathlib import Path

# from vertexai import init as vertex_init
# from vertexai.generative_models import GenerativeModel, Part
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

# Gemini API import (lazy so openai/vllm-only runs don't require google-genai)
genai = None
types = None

def _ensure_genai():
    global genai, types
    if genai is None:
        from google import genai as _genai
        from google.genai import types as _types
        genai = _genai
        types = _types

from .models import get_model_pricing

load_dotenv()

# Vertex AI initialization flag
# _VERTEX_INITIALIZED = False

# def _ensure_vertex_init():
#     """Initialize Vertex AI once (lazy initialization)."""
#     global _VERTEX_INITIALIZED
#     if not _VERTEX_INITIALIZED:
#         project_id = os.getenv("GCP_PROJECT_ID", "")
#         location = os.getenv("GCP_LOCATION", "")
#         
#         # Set credentials path if config.json exists
#         config_json_path = Path(__file__).parent / "config.json"
#         if config_json_path.exists():
#             os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config_json_path)
#         
#         vertex_init(project=project_id, location=location)
#         _VERTEX_INITIALIZED = True


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    cost: float
    success: bool
    error: Optional[str] = None


@dataclass
class PDFHandle:
    """Provider-specific handle for an uploaded PDF (for generate_with_pdf)."""
    provider: str
    file_id: Optional[str] = None  # OpenAI Files API
    file_part: Optional[Any] = None  # Gemini Part
    image_parts: Optional[List[Any]] = None  # Fallback: list of image Parts


class LLMProvider:
    """
    Unified interface for LLM providers.
    
    Usage:
        provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
        response = provider.generate("Hello, world!")
        response = provider.generate_with_image("Describe this", image_pil)
        responses = provider.batch_generate(["prompt1", "prompt2"])
    
    Supported providers:
        - gemini: Google Gemini via Vertex AI
        - openai: OpenAI GPT models
        - novita: Novita AI (OpenAI-compatible)
        - groq: Groq (fast inference)
        - deepinfra: DeepInfra (OpenAI-compatible, supports multimodal)
    """
    
    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize LLM provider.
        
        Args:
            provider: "openai", "novita", "groq", "deepinfra", "gemini"
            model: Specific model name (uses default if not provided)
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self._client = None
        self._init_client()
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "gemini": "gemini-2.5-flash",  # Default Gemini API model
            "openai": "gpt-4o",
            "novita": "meta-llama/llama-3.1-8b-instruct",
            "groq": "llama-3.1-70b-versatile",
            "deepinfra": "Qwen/Qwen2.5-VL-32B-Instruct",
        }
        return defaults.get(self.provider, "gpt-4o")
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "gemini":
            _ensure_genai()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self._client = genai.Client(api_key=api_key)
        
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        
        elif self.provider == "novita":
            api_key = os.getenv("NOVITA_API_KEY")
            if not api_key:
                raise ValueError("NOVITA_API_KEY environment variable not set")
            self._client = OpenAI(
                base_url="https://api.novita.ai/v3/openai",
                api_key=api_key
            )
        
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLAMA_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY or LLAMA_KEY environment variable not set")
            self._client = Groq(api_key=api_key)
        
        elif self.provider == "deepinfra":
            api_key = os.getenv("DEEPINFRA_API_KEY")
            if not api_key:
                raise ValueError("DEEPINFRA_API_KEY environment variable not set")
            self._client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported: gemini, openai, novita, groq, deepinfra")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        pricing = get_model_pricing(self.provider, self.model)
        return (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 16000
    ) -> LLMResponse:
        """
        Generate text response.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction (optional)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum output tokens
        
        Returns:
            LLMResponse with text, tokens, cost, and success status
        """
        try:
            if self.provider == "gemini":
                return self._generate_gemini_api(prompt, system_prompt, temperature, max_tokens)
            else:
                return self._generate_openai_compatible(prompt, system_prompt, temperature, max_tokens)
        
        except Exception as e:
            return LLMResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                provider=self.provider,
                cost=0.0,
                success=False,
                error=str(e)
            )
    
    def _generate_gemini_api(
        self, 
        prompt: str, 
        system_prompt: str, 
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """Generate using Gemini API with proper system_instruction."""
        _ensure_genai()
        # Build config with system_instruction inside GenerateContentConfig
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None
        )
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,  # Can be string or list
            config=config
        )
        
        # Gemini API: usage_metadata may not always be present
        usage = getattr(response, 'usage_metadata', None)
        input_tokens = getattr(usage, 'prompt_token_count', 0) if usage else 0
        output_tokens = getattr(usage, 'candidates_token_count', 0) if usage else 0
        
        return LLMResponse(
            text=response.text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.provider,
            cost=self._calculate_cost(input_tokens, output_tokens),
            success=True
        )
    
    def _generate_openai_compatible(
        self, 
        prompt: str, 
        system_prompt: str, 
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """Generate using OpenAI-compatible API (OpenAI, Novita, Groq)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.provider,
            cost=self._calculate_cost(input_tokens, output_tokens),
            success=True
        )
    
    def generate_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, bytes],
        temperature: float = 0.0,
        max_tokens: int = 16000,
        system_prompt: str = None
    ) -> LLMResponse:
        """
        Generate response with image input (multimodal).
        Gemini API, OpenAI GPT-4 models, and DeepInfra multimodal models supported.
        """
        try:
            if self.provider == "gemini":
                return self._generate_gemini_api_with_image(prompt, image, temperature, max_tokens, system_prompt)
            if self.provider == "openai" and "gpt-4" in self.model:
                return self._generate_openai_with_image(prompt, image, temperature, max_tokens)
            if self.provider == "deepinfra":
                # DeepInfra supports multimodal models like Qwen2.5-VL
                return self._generate_openai_with_image(prompt, image, temperature, max_tokens)
            else:
                raise ValueError(f"Multimodal not supported for {self.provider}/{self.model}")
        
        except Exception as e:
            return LLMResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                model=self.model,
                provider=self.provider,
                cost=0.0,
                success=False,
                error=str(e)
            )
    
    def _generate_gemini_api_with_image(
        self, 
        prompt: str, 
        image: Union[Image.Image, bytes], 
        temperature: float, 
        max_tokens: int,
        system_prompt: str = None
    ) -> LLMResponse:
        """Generate with image using Gemini API with proper system_instruction."""
        _ensure_genai()
        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = image
        # Gemini API expects dict with 'mime_type' and 'data'
        gemini_image = {
            "mime_type": "image/png",
            "data": image_bytes
        }
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt if system_prompt else None
        )
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=[prompt, gemini_image],  # List with prompt and image
            config=config
        )
        usage = getattr(response, 'usage_metadata', None)
        input_tokens = getattr(usage, 'prompt_token_count', 0) if usage else 0
        output_tokens = getattr(usage, 'candidates_token_count', 0) if usage else 0
        return LLMResponse(
            text=response.text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.provider,
            cost=self._calculate_cost(input_tokens, output_tokens),
            success=True
        )
    
    def _generate_openai_with_image(
        self, 
        prompt: str, 
        image: Union[Image.Image, bytes], 
        temperature: float, 
        max_tokens: int
    ) -> LLMResponse:
        """Generate with image using GPT-4V."""
        import base64
        
        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = image
        
        b64_image = base64.b64encode(image_bytes).decode()
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        usage = getattr(response, 'usage', None)
        input_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
        
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.provider,
            cost=self._calculate_cost(input_tokens, output_tokens),
            success=True
        )

    def upload_pdf(self, pdf_path: Union[str, Path]) -> PDFHandle:
        """
        Upload PDF to provider's file API.
        Returns provider-specific handle for use with generate_with_pdf.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if self.provider == "openai":
            with open(pdf_path, "rb") as f:
                uploaded = self._client.files.create(file=f, purpose="assistants")
            return PDFHandle(provider="openai", file_id=uploaded.id)
        if self.provider == "gemini":
            _ensure_genai()
            pdf_bytes = pdf_path.read_bytes()
            part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            return PDFHandle(provider="gemini", file_part=part)
        raise ValueError(f"PDF upload not supported for {self.provider}")

    def generate_with_pdf(
        self,
        prompt: str,
        pdf_handle: PDFHandle,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """
        Generate text response with PDF context.
        Uses provider's native PDF support (OpenAI Responses API, Gemini PDF Part).
        """
        if pdf_handle.provider != self.provider:
            raise ValueError(f"PDFHandle from {pdf_handle.provider} does not match provider {self.provider}")
        if self.provider == "openai":
            if not pdf_handle.file_id:
                raise ValueError("OpenAI PDFHandle missing file_id")
            response = self._client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": pdf_handle.file_id},
                    ],
                }],
            )
            text = (response.output_text or "").strip()
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                provider=self.provider,
                cost=self._calculate_cost(input_tokens, output_tokens),
                success=True,
            )
        if self.provider == "gemini":
            if not pdf_handle.file_part:
                raise ValueError("Gemini PDFHandle missing file_part")
            _ensure_genai()
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt if system_prompt else None,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[pdf_handle.file_part, prompt],
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            return LLMResponse(
                text=(response.text or "").strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                provider=self.provider,
                cost=self._calculate_cost(input_tokens, output_tokens),
                success=True,
            )
        raise ValueError(f"generate_with_pdf not supported for {self.provider}")

    def cleanup_pdf(self, pdf_handle: PDFHandle) -> None:
        """Delete uploaded PDF file (OpenAI only; no-op for Gemini)."""
        if pdf_handle.provider == "openai" and pdf_handle.file_id:
            try:
                self._client.files.delete(pdf_handle.file_id)
            except Exception:
                pass

    def batch_generate(
        self,
        prompts: list,
        system_prompt: str = None,
        max_workers: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 16000
    ) -> list:
        """
        Generate responses for multiple prompts in parallel.
        
        Args:
            prompts: List of prompts
            system_prompt: System instruction (applied to all)
            max_workers: Number of parallel threads
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
        
        Returns:
            List of LLMResponse in same order as prompts
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.generate, prompt, system_prompt, temperature, max_tokens
                ): i
                for i, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = LLMResponse(
                        text="",
                        input_tokens=0,
                        output_tokens=0,
                        model=self.model,
                        provider=self.provider,
                        cost=0.0,
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    @property
    def client(self):
        """Raw client for provider-specific use (e.g. fallback structurer)."""
        return self._client

    def __repr__(self) -> str:
        return f"LLMProvider(provider='{self.provider}', model='{self.model}')"

