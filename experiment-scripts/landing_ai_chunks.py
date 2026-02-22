"""
Landing AI chunks module.

Fetches document chunks from Landing AI ADE Parse API and converts them
to the format expected by verify_plans_with_pdf.py (compatible with
find_relevant_chunks_for_group, format_chunks, etc.).

Usage:
    chunks = load_landing_ai_chunks(pdf_path, cache_dir=None)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Map Landing AI chunk types to pipeline format (text, table, figure)
# Landing AI uses types like chunkText, chunkTable, chunkFigure (or lowercase)
def _landing_type_to_pipeline(landing_type: str) -> str:
    t = str(landing_type or "").lower()
    if "table" in t or t == "table":
        return "table"
    if "figure" in t or "logo" in t or "scancode" in t:
        return "figure"
    return "text"


def _ensure_api_key() -> None:
    api_key = os.getenv("VISION_AGENT_API_KEY") or os.getenv("LANDING_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set VISION_AGENT_API_KEY (preferred) or LANDING_AI_API_KEY."
        )
    if not os.getenv("VISION_AGENT_API_KEY"):
        os.environ["VISION_AGENT_API_KEY"] = api_key


def _init_client():
    from landingai_ade import LandingAIADE

    env = os.getenv("LANDING_AI_ENV", "").strip().lower()
    if env == "eu":
        return LandingAIADE(environment="eu")
    return LandingAIADE()


def _serialize_chunk(chunk: Any) -> Dict[str, Any]:
    """Convert a Landing AI Chunk object to a dict."""
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()
    if isinstance(chunk, dict):
        return chunk
    return dict(chunk)





def parse_chunk_to_pipeline_format(chunk: Any, chunk_index: int) -> Dict[str, Any]:
    """
    Convert a single Landing AI parse chunk to pipeline format.

    Pipeline format expects:
        - type: "text" | "table" | "figure"
        - page: int (1-based) or str like "1-4"
        - content: str
        - table_content: str (for tables; markdown table body)
    """
    d = _serialize_chunk(chunk)
    landing_type = str(d.get("type", "text")).lower()
    pipeline_type = _landing_type_to_pipeline(landing_type)

    # Page: Landing AI uses 0-indexed; pipeline uses 1-based
    grounding = d.get("grounding") or {}
    if isinstance(grounding, dict):
        page_0 = grounding.get("page", 0)
    else:
        page_0 = getattr(grounding, "page", 0)
    page = int(page_0) + 1 if isinstance(page_0, (int, float)) else 1

    markdown = str(d.get("markdown", "") or "")

    out: Dict[str, Any] = {
        "type": pipeline_type,
        "page": page,
        "content": markdown,
        "table_content": "",
    }
    if pipeline_type == "table":
        out["table_content"] = markdown
    return out


def parse_response_to_pipeline_chunks(parse_response: Any) -> List[Dict[str, Any]]:
    """
    Convert a full Landing AI ParseResponse to a list of pipeline-format chunks.
    """
    chunks = getattr(parse_response, "chunks", None) or []
    if not chunks and isinstance(parse_response, dict):
        chunks = parse_response.get("chunks", [])

    result: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        result.append(parse_chunk_to_pipeline_format(chunk, idx))
    return result


def load_landing_ai_chunks(
    pdf_path: Path,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load chunks from Landing AI ADE Parse for the given PDF.

    Args:
        pdf_path: Path to the PDF file.
        cache_dir: Optional directory to cache parse output (saves parse_output.json).
        use_cache: If True and cache_dir has parse_output.json, load from cache.

    Returns:
        List of chunks in pipeline format (type, page, content, table_content).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "landing_ai_parse_output.json"

    if use_cache and cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return parse_response_to_pipeline_chunks(data)
        except Exception:
            pass

    _ensure_api_key()
    client = _init_client()

    parse_model = os.getenv("LANDING_AI_PARSE_MODEL", "dpt-2-latest")
    response = client.parse(
        document=pdf_path,
        model=parse_model,
    )

    chunks = parse_response_to_pipeline_chunks(response)

    if cache_file:
        # Serialize response for caching
        if hasattr(response, "model_dump"):
            to_save = response.model_dump()
        elif hasattr(response, "dict"):
            to_save = response.dict()
        else:
            to_save = {"chunks": chunks}
        cache_file.write_text(json.dumps(to_save, indent=2, ensure_ascii=False), encoding="utf-8")

    return chunks
