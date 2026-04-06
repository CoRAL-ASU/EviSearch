"""
Landing AI parse service for QA.

Runs Landing AI ADE Parse, stores parsed_markdown.md and landing_ai_parse_output.json
in results/<doc_id>/chunking/. Used by QA prepare-document flow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.config.runtime_paths import RESULTS_ROOT, ensure_runtime_dirs


def _ensure_api_key() -> None:
    api_key = os.getenv("VISION_AGENT_API_KEY") or os.getenv("LANDING_AI_API_KEY")
    if not api_key:
        raise ValueError("VISION_AGENT_API_KEY or LANDING_AI_API_KEY required")
    if not os.getenv("VISION_AGENT_API_KEY"):
        os.environ["VISION_AGENT_API_KEY"] = api_key


def _init_client():
    from landingai_ade import LandingAIADE

    env = os.getenv("LANDING_AI_ENV", "").strip().lower()
    if env == "eu":
        return LandingAIADE(environment="eu")
    return LandingAIADE()


def _serialize_response(resp: Any) -> Dict[str, Any]:
    """Serialize parse response to JSON-serializable dict."""
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    if isinstance(resp, dict):
        return resp
    return {"markdown": getattr(resp, "markdown", ""), "chunks": getattr(resp, "chunks", [])}


def parse_pdf_for_qa(
    doc_id: str,
    pdf_path: Path,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Landing AI Parse. Store parsed_markdown.md and landing_ai_parse_output.json
    in results/<doc_id>/chunking/.

    Args:
        doc_id: Document ID (e.g. upload_abc123 or NCT00268476_Attard_STAMPEDE_Lancet'23)
        pdf_path: Path to PDF file
        on_event: Optional callback for progress events {"stage": str, "message": str}

    Returns:
        {"success": bool, "error": str?, "parsed_markdown_path": str?, "parse_output_path": str?}
    """
    def emit(stage: str, message: str, **kwargs: Any) -> None:
        if on_event:
            try:
                on_event({"stage": stage, "message": message, **kwargs})
            except Exception:
                pass

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"success": False, "error": f"PDF not found: {pdf_path}"}

    ensure_runtime_dirs()
    chunk_dir = RESULTS_ROOT / doc_id / "chunking"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    md_path = chunk_dir / "parsed_markdown.md"
    json_path = chunk_dir / "landing_ai_parse_output.json"

    emit("parsing", "Parsing PDF with Landing AI…")

    try:
        _ensure_api_key()
    except ValueError as e:
        return {"success": False, "error": str(e)}

    try:
        client = _init_client()
        parse_model = os.getenv("LANDING_AI_PARSE_MODEL", "dpt-2-latest")
        response = client.parse(document=pdf_path, model=parse_model)
    except Exception as e:
        return {"success": False, "error": str(e)}

    markdown = getattr(response, "markdown", None)
    if not markdown:
        return {"success": False, "error": "No markdown returned from Landing AI Parse"}

    emit("saving", "Saving parsed output…")

    md_path.write_text(markdown, encoding="utf-8")
    data = _serialize_response(response)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "parsed_markdown_path": str(md_path),
        "parse_output_path": str(json_path),
    }
