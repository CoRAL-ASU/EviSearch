# src/utils/costing.py
"""
End-to-end costing for the extraction pipeline.
Uses token counts from LLM calls and per-model pricing to compute cost per module and total.
"""
from typing import Any, Dict, List

from src.LLMProvider.models import get_model_pricing


def compute_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Compute cost in USD from token usage.

    Args:
        provider: Provider name (gemini, openai, etc.)
        model: Model name
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Cost in USD (pricing is per 1K tokens)
    """
    pricing = get_model_pricing(provider, model)
    return (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)


def usage_to_cost_dict(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> Dict[str, Any]:
    """
    Build a cost dict for a single module/call: tokens and cost.

    Returns:
        dict with keys: input_tokens, output_tokens, provider, model, cost_usd
    """
    cost = compute_cost(provider, model, input_tokens, output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider": provider,
        "model": model,
        "cost_usd": round(cost, 6),
    }


def aggregate_usage(
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate multiple usage dicts (sum input_tokens, output_tokens).
    Assumes each item has input_tokens, output_tokens, provider, model.
    Uses provider/model from first item for the aggregated record; cost recomputed from totals.
    """
    total_in = sum(i.get("input_tokens", 0) for i in items)
    total_out = sum(i.get("output_tokens", 0) for i in items)
    if not items:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "provider": None,
            "model": None,
            "cost_usd": 0.0,
        }
    provider = items[0].get("provider", "")
    model = items[0].get("model", "")
    cost = compute_cost(provider, model, total_in, total_out)
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "provider": provider,
        "model": model,
        "cost_usd": round(cost, 6),
    }


def build_pipeline_cost_summary(
    chunking: Dict[str, Any] | None = None,
    planning: Dict[str, Any] | None = None,
    extraction: Dict[str, Any] | None = None,
    evaluation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the full pipeline cost summary for one PDF.

    Each module dict should have at least: input_tokens, output_tokens, cost_usd (and optionally provider, model).
    """
    modules = {
        "chunking": chunking or _empty_module(),
        "planning": planning or _empty_module(),
        "extraction": extraction or _empty_module(),
        "evaluation": evaluation or _empty_module(),
    }
    total_input = sum(m.get("input_tokens", 0) for m in modules.values())
    total_output = sum(m.get("output_tokens", 0) for m in modules.values())
    total_cost = sum(m.get("cost_usd", 0) for m in modules.values())
    return {
        "by_module": modules,
        "total": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 6),
        },
    }


def _empty_module() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "provider": None,
        "model": None,
        "cost_usd": 0.0,
    }
