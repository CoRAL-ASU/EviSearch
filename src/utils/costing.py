# src/utils/costing.py
"""Token usage to cost, using the per-model prices in src/config/catalog.yaml."""
from typing import Any, Dict, List, Optional


def usage_to_cost_dict(model_key: Optional[str], input_tokens: int, output_tokens: int) -> Dict[str, Any]:
    from src.config.config import CATALOG
    from src.inference.factory import cost_usd

    model = CATALOG.models.get(model_key or "")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model_key,
        "endpoint": model.endpoint if model else None,
        "cost_usd": cost_usd(model_key, input_tokens, output_tokens),
    }


def aggregate_usage(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum usage dicts that share a model; cost is recomputed from the totals."""
    model_key = items[0].get("model") if items else None
    return usage_to_cost_dict(
        model_key,
        sum(i.get("input_tokens", 0) for i in items),
        sum(i.get("output_tokens", 0) for i in items),
    )
