# src/LLMProvider/models.py
"""
Model definitions and pricing per 1K tokens.
Prices are approximate and should be updated as needed.
"""

# Pricing: USD per 1K tokens (input, output). Update as provider pricing changes.
SUPPORTED_MODELS = {
    "gemini": {
        "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-2.0-flash-001": {"input": 0.0001, "output": 0.0004},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    },
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4.1": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "novita": {
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.0002, "output": 0.0002},
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.0009, "output": 0.0009},
        "meta-llama/llama-3.3-70b-instruct": {"input": 0.0009, "output": 0.0009},
        "deepseek/deepseek-v3": {"input": 0.0014, "output": 0.0028},
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    },
    "deepinfra": {
        "Qwen/Qwen2.5-VL-32B-Instruct": {"input": 0.0005, "output": 0.0005},
        "deepinfra/deepseek-v3": {"input": 0.0007, "output": 0.0014},
        "meta-llama/Llama-3.1-70B-Instruct": {"input": 0.00059, "output": 0.00079},
        "meta-llama/Llama-3.1-8B-Instruct": {"input": 0.0001, "output": 0.0001},
    },
    "local": {
        "Qwen3-8B": {"input": 0.0, "output": 0.0},  # Local model, no API costs
        "Qwen/Qwen3-8B": {"input": 0.0, "output": 0.0},
    }
}


def get_model_pricing(provider: str, model: str) -> dict:
    """
    Get pricing for a specific model.
    
    Args:
        provider: Provider name (gemini, openai, novita, groq, deepinfra)
        model: Model name
    
    Returns:
        dict with 'input' and 'output' costs per 1K tokens
    """
    return SUPPORTED_MODELS.get(provider, {}).get(model, {"input": 0, "output": 0})


def list_supported_models(provider: str = None) -> dict:
    """
    List supported models, optionally filtered by provider.
    
    Args:
        provider: Optional provider to filter by
    
    Returns:
        dict of supported models
    """
    if provider:
        return {provider: SUPPORTED_MODELS.get(provider, {})}
    return SUPPORTED_MODELS

