"""
QA adapter for single-query extraction.

Builds definition string with conversation context for multi-turn QA.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Configurable: number of previous Q&A turns to inject into extraction prompt
QA_CONTEXT_TURNS = 5


def build_definition_with_context(
    current_question: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Build definition string with conversation context for Agent and Search extraction.

    Args:
        current_question: The user's current question
        history: List of {"question": str, "answer": str} (most recent last). Use reconciled answer.

    Returns:
        Definition string to pass as column definition. Includes previous Q&A when available.
    """
    if not history:
        return current_question

    turns = history[-QA_CONTEXT_TURNS:]
    block = "\n".join(
        f"- Q: {h.get('question', '')}\n  A: {h.get('answer', '')}"
        for h in turns
    )
    return f"Previous Q&A:\n{block}\n\nCurrent question: {current_question}"
