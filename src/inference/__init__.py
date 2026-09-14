"""Inference layer: role-based access to chat, embedding and reranking models.

    from src.inference import get_chat, Message
    chat = get_chat("search_agent")          # model comes from src/config/config.py
    result = chat.chat([Message.system("..."), Message.user("...")], tools=[...])
"""
from src.inference.base import ChatModel, Embedder, Reranker
from src.inference.factory import (
    check_selection,
    cost_usd,
    embedding_model_id,
    get_chat,
    get_embedder,
    get_reranker,
    reset_cache,
)
from src.inference.tool_loop import LoopResult, Tool, ToolOutput, run_tool_loop
from src.inference.types import (
    ChatResult,
    ImagePart,
    InferenceError,
    Message,
    Part,
    PdfPart,
    TextPart,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    parse_json_text,
)

__all__ = [
    "ChatModel",
    "ChatResult",
    "Embedder",
    "ImagePart",
    "InferenceError",
    "LoopResult",
    "Message",
    "Part",
    "PdfPart",
    "Reranker",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolOutput",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "check_selection",
    "cost_usd",
    "embedding_model_id",
    "get_chat",
    "get_embedder",
    "get_reranker",
    "parse_json_text",
    "reset_cache",
    "run_tool_loop",
]
