"""Provider-neutral tool-calling loop shared by the search and reconciliation agents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.inference.base import ChatModel
from src.inference.openai_compat import INVALID_ARGUMENTS_KEY
from src.inference.types import ImagePart, InferenceError, Message, Part, TextPart, ToolCall, ToolResult, ToolSpec, Usage

CHARS_PER_TOKEN = 3.0  # conservative estimate for dense tables and numbers
IMAGE_TOKEN_ESTIMATE = 2000
CONTEXT_MARGIN_TOKENS = 1024
EVICTED_CONTENT = {
    "evicted": True,
    "note": "This tool output was removed to fit the context window. Call the tool again if you still need it.",
}


@dataclass
class ToolOutput:
    content: Dict[str, Any]
    attachments: List[Part] = field(default_factory=list)
    stop: bool = False  # end the loop after this turn (e.g. final submission)
    on_evict: Optional[Callable[[], None]] = None  # called if the loop drops this output to fit the context window


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[Dict[str, Any]], ToolOutput]


@dataclass
class LoopResult:
    messages: List[Message]
    usage: Usage
    turns: int
    tool_calls: int
    stopped_by: str  # finish_tool | done | no_tool_call | max_turns | max_tool_calls | error
    error: Optional[str]
    transcript: List[Dict[str, Any]]


def run_tool_loop(
    chat: ChatModel,
    *,
    system: str,
    user: str,
    tools: Sequence[Tool],
    max_turns: int,
    max_tool_calls: int,
    max_tokens: int,
    temperature: float = 0.0,
    follow_up: Optional[str] = None,
    is_done: Optional[Callable[[], bool]] = None,
) -> LoopResult:
    """Let the model call tools until a handler stops the loop, is_done() is true, or a limit is hit.

    The full conversation is kept. When a model has a finite context window, the oldest tool outputs are
    replaced with a note (and their on_evict hooks run) so the next request still fits.
    """
    registry = {tool.spec.name: tool for tool in tools}
    specs = [tool.spec for tool in tools]
    messages: List[Message] = [Message.system(system), Message.user(user)]
    transcript: List[Dict[str, Any]] = [{"turn": 0, "role": "user", "content": user}]
    evictable: List[Tuple[ToolResult, Optional[Callable[[], None]]]] = []
    usage = Usage()
    turns = 0
    tool_calls = 0
    stopped_by = "max_turns"
    error: Optional[str] = None

    while turns < max_turns:
        if tool_calls >= max_tool_calls:
            stopped_by = "max_tool_calls"
            break
        _fit_context(chat, messages, evictable, max_tokens)
        turns += 1
        try:
            result = chat.chat(messages, tools=specs, max_tokens=max_tokens, temperature=temperature)
        except InferenceError as exc:
            error = str(exc)
            stopped_by = "error"
            transcript.append({"turn": len(transcript), "role": "model", "content": "", "error": error})
            break
        usage.add(result.usage)
        messages.append(result.message)
        if result.text:
            transcript.append({"turn": len(transcript), "role": "model", "content": result.text})
        if not result.tool_calls:
            stopped_by = "no_tool_call"
            break

        results: List[ToolResult] = []
        stop = False
        for call in result.tool_calls:
            if tool_calls >= max_tool_calls:
                # Every call needs a response for the conversation to stay valid.
                results.append(ToolResult(call.id, call.name, {"error": "Tool call limit reached; this call was not executed."}))
                continue
            tool_calls += 1
            output = _execute(registry, call)
            tool_result = ToolResult(call.id, call.name, output.content, list(output.attachments))
            results.append(tool_result)
            evictable.append((tool_result, output.on_evict))
            transcript.append({"turn": len(transcript), "role": "tool", "name": call.name, "args": call.arguments, "response": output.content})
            stop = stop or output.stop

        if stop:
            messages.append(Message.tool(results))
            stopped_by = "finish_tool"
            break
        messages.append(Message.tool(results, *([follow_up] if follow_up else [])))
        if is_done and is_done():
            stopped_by = "done"
            break

    return LoopResult(
        messages=messages,
        usage=usage,
        turns=turns,
        tool_calls=tool_calls,
        stopped_by=stopped_by,
        error=error,
        transcript=transcript,
    )


def _execute(registry: Dict[str, Tool], call: ToolCall) -> ToolOutput:
    tool = registry.get(call.name)
    if tool is None:
        return ToolOutput({"error": f"Unknown tool '{call.name}'. Available tools: {', '.join(registry)}"})
    if INVALID_ARGUMENTS_KEY in call.arguments:
        return ToolOutput({"error": "Tool arguments were not valid JSON. Call the tool again with a JSON object."})
    try:
        output = tool.handler(call.arguments)
    except Exception as exc:  # a failing tool should not end the agent run
        return ToolOutput({"error": f"{type(exc).__name__}: {exc}"})
    return output if isinstance(output, ToolOutput) else ToolOutput(dict(output))


def estimate_tokens(messages: Sequence[Message]) -> int:
    chars = 0
    images = 0
    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart):
                chars += len(part.text)
            elif isinstance(part, ImagePart):
                images += 1
        for call in message.tool_calls:
            chars += len(json.dumps(call.arguments, ensure_ascii=False, default=str))
        for result in message.tool_results:
            chars += len(result.content_json())
            for part in result.attachments:
                if isinstance(part, TextPart):
                    chars += len(part.text)
                elif isinstance(part, ImagePart):
                    images += 1
    return int(chars / CHARS_PER_TOKEN) + images * IMAGE_TOKEN_ESTIMATE


def _fit_context(
    chat: ChatModel,
    messages: List[Message],
    evictable: List[Tuple[ToolResult, Optional[Callable[[], None]]]],
    max_tokens: int,
) -> None:
    limit = chat.spec.context_tokens
    if not limit:
        return
    budget = limit - max_tokens - CONTEXT_MARGIN_TOKENS
    while evictable and estimate_tokens(messages) > budget:
        tool_result, on_evict = evictable.pop(0)
        tool_result.content = dict(EVICTED_CONTENT)
        tool_result.attachments = []
        if on_evict:
            on_evict()
