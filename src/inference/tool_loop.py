"""Provider-neutral tool-calling loop shared by the search and reconciliation agents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.config.catalog import ImageTokens
from src.inference.base import ChatModel
from src.inference.openai_compat import INVALID_ARGUMENTS_KEY
from src.inference.types import ImagePart, InferenceError, Message, Part, TextPart, ToolCall, ToolResult, ToolSpec, Usage

CHARS_PER_TOKEN = 3.0  # conservative estimate for dense tables and numbers
IMAGE_TOKEN_ESTIMATE = 2000  # images whose size cannot be read
CONTEXT_MARGIN_TOKENS = 1024
EVICTED_CONTENT = {
    "evicted": True,
    "note": "This tool output was removed to fit the context window. Call the tool again if you still need it.",
}
FORCED_FINISH = (
    "You have no tool calls left. Call {tool} now with every value you have found so far; "
    'use "Not reported" for anything you could not find.'
)
# Ways a loop can end without the finish tool; with finish_tool set, the model then gets one forced turn.
UNFINISHED = ("max_tool_calls", "max_turns", "no_tool_call")


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
    stopped_by: str  # finish_tool | done | forced_finish | no_tool_call | max_turns | max_tool_calls | error
    error: Optional[str]
    transcript: List[Dict[str, Any]]
    calls: List[Dict[str, Any]] = field(default_factory=list)  # per model call: turn, start time, duration, tokens


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
    finish_tool: Optional[str] = None,
) -> LoopResult:
    """Let the model call tools until a handler stops the loop, is_done() is true, or a limit is hit.

    When the loop ends unfinished (tool budget or turns used up, or a reply without a tool call) and finish_tool is
    given, the model gets one more turn in which it must call that tool, so work done so far is submitted rather than
    lost; stopped_by is then "forced_finish".

    The full conversation is kept. When a model has a finite context window, the oldest tool outputs are
    replaced with a note (and their on_evict hooks run) so the next request still fits.
    """
    registry = {tool.spec.name: tool for tool in tools}
    specs = [tool.spec for tool in tools]
    messages: List[Message] = [Message.system(system), Message.user(user)]
    transcript: List[Dict[str, Any]] = [{"turn": 0, "role": "user", "content": user}]
    evictable: List[Tuple[ToolResult, Optional[Callable[[], None]]]] = []
    usage = Usage()
    calls: List[Dict[str, Any]] = []
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
        calls.append({"turn": turns, **result.call_record()})
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

    if finish_tool and finish_tool in registry and stopped_by in UNFINISHED and not (is_done and is_done()):
        instruction = FORCED_FINISH.format(tool=finish_tool)
        if messages[-1].role == "tool":
            messages[-1] = Message.tool(messages[-1].tool_results, instruction)  # replaces the usual follow-up
        else:
            messages.append(Message.user(instruction))
        transcript.append({"turn": len(transcript), "role": "user", "content": instruction})
        _fit_context(chat, messages, evictable, max_tokens)
        turns += 1
        try:
            result = chat.chat(messages, tools=[registry[finish_tool].spec], tool_choice="required", max_tokens=max_tokens, temperature=temperature)
        except InferenceError as exc:
            error = str(exc)
            transcript.append({"turn": len(transcript), "role": "model", "content": "", "error": error})
        else:
            usage.add(result.usage)
            calls.append({"turn": turns, **result.call_record()})
            messages.append(result.message)
            if result.text:
                transcript.append({"turn": len(transcript), "role": "model", "content": result.text})
            results = []
            for call in result.tool_calls:
                if call.name != finish_tool:
                    results.append(ToolResult(call.id, call.name, {"error": f"Only {finish_tool} can be called now."}))
                    continue
                tool_calls += 1
                output = _execute(registry, call)
                results.append(ToolResult(call.id, call.name, output.content, list(output.attachments)))
                transcript.append({"turn": len(transcript), "role": "tool", "name": call.name, "args": call.arguments, "response": output.content})
                stopped_by = "forced_finish"
            if results:
                messages.append(Message.tool(results))

    return LoopResult(
        messages=messages,
        usage=usage,
        turns=turns,
        tool_calls=tool_calls,
        stopped_by=stopped_by,
        error=error,
        transcript=transcript,
        calls=calls,
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


def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    """(width, height) from a PNG header; None for anything else."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def estimate_tokens(messages: Sequence[Message], image_tokens: Optional[ImageTokens] = None) -> int:
    """Upper estimate of a request's prompt tokens. PNG images cost what image_tokens (the model's catalog entry) says
    for their size; other images IMAGE_TOKEN_ESTIMATE."""
    chars = 0
    image_total = 0

    def add_image(part: ImagePart) -> int:
        size = _png_size(part.data) if image_tokens else None
        return image_tokens.count(*size) if size else IMAGE_TOKEN_ESTIMATE

    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart):
                chars += len(part.text)
            elif isinstance(part, ImagePart):
                image_total += add_image(part)
        for call in message.tool_calls:
            chars += len(json.dumps(call.arguments, ensure_ascii=False, default=str))
        for result in message.tool_results:
            chars += len(result.content_json())
            for part in result.attachments:
                if isinstance(part, TextPart):
                    chars += len(part.text)
                elif isinstance(part, ImagePart):
                    image_total += add_image(part)
    return int(chars / CHARS_PER_TOKEN) + image_total


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
    while evictable and estimate_tokens(messages, chat.spec.image_tokens) > budget:
        tool_result, on_evict = evictable.pop(0)
        tool_result.content = dict(EVICTED_CONTENT)
        tool_result.attachments = []
        if on_evict:
            on_evict()
