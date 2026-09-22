"""OpenAI-compatible adapters. OpenAI's API and local vLLM servers share this code."""
from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config.catalog import ModelSpec
from src.inference.base import ChatModel, Embedder
from src.inference.types import (
    ChatResult,
    ImagePart,
    InferenceError,
    Message,
    Part,
    PdfPart,
    TextPart,
    ToolCall,
    ToolSpec,
    Usage,
    strip_think,
)

INVALID_ARGUMENTS_KEY = "__invalid_arguments__"


class OpenAICompatChat(ChatModel):
    def __init__(self, key: str, spec: ModelSpec, client: Any, local: bool):
        super().__init__(key, spec)
        self.client = client
        self.local = local

    def _chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        tool_choice: str,
        response_schema: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        kwargs: Dict[str, Any] = {
            "model": self.spec.name,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
        }
        if self.spec.reasoning_effort:
            # reasoning models reject temperature != 1, and on Chat Completions they reject function tools unless
            # reasoning is off, so the effort comes from the catalog and no temperature is sent
            kwargs["reasoning_effort"] = self.spec.reasoning_effort
        else:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            kwargs["tool_choice"] = tool_choice
        if response_schema is not None and self.capabilities.json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": False},
            }
        extra: Dict[str, Any] = dict(self.spec.extra_body or {})
        if self.local and self.spec.thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.spec.thinking}
        if extra:
            kwargs["extra_body"] = extra

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise InferenceError(f"{self.key} ({self.spec.name}) request failed: {exc}") from exc
        if not getattr(response, "choices", None):
            raise InferenceError(f"{self.key}: response contained no choices")

        choice = response.choices[0]
        message = choice.message
        text = strip_think(message.content or "")
        calls: List[ToolCall] = []
        for tool_call in message.tool_calls or []:
            function = getattr(tool_call, "function", None)
            if function is None:
                continue
            calls.append(
                ToolCall(
                    id=tool_call.id or f"call_{uuid.uuid4().hex[:12]}",
                    name=function.name,
                    arguments=_parse_arguments(function.arguments),
                )
            )
        usage_data = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            api_calls=1,
            # vLLM reports this only with --enable-prompt-tokens-details
            cached_input_tokens=getattr(getattr(usage_data, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        )
        assistant = Message(role="assistant", parts=[TextPart(text)] if text else [], tool_calls=calls)
        return ChatResult(
            text=text,
            tool_calls=calls,
            usage=usage,
            message=assistant,
            model=self.key,
            finish_reason=choice.finish_reason,
        )


class OpenAICompatEmbedder(Embedder):
    batch_size = 32

    def __init__(self, key: str, spec: ModelSpec, client: Any):
        super().__init__(key, spec)
        self.client = client
        self.usage = Usage()

    def embed(self, texts: Sequence[str], kind: str = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefix = self.spec.query_instruction if kind == "query" else None
        inputs = [f"{prefix}{text}" if prefix else text for text in texts]
        vectors: List[List[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            try:
                response = self.client.embeddings.create(model=self.spec.name, input=batch,
                                                         **({"extra_body": dict(self.spec.extra_body)} if self.spec.extra_body else {}))
            except Exception as exc:
                raise InferenceError(f"{self.key} ({self.spec.name}) embedding request failed: {exc}") from exc
            data = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in data)
            usage = getattr(response, "usage", None)
            self.usage.add(Usage(input_tokens=getattr(usage, "prompt_tokens", 0) or 0, api_calls=1))
        return np.asarray(vectors, dtype=np.float32)


def to_openai_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            out.append({"role": "system", "content": message.text})
        elif message.role == "user":
            out.append({"role": "user", "content": _content(message.parts)})
        elif message.role == "assistant":
            entry: Dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                    }
                    for call in message.tool_calls
                ]
            out.append(entry)
        elif message.role == "tool":
            follow_up: List[Part] = []
            for result in message.tool_results:
                out.append({"role": "tool", "tool_call_id": result.call_id, "content": result.content_json()})
                follow_up.extend(result.attachments)
            # Tool messages carry text only; attachments (page images) and follow-up text go in a user turn.
            follow_up.extend(message.parts)
            if follow_up:
                out.append({"role": "user", "content": _content(follow_up)})
        else:
            raise InferenceError(f"Unknown message role '{message.role}'")
    return out


def _content(parts: List[Part]) -> Any:
    if all(isinstance(part, TextPart) for part in parts):
        return "\n\n".join(part.text for part in parts)
    items: List[Dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            items.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            encoded = base64.b64encode(part.data).decode()
            items.append({"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{encoded}"}})
        elif isinstance(part, PdfPart):
            encoded = base64.b64encode(part.data).decode()
            items.append(
                {"type": "file", "file": {"filename": part.filename, "file_data": f"data:application/pdf;base64,{encoded}"}}
            )
    return items


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {INVALID_ARGUMENTS_KEY: raw}
    return value if isinstance(value, dict) else {INVALID_ARGUMENTS_KEY: raw}
