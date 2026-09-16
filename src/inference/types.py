"""Provider-neutral message, tool and result types shared by every model adapter."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class InferenceError(RuntimeError):
    """A model call failed: network, API, authentication or unsupported input."""


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime_type: str = "image/png"


@dataclass(frozen=True)
class PdfPart:
    data: bytes
    filename: str = "document.pdf"


Part = Union[TextPart, ImagePart, PdfPart]


def _as_parts(content: tuple) -> List[Part]:
    return [TextPart(item) if isinstance(item, str) else item for item in content]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema of the arguments object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: Dict[str, Any]
    attachments: List[Part] = field(default_factory=list)

    def content_json(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, default=str)


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    parts: List[Part] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    # Adapter-native assistant content, replayed verbatim (keeps Gemini thought signatures intact).
    provider_state: Any = None

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role="system", parts=[TextPart(text)])

    @classmethod
    def user(cls, *content: Union[str, Part]) -> "Message":
        return cls(role="user", parts=_as_parts(content))

    @classmethod
    def tool(cls, results: List[ToolResult], *follow_up: Union[str, Part]) -> "Message":
        """Tool results for the previous assistant turn, optionally followed by more user content."""
        return cls(role="tool", tool_results=list(results), parts=_as_parts(follow_up))

    @property
    def text(self) -> str:
        return "\n\n".join(part.text for part in self.parts if isinstance(part, TextPart))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cached_input_tokens: int = 0  # part of input_tokens served from the provider's prompt cache

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.api_calls += other.api_calls
        self.cached_input_tokens += other.cached_input_tokens
        return self

    def to_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
        }


@dataclass
class ChatResult:
    text: str
    tool_calls: List[ToolCall]
    usage: Usage
    message: Message  # assistant message to append to the conversation
    model: str  # catalog model key
    finish_reason: Optional[str] = None

    def json(self) -> Any:
        return parse_json_text(self.text)


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def parse_json_text(text: str) -> Any:
    """Parse JSON from model text, tolerating <think> blocks, code fences and surrounding prose."""
    raw = strip_think(text)
    if not raw:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON found in response: {raw[:200]!r}")
