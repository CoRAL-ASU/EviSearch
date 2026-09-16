from __future__ import annotations

import pytest
from google.genai import types

from src.config.catalog import load_catalog
from src.evaluation.evaluator_v2 import EvaluationResults
from src.inference.gemini import SYNTHETIC_ID_PREFIX, GeminiChat
from src.inference.types import ImagePart, InferenceError, Message, ToolResult, ToolSpec

CATALOG = load_catalog()
GET_PAGE = ToolSpec(
    name="get_page",
    description="Load pages",
    parameters={"type": "object", "properties": {"page_numbers": {"type": "array", "items": {"type": "integer"}}}, "required": ["page_numbers"]},
)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def _response(parts, **usage):
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}],
            "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 20, **usage},
        }
    )


def test_tool_turn_config_parsing_and_thought_signature_replay():
    first = _response(
        [
            {"text": "planning", "thought": True},
            {"function_call": {"name": "get_page", "args": {"page_numbers": [2]}}, "thought_signature": "c2ln"},
        ],
        thoughts_token_count=5,
        cached_content_token_count=40,
    )
    second = _response([{"text": "done"}])
    client = FakeClient([first, second])
    chat = GeminiChat("gemini-2.5-flash", CATALOG.models["gemini-2.5-flash"], client)

    result = chat.chat([Message.system("be precise"), Message.user("reconcile")], tools=[GET_PAGE], max_tokens=1000)

    config = client.models.calls[0]["config"]
    assert config.system_instruction == "be precise"
    assert config.max_output_tokens == 1000
    assert config.thinking_config.thinking_budget == 0  # catalog: thinking false, like the local models
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "get_page"
    assert declaration.parameters.properties["page_numbers"].items.type == types.Type.INTEGER
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO
    assert result.text == ""  # thought text is not returned
    assert result.tool_calls[0].name == "get_page"
    assert result.tool_calls[0].id.startswith(SYNTHETIC_ID_PREFIX)
    assert result.usage.output_tokens == 25  # candidates + thoughts
    assert result.usage.cached_input_tokens == 40

    history = [
        Message.system("be precise"),
        Message.user("reconcile"),
        result.message,
        Message.tool([ToolResult(result.tool_calls[0].id, "get_page", {"pages_returned": [2]}, [ImagePart(b"png")])], "Continue."),
    ]
    chat.chat(history, tools=[GET_PAGE])

    contents = client.models.calls[1]["contents"]
    assert contents[1] is result.message.provider_state  # replayed verbatim, signature intact
    assert contents[1].parts[1].thought_signature == b"sig"
    tool_turn = contents[2]
    assert tool_turn.role == "user"
    assert tool_turn.parts[0].function_response.name == "get_page"
    assert tool_turn.parts[0].function_response.id is None  # synthetic ids are never sent
    assert tool_turn.parts[1].inline_data.mime_type == "image/png"
    assert tool_turn.parts[2].text == "Continue."


def test_response_schema_config_accepts_pydantic_json_schema():
    client = FakeClient([_response([{"text": "{\"results\": []}"}])])
    chat = GeminiChat("gemini-2.5-flash", CATALOG.models["gemini-2.5-flash"], client)

    result = chat.chat([Message.user("score")], response_schema=EvaluationResults.model_json_schema())

    config = client.models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None
    assert result.json() == {"results": []}


def test_no_candidates_raises():
    empty = types.GenerateContentResponse.model_validate({"candidates": [], "prompt_feedback": {"block_reason": "SAFETY"}})
    chat = GeminiChat("gemini-2.5-flash", CATALOG.models["gemini-2.5-flash"], FakeClient([empty]))
    with pytest.raises(InferenceError, match="no candidates"):
        chat.chat([Message.user("x")])
