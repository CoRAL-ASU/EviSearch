from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI

from src.config.catalog import load_catalog
from src.inference.openai_compat import OpenAICompatChat, OpenAICompatEmbedder
from src.inference.types import ImagePart, InferenceError, Message, PdfPart, ToolCall, ToolResult, ToolSpec

CATALOG = load_catalog()
GET_PAGE = ToolSpec(
    name="get_page",
    description="Load pages",
    parameters={"type": "object", "properties": {"page_numbers": {"type": "array", "items": {"type": "integer"}}}, "required": ["page_numbers"]},
)


def _client(handler, requests):
    def capture(request: httpx.Request) -> httpx.Response:
        requests.append({"path": request.url.path, "body": json.loads(request.content or b"{}")})
        return handler(request)

    return OpenAI(
        base_url="http://vllm.test/v1",
        api_key="EMPTY",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(capture)),
    )


def _completion(message: dict, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "Qwen/Qwen3.6-27B",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        },
    )


def test_vllm_tool_call_request_and_parsing():
    requests = []
    reply = {
        "content": "<think>scan first pages</think>Loading page 1.",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_page", "arguments": "{\"page_numbers\": [1]}"}}],
    }
    chat = OpenAICompatChat("qwen3.6-27b", CATALOG.models["qwen3.6-27b"], _client(lambda r: _completion(reply, "tool_calls"), requests), local=True)

    result = chat.chat([Message.system("sys"), Message.user("find the trial name")], tools=[GET_PAGE], max_tokens=512)

    body = requests[0]["body"]
    assert requests[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "Qwen/Qwen3.6-27B"
    assert body["tools"][0]["function"]["name"] == "get_page"
    assert body["tool_choice"] == "auto"
    assert body["max_completion_tokens"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.text == "Loading page 1."
    assert result.tool_calls == [ToolCall(id="call_1", name="get_page", arguments={"page_numbers": [1]})]
    assert result.usage.to_dict() == {"input_tokens": 120, "output_tokens": 30, "api_calls": 1, "total_tokens": 150, "cached_input_tokens": 0}
    assert result.message.tool_calls == result.tool_calls


def test_conversation_layout_puts_attachments_after_tool_messages():
    requests = []
    chat = OpenAICompatChat("qwen3.6-27b", CATALOG.models["qwen3.6-27b"], _client(lambda r: _completion({"content": "{\"ok\": true}"}), requests), local=True)
    call = ToolCall(id="call_1", name="get_page", arguments={"page_numbers": [3]})
    history = [
        Message.system("sys"),
        Message.user("reconcile"),
        Message(role="assistant", tool_calls=[call]),
        Message.tool([ToolResult("call_1", "get_page", {"pages_returned": [3]}, [ImagePart(b"png-bytes")])], "Continue."),
    ]

    result = chat.chat(history, tools=[GET_PAGE], response_schema={"type": "object"})

    messages = requests[0]["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"page_numbers": [3]}
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "{\"pages_returned\": [3]}"}
    assert messages[4]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert messages[4]["content"][1] == {"type": "text", "text": "Continue."}
    assert requests[0]["body"]["response_format"]["type"] == "json_schema"
    assert result.json() == {"ok": True}


def test_openai_cloud_model_does_not_send_vllm_extras():
    requests = []
    chat = OpenAICompatChat("gpt-4.1", CATALOG.models["gpt-4.1"], _client(lambda r: _completion({"content": "hi"}), requests), local=False)
    chat.chat([Message.user("hello")])
    assert "chat_template_kwargs" not in requests[0]["body"]


def test_invalid_tool_arguments_are_flagged_not_raised():
    reply = {"content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_page", "arguments": "{not json"}}]}
    chat = OpenAICompatChat("qwen3.6-27b", CATALOG.models["qwen3.6-27b"], _client(lambda r: _completion(reply, "tool_calls"), []), local=True)
    result = chat.chat([Message.user("x")], tools=[GET_PAGE])
    assert "__invalid_arguments__" in result.tool_calls[0].arguments


def test_capability_and_http_errors_raise_inference_error():
    chat = OpenAICompatChat("qwen3.6-27b", CATALOG.models["qwen3.6-27b"], _client(lambda r: httpx.Response(500, json={"error": "boom"}), []), local=True)
    with pytest.raises(InferenceError, match="does not accept PDF"):
        chat.chat([Message.user("read", PdfPart(b"%PDF"))])
    with pytest.raises(InferenceError, match="request failed"):
        chat.chat([Message.user("hello")])


def test_embedder_adds_query_instruction_and_keeps_input_order():
    requests = []

    def handler(request):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "Qwen/Qwen3-Embedding-8B",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    embedder = OpenAICompatEmbedder("qwen3-embedding-8b", CATALOG.models["qwen3-embedding-8b"], _client(handler, requests))
    vectors = embedder.embed(["median OS", "arms"], kind="query")

    sent = requests[0]["body"]["input"]
    assert sent[0].startswith("Instruct: ") and sent[0].endswith("Query:median OS")
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.usage.input_tokens == 7

    embedder.embed(["page text"], kind="document")
    assert requests[1]["body"]["input"] == ["page text"]
