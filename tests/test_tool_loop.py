from __future__ import annotations

from src.config.catalog import Capabilities, ModelSpec, load_catalog
from src.inference.base import ChatModel
from src.inference.tool_loop import EVICTED_CONTENT, IMAGE_TOKEN_ESTIMATE, Tool, ToolOutput, estimate_tokens, run_tool_loop
from src.inference.types import ChatResult, ImagePart, InferenceError, Message, TextPart, ToolCall, ToolSpec, Usage


def _spec(context_tokens=None):
    return ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(tools=True, json_schema=True), context_tokens=context_tokens)


class ScriptedChat(ChatModel):
    """Returns queued turns; each turn is a list of (tool name, args) or a text string."""

    def __init__(self, turns, context_tokens=None):
        super().__init__("fake", _spec(context_tokens))
        self.turns = list(turns)
        self.seen = []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        self.seen.append(list(messages))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, str):
            calls = []
            text = turn
        else:
            calls = [ToolCall(id=f"c{len(self.seen)}_{i}", name=name, arguments=args) for i, (name, args) in enumerate(turn)]
            text = ""
        message = Message(role="assistant", parts=[TextPart(text)] if text else [], tool_calls=calls)
        return ChatResult(text=text, tool_calls=calls, usage=Usage(10, 2, 1), message=message, model="fake")


def _tool(name, handler):
    return Tool(ToolSpec(name, name, {"type": "object", "properties": {}}), handler)


def test_exhausted_tool_budget_forces_one_submit_with_only_the_finish_tool():
    submitted = {}
    tools = [
        _tool("lookup", lambda args: ToolOutput({"text": "Trial: STAMPEDE"})),
        _tool("submit", lambda args: (submitted.update(args), ToolOutput({"ok": True}, stop=True))[1]),
    ]
    offered = []

    class RecordingChat(ScriptedChat):
        def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
            offered.append(([t.name for t in tools], tool_choice))
            return super()._chat(messages, tools, tool_choice, response_schema, temperature, max_tokens)

    chat = RecordingChat([[("lookup", {})], [("lookup", {})], [("submit", {"Trial": "STAMPEDE"})]])
    result = run_tool_loop(chat, system="s", user="u", tools=tools, max_turns=5, max_tool_calls=2, max_tokens=100,
                           follow_up="Continue.", finish_tool="submit")

    assert result.stopped_by == "forced_finish" and submitted == {"Trial": "STAMPEDE"}
    assert offered[-1] == (["submit"], "required")
    assert "no tool calls left" in chat.seen[-1][-1].text and "Continue." not in chat.seen[-1][-1].text

    no_finish = run_tool_loop(ScriptedChat([[("lookup", {})], [("lookup", {})]]), system="s", user="u", tools=tools,
                              max_turns=5, max_tool_calls=2, max_tokens=100)
    assert no_finish.stopped_by == "max_tool_calls"

    replied_in_text = run_tool_loop(ScriptedChat(["I think it is STAMPEDE.", [("submit", {"Trial": "STAMPEDE"})]]), system="s",
                                    user="u", tools=tools, max_turns=5, max_tool_calls=5, max_tokens=100, finish_tool="submit")
    assert replied_in_text.stopped_by == "forced_finish"


def test_loop_runs_tools_until_finish_tool():
    submitted = {}
    tools = [
        _tool("lookup", lambda args: ToolOutput({"page": args["page"], "text": "Trial: STAMPEDE"})),
        _tool("submit", lambda args: (submitted.update(args), ToolOutput({"ok": True}, stop=True))[1]),
    ]
    chat = ScriptedChat([[("lookup", {"page": 1})], [("submit", {"Trial": "STAMPEDE"})]])

    result = run_tool_loop(chat, system="s", user="u", tools=tools, max_turns=5, max_tool_calls=5, max_tokens=100, follow_up="Continue.")

    assert result.stopped_by == "finish_tool"
    assert result.turns == 2 and result.tool_calls == 2
    assert submitted == {"Trial": "STAMPEDE"}
    assert result.usage.to_dict()["api_calls"] == 2
    # second request saw the whole history, including the tool result and follow-up
    second_request = chat.seen[1]
    assert [m.role for m in second_request] == ["system", "user", "assistant", "tool"]
    assert second_request[3].tool_results[0].content == {"page": 1, "text": "Trial: STAMPEDE"}
    assert second_request[3].text == "Continue."
    assert [entry["role"] for entry in result.transcript] == ["user", "tool", "tool"]


def test_unknown_tools_bad_arguments_and_handler_errors_are_returned_to_the_model():
    def explode(args):
        raise RuntimeError("index missing")

    chat = ScriptedChat([[("nope", {}), ("boom", {}), ("boom", {"__invalid_arguments__": "{"})], "giving up"])
    result = run_tool_loop(chat, system="s", user="u", tools=[_tool("boom", explode)], max_turns=5, max_tool_calls=10, max_tokens=100)

    errors = [r.content["error"] for r in chat.seen[1][3].tool_results]
    assert "Unknown tool 'nope'" in errors[0]
    assert "RuntimeError: index missing" in errors[1]
    assert "not valid JSON" in errors[2]
    assert result.stopped_by == "no_tool_call"


def test_tool_call_limit_still_answers_every_call():
    chat = ScriptedChat([[("t", {}), ("t", {}), ("t", {})]])
    result = run_tool_loop(chat, system="s", user="u", tools=[_tool("t", lambda a: ToolOutput({"ok": 1}))], max_turns=5, max_tool_calls=2, max_tokens=100)

    tool_message = result.messages[-1]
    assert len(tool_message.tool_results) == 3
    assert "limit reached" in tool_message.tool_results[2].content["error"]
    assert result.stopped_by == "max_tool_calls"


def test_old_tool_outputs_are_evicted_to_fit_context_and_hooks_run():
    evicted = []
    big_page = "x" * 3000  # ~1000 estimated tokens

    def load(args):
        page = args["page"]
        return ToolOutput({"page": page, "text": big_page}, on_evict=lambda: evicted.append(page))

    chat = ScriptedChat([[("load", {"page": 1})], [("load", {"page": 2})], [("load", {"page": 3})], "done"], context_tokens=3300)
    result = run_tool_loop(chat, system="s", user="u", tools=[_tool("load", load)], max_turns=10, max_tool_calls=10, max_tokens=500)

    assert result.stopped_by == "no_tool_call"
    assert evicted == [1, 2]
    last_request = chat.seen[-1]
    tool_results = [r for m in last_request for r in m.tool_results]
    assert [r.content for r in tool_results[:2]] == [EVICTED_CONTENT, EVICTED_CONTENT]
    assert tool_results[2].content["page"] == 3


def test_inference_error_and_is_done_stop_the_loop():
    failing = ScriptedChat([InferenceError("server down")])
    result = run_tool_loop(failing, system="s", user="u", tools=[], max_turns=3, max_tool_calls=3, max_tokens=10)
    assert result.stopped_by == "error" and result.error == "server down"

    done = {"flag": False}

    def mark(args):
        done["flag"] = True
        return ToolOutput({"ok": True})

    chat = ScriptedChat([[("mark", {})], "unreachable"])
    result = run_tool_loop(chat, system="s", user="u", tools=[_tool("mark", mark)], max_turns=5, max_tool_calls=5, max_tokens=10, is_done=lambda: done["flag"])
    assert result.stopped_by == "done" and result.turns == 1


def test_context_estimate_prices_page_images_for_the_model():
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (1224).to_bytes(4, "big") + (1584).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    page = [Message.user(ImagePart(png))]
    models = load_catalog().models
    assert estimate_tokens(page, models["mistral-small-3.2-24b"].image_tokens) == 55 * (43 + 1)
    assert estimate_tokens(page, models["qwen3.6-27b"].image_tokens) == 39 * 50
    assert estimate_tokens(page) == estimate_tokens([Message.user(ImagePart(b"jpeg"))], models["qwen3.6-27b"].image_tokens) == IMAGE_TOKEN_ESTIMATE


def test_every_model_call_is_timed_with_its_tokens_and_images():
    class Reader(ScriptedChat):
        def __init__(self, turns):
            super().__init__(turns)
            self.spec = ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(tools=True, images=True))

    page = _tool("page", lambda args: ToolOutput({"page": 1}, attachments=[ImagePart(b"png")]))
    submit = _tool("submit", lambda args: ToolOutput({"ok": True}, stop=True))
    result = run_tool_loop(Reader([[("page", {})], "no tool", [("submit", {})]]), system="s", user="u", tools=[page, submit],
                           max_turns=5, max_tool_calls=5, max_tokens=100, finish_tool="submit")

    assert result.stopped_by == "forced_finish"
    assert [call["turn"] for call in result.calls] == [1, 2, 3]  # the forced submit is recorded too
    assert [call["input_images"] for call in result.calls] == [0, 1, 1]
    assert all(call["started_at"] and call["duration_s"] >= 0 and call["input_tokens"] == 10 for call in result.calls)
    usage = result.usage.to_dict()
    assert usage["input_images"] == 2 and usage["api_calls"] == 3
    assert usage["model_seconds"] == round(sum(call["duration_s"] for call in result.calls), 3)
