"""Reading list arguments that models pass to submit tools.

Tool-calling models sometimes pass a list argument as a JSON string instead of an array, and with long nested items
(evidence quotes inside attribution lists) that string can have unbalanced brackets or items whose opening brace is
missing. json.loads then fails and the whole submission would be lost. decode_items keeps every item that decodes on
its own.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

_KEY = re.compile(r'"(column|name)"\s*:')


def decode_items(value: Any) -> Tuple[Optional[Any], Optional[str]]:
    """(decoded value, note). Lists and dicts pass through; a JSON string is decoded; a malformed string yields the
    list of items that decode on their own (objects with a "column" or "name" key, also when their opening brace is
    missing). The note says what was done, or why nothing could be read (decoded value None)."""
    if value is None or isinstance(value, (list, dict)):
        return value, None
    if not isinstance(value, str):
        return None, f"expected a JSON array, got {type(value).__name__}"
    text = value.strip()
    try:
        return json.loads(text), "results arrived as a JSON string"
    except json.JSONDecodeError as exc:
        error = f"{exc.msg} at character {exc.pos}"
    decoder = json.JSONDecoder()
    items: List[Any] = []
    position = 0
    while True:
        match = _KEY.search(text, position)
        if not match:
            break
        brace = text.rfind("{", position, match.start())
        # A key right after "{" (only whitespace between) starts an item; otherwise the item lost its brace.
        starts_item = brace != -1 and not text[brace + 1:match.start()].strip()
        candidate, offset = (text[brace:], brace) if starts_item else ("{" + text[match.start():], match.start() - 1)
        try:
            item, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            position = match.end()
            continue
        if isinstance(item, dict):
            items.append(item)
        position = offset + end
    if not items:
        return None, f"results is not valid JSON ({error})"
    return items, f"results was malformed JSON ({error}); recovered {len(items)} item(s)"
