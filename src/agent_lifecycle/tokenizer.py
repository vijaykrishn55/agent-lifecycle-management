from __future__ import annotations

import math
import re
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_text_tokens(text: str, model: str | None = None) -> int:
    """Count tokens exactly with tiktoken when available; otherwise estimate safely."""
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Character-based fallback avoids requiring a download for local verification.
        return max(1, math.ceil(len(text) / 4)) if text else 0


def count_messages_tokens(messages: Iterable[dict[str, str]], model: str | None = None) -> int:
    total = 0
    for message in messages:
        total += 4
        total += count_text_tokens(message.get("role", ""), model)
        total += count_text_tokens(message.get("content", ""), model)
    return total + 2 if total else 0
