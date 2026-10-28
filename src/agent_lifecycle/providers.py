
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Protocol

from .context import TokenUsage
from .errors import ProviderRateLimitError
from .tokenizer import count_messages_tokens, count_text_tokens

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class ProviderResponse:
    text: str
    usage: TokenUsage
    raw: object | None = None


class CompletionProvider(Protocol):
    async def complete(
        self, messages: list[dict[str, str]]
    ) -> ProviderResponse: ...


@dataclass
class MockProvider:
    """Deterministic provider for offline tests."""
    responses: list[str] = field(default_factory=list)
    fail_next: Exception | None = None
    calls: int = 0

    async def complete(
        self, messages: list[dict[str, str]]
    ) -> ProviderResponse:
        self.calls += 1
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error

        prompt = messages[-1].get("content", "") if messages else ""
        text = self.responses.pop(0) if self.responses else (
            f"Demo response: {prompt[:160]}" if prompt else "Demo response."
        )
        prompt_tokens = count_messages_tokens(messages)
        completion_tokens = count_text_tokens(text)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ProviderResponse(text=text, usage=usage)


@dataclass
class OpenAICompatibleProvider:
    """Async provider for Groq and other OpenAI-compatible APIs."""
    model: str = "openai/gpt-oss-20b"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 45.0

    async def complete(
        self, messages: list[dict[str, str]]
    ) -> ProviderResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                'Install live dependencies: python -m pip install -e ".[live,test]"'
            ) from exc

        api_key = (
            self.api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Set GROQ_API_KEY or LLM_API_KEY in your environment or .env file."
            )

        base_url = self.base_url or os.getenv("LLM_BASE_URL")
        if not base_url and os.getenv("GROQ_API_KEY"):
            base_url = "https://api.groq.com/openai/v1"

        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", self.timeout_seconds))
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", self.model),
                    messages=messages,
                    temperature=self.temperature,
                ),
                timeout=timeout + 5,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "429" in message:
                raise ProviderRateLimitError(str(exc)) from exc
            if isinstance(exc, asyncio.TimeoutError):
                raise TimeoutError(
                    f"LLM request exceeded {timeout:g} seconds."
                ) from exc
            raise
        finally:
            await client.close()

        text = response.choices[0].message.content or ""
        raw_usage = response.usage
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(
            getattr(raw_usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(raw_usage, "total_tokens", prompt_tokens + completion_tokens)
            or 0
        )
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return ProviderResponse(text=text, usage=usage, raw=response)
