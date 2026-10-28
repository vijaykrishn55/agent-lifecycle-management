from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from .builtin_hooks import (ContextTraceHook, ErrorTelemetryHook, PostTelemetryHook,
                            PreTelemetryHook, TokenBudgetGuard)
from .context import AgentContext, TokenUsage
from .errors import ContextOverflowError
from .hooks import Hook, HookRegistry
from .providers import CompletionProvider, MockProvider
from .resilience import CircuitBreaker
from .telemetry import JsonlTelemetry
from .tokenizer import count_messages_tokens


class AgentRunner:
    """Single-agent runner instrumented with ordered lifecycle hooks and guardrails."""

    def __init__(self, provider: CompletionProvider | None = None,
                 telemetry: JsonlTelemetry | None = None,
                 max_prompt_tokens: int = 4000, max_total_tokens: int = 12000,
                 failure_threshold: int = 3, fallback_response: str | None = None,
                 prompt_cost_per_million: float = 0.0,
                 completion_cost_per_million: float = 0.0):
        self.provider = provider or MockProvider()
        self.telemetry = telemetry or JsonlTelemetry(Path("logs/telemetry.jsonl"))
        self.fallback_response = fallback_response or (
            "I'm sorry, I couldn't complete that request right now. Please try again."
        )
        self.prompt_cost_per_million = prompt_cost_per_million
        self.completion_cost_per_million = completion_cost_per_million
        self.circuit_breaker = CircuitBreaker(failure_threshold)
        self.hooks = HookRegistry()
        self.hooks.register_pre(ContextTraceHook(self.telemetry), priority=90)
        self.hooks.register_pre(TokenBudgetGuard(max_prompt_tokens, max_total_tokens), priority=100)
        self.hooks.register_pre(PreTelemetryHook(self.telemetry), priority=110)
        self.hooks.register_post(PostTelemetryHook(self.telemetry), priority=100)
        self.hooks.register_error(ErrorTelemetryHook(self.telemetry), priority=100)

    def register_pre_hook(self, hook: Hook, priority: int = 50, name: str | None = None) -> None:
        """Register context preparation before tracing and budget enforcement by default."""
        self.hooks.register_pre(hook, priority, name)

    def register_post_hook(self, hook: Hook, priority: int = 50, name: str | None = None) -> None:
        self.hooks.register_post(hook, priority, name)

    def register_error_hook(self, hook: Hook, priority: int = 50, name: str | None = None) -> None:
        self.hooks.register_error(hook, priority, name)

    async def run(self, prompt: str, context: AgentContext | None = None) -> tuple[str, AgentContext]:
        context = context or AgentContext()
        context.turn_index += 1
        context.metadata["current_span_id"] = str(uuid4())
        context.messages.append({"role": "user", "content": prompt})
        context.metadata["estimated_prompt_tokens"] = count_messages_tokens(context.messages)
        started = time.perf_counter()
        try:
            self.circuit_breaker.check(context)
            await self.hooks.dispatch_pre(context)
            response = await self.provider.complete(context.messages)
            if response.usage.total_tokens == 0:
                response.usage.prompt_tokens = int(context.metadata["estimated_prompt_tokens"])
                from .tokenizer import count_text_tokens
                response.usage.completion_tokens = count_text_tokens(response.text)
                response.usage.total_tokens = (response.usage.prompt_tokens +
                                               response.usage.completion_tokens)
            cost = (
                response.usage.prompt_tokens * self.prompt_cost_per_million
                + response.usage.completion_tokens * self.completion_cost_per_million
            ) / 1_000_000
            response.usage.estimated_cost_usd = cost
            context.token_usage.add(response.usage)
            context.messages.append({"role": "assistant", "content": response.text})
            self.circuit_breaker.record_success(context)
            context.metadata["execution_status"] = "success"
            context.metadata["latency_ms"] = (time.perf_counter() - started) * 1000
            await self.hooks.dispatch_post(context, response.text)
            return response.text, context
        except Exception as error:
            context.metadata["latency_ms"] = (time.perf_counter() - started) * 1000
            context.metadata["execution_status"] = "error"
            if not isinstance(error, ContextOverflowError):
                self.circuit_breaker.record_failure(context)
            await self.hooks.dispatch_error(context, error)
            await self.telemetry.flush()
            if isinstance(error, ContextOverflowError):
                raise
            context.metadata["execution_status"] = "fallback"
            context.messages.append({"role": "assistant", "content": self.fallback_response})
            await self.hooks.dispatch_post(context, self.fallback_response)
            return self.fallback_response, context

    async def aclose(self) -> None:
        await self.telemetry.close()
