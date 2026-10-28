from __future__ import annotations

import hashlib
import traceback
from typing import Any

from .context import AgentContext
from .errors import ContextOverflowError, TokenBudgetExceededError
from .telemetry import JsonlTelemetry, telemetry_event
from .tokenizer import count_messages_tokens


def _snapshot(context: AgentContext) -> dict[str, Any]:
    serialized = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in context.messages)
    return {
        "message_count": len(context.messages),
        "roles": [m.get("role", "unknown") for m in context.messages],
        "content_characters": len(serialized),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


class ContextTraceHook:
    def __init__(self, telemetry: JsonlTelemetry):
        self.telemetry = telemetry

    async def __call__(self, context: AgentContext) -> None:
        current = _snapshot(context)
        previous = context.metadata.get("last_context_snapshot")
        trace = {
            "before_hash": previous.get("sha256") if previous else None,
            "after_hash": current["sha256"],
            "changed": previous is None or previous.get("sha256") != current["sha256"],
            "message_count": current["message_count"],
            "roles": current["roles"],
            "content_characters": current["content_characters"],
            "estimated_tokens": count_messages_tokens(context.messages),
        }
        context.metadata["estimated_prompt_tokens"] = trace["estimated_tokens"]
        context.context_trace.append(trace)
        context.metadata["last_context_snapshot"] = current
        await self.telemetry.emit(telemetry_event(context, "context_trace", hook_type="pre",
                                                   context_state=trace))


class TokenBudgetGuard:
    def __init__(self, max_prompt_tokens: int = 4000, max_total_tokens: int = 12000):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_total_tokens = max_total_tokens

    async def __call__(self, context: AgentContext) -> None:
        estimated = int(context.metadata.get("estimated_prompt_tokens", 0))
        if estimated > self.max_prompt_tokens:
            raise ContextOverflowError(
                f"Prompt context overflow: estimated {estimated} tokens, limit {self.max_prompt_tokens}."
            )
        if context.token_usage.total_tokens + estimated > self.max_total_tokens:
            raise TokenBudgetExceededError(
                "Token budget exceeded: cumulative usage plus the next prompt would be "
                f"{context.token_usage.total_tokens + estimated}, limit {self.max_total_tokens}."
            )


class PreTelemetryHook:
    def __init__(self, telemetry: JsonlTelemetry):
        self.telemetry = telemetry

    async def __call__(self, context: AgentContext) -> None:
        await self.telemetry.emit(telemetry_event(
            context, "agent_pre_execution", hook_type="pre",
            estimated_prompt_tokens=context.metadata.get("estimated_prompt_tokens", 0),
            context_state=context.context_trace[-1] if context.context_trace else {},
        ))


def _safe_preview(context: AgentContext, result: str) -> str:
    preview = result[:120]
    for message in context.messages:
        if message.get("role") == "user":
            user_text = message.get("content", "")
            if user_text and user_text in preview:
                return "[redacted response containing user content]"
    return preview


class PostTelemetryHook:
    def __init__(self, telemetry: JsonlTelemetry):
        self.telemetry = telemetry

    async def __call__(self, context: AgentContext, result: str) -> None:
        latency_ms = float(context.metadata.get("latency_ms", 0.0))
        await self.telemetry.emit(telemetry_event(
            context, "agent_post_execution", hook_type="post",
            status=context.metadata.get("execution_status", "success"),
            prompt_tokens=context.token_usage.prompt_tokens,
            completion_tokens=context.token_usage.completion_tokens,
            total_tokens=context.token_usage.total_tokens,
            estimated_cost_usd=context.token_usage.estimated_cost_usd,
            latency_ms=round(latency_ms, 3), response_preview=_safe_preview(context, result),
        ))


class ErrorTelemetryHook:
    def __init__(self, telemetry: JsonlTelemetry):
        self.telemetry = telemetry

    async def __call__(self, context: AgentContext, error: Exception) -> None:
        def redact(value: str) -> str:
            for message in context.messages:
                user_text = message.get("content", "") if message.get("role") == "user" else ""
                if user_text:
                    value = value.replace(user_text, "[redacted user content]")
            return value

        error_message = redact(str(error))
        stack_trace = redact("".join(traceback.format_exception(
            type(error), error, error.__traceback__
        )))
        detail = {"type": type(error).__name__, "message": error_message}
        context.errors.append(detail)
        await self.telemetry.emit(telemetry_event(
            context, "agent_error", hook_type="error", level="ERROR",
            error_type=detail["type"], error_message=detail["message"],
            stack_trace=stack_trace[-8000:],
            accumulated_tokens=context.token_usage.total_tokens,
        ))
