from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.estimated_cost_usd += other.estimated_cost_usd


class AgentContext(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    context_trace: list[dict[str, Any]] = Field(default_factory=list)
    turn_index: int = 0
    created_at: datetime = Field(default_factory=utc_now)
