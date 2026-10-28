from __future__ import annotations

from .context import AgentContext
from .errors import CircuitOpenError


class CircuitBreaker:
    """Session-scoped consecutive-failure circuit breaker with manual reset."""

    def __init__(self, failure_threshold: int = 3) -> None:
        self.failure_threshold = max(1, failure_threshold)

    def check(self, context: AgentContext) -> None:
        failures = int(context.metadata.get("consecutive_failures", 0))
        if failures >= self.failure_threshold:
            raise CircuitOpenError(
                f"Circuit open after {failures} consecutive execution failures."
            )

    def record_failure(self, context: AgentContext) -> None:
        context.metadata["consecutive_failures"] = int(
            context.metadata.get("consecutive_failures", 0)
        ) + 1

    def record_success(self, context: AgentContext) -> None:
        context.metadata["consecutive_failures"] = 0

    def reset(self, context: AgentContext) -> None:
        context.metadata["consecutive_failures"] = 0
