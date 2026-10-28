from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .context import AgentContext

Hook = Callable[..., Any | Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredHook:
    name: str
    callback: Hook
    priority: int
    sequence: int


class HookRegistry:
    """Async hook dispatcher; callbacks run by priority, then registration order."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[RegisteredHook]] = {
            "pre": [], "post": [], "error": []
        }
        self._sequence = 0
        self.dispatch_errors: list[dict[str, str]] = []

    def _register(self, phase: str, callback: Hook, priority: int, name: str | None) -> None:
        self._sequence += 1
        label = name or getattr(callback, "__name__", callback.__class__.__name__)
        self._hooks[phase].append(RegisteredHook(label, callback, priority, self._sequence))

    def register_pre(self, callback: Hook, priority: int = 100, name: str | None = None) -> None:
        self._register("pre", callback, priority, name)

    def register_post(self, callback: Hook, priority: int = 100, name: str | None = None) -> None:
        self._register("post", callback, priority, name)

    def register_error(self, callback: Hook, priority: int = 100, name: str | None = None) -> None:
        self._register("error", callback, priority, name)

    def names(self, phase: str) -> list[str]:
        return [hook.name for hook in self._ordered(phase)]

    def _ordered(self, phase: str) -> list[RegisteredHook]:
        return sorted(self._hooks[phase], key=lambda hook: (hook.priority, hook.sequence))

    async def _call(self, hook: RegisteredHook, *args: Any) -> Any:
        result = hook.callback(*args)
        return await result if inspect.isawaitable(result) else result

    async def dispatch_pre(self, context: AgentContext) -> None:
        for hook in self._ordered("pre"):
            await self._call(hook, context)

    async def dispatch_post(self, context: AgentContext, result: str) -> None:
        for hook in self._ordered("post"):
            try:
                await self._call(hook, context, result)
            except Exception as hook_error:
                self.dispatch_errors.append({
                    "hook": hook.name,
                    "error": f"{type(hook_error).__name__}: {hook_error}",
                })
                await self.dispatch_error(context, hook_error)

    async def dispatch_error(self, context: AgentContext, error: Exception) -> None:
        for hook in self._ordered("error"):
            try:
                await self._call(hook, context, error)
            except Exception as hook_error:
                self.dispatch_errors.append({
                    "hook": hook.name, "error": f"{type(hook_error).__name__}: {hook_error}"
                })
