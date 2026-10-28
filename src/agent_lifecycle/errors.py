class AgentLifecycleError(Exception):
    """Base class for expected lifecycle failures."""


class ContextOverflowError(AgentLifecycleError):
    """Raised before provider invocation when a token limit would be exceeded."""


class TokenBudgetExceededError(ContextOverflowError):
    """Raised when a prompt or session token budget is exceeded."""


class ProviderRateLimitError(AgentLifecycleError):
    """Provider-compatible rate-limit failure used by adapters and tests."""


class CircuitOpenError(AgentLifecycleError):
    """Raised when the configured failure threshold opens the circuit breaker."""
