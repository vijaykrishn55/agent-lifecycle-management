import pytest

from agent_lifecycle import AgentContext, AgentRunner, ContextOverflowError, JsonlTelemetry, MockProvider


@pytest.mark.asyncio
async def test_prompt_budget_blocks_provider_before_call(tmp_path):
    provider = MockProvider()
    runner = AgentRunner(provider=provider, telemetry=JsonlTelemetry(tmp_path / "events.jsonl"),
                         max_prompt_tokens=5)
    try:
        with pytest.raises(ContextOverflowError):
            await runner.run("Write a paragraph about artificial intelligence.")
        assert provider.calls == 0
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_session_budget_counts_prior_usage(tmp_path):
    provider = MockProvider(responses=["first answer", "second answer"])
    runner = AgentRunner(provider=provider, telemetry=JsonlTelemetry(tmp_path / "events.jsonl"),
                         max_prompt_tokens=1000, max_total_tokens=1000)
    try:
        _, context = await runner.run("first turn")
        first_total = context.token_usage.total_tokens
        _, context = await runner.run("second turn", context)
        assert context.token_usage.total_tokens > first_total
        assert context.token_usage.prompt_tokens > 0
        assert context.token_usage.completion_tokens > 0
        assert context.token_usage.total_tokens == (
            context.token_usage.prompt_tokens + context.token_usage.completion_tokens
        )
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_cumulative_token_budget_blocks_next_provider_call(tmp_path):
    provider = MockProvider(responses=["okay"])
    runner = AgentRunner(provider=provider, telemetry=JsonlTelemetry(tmp_path / "events.jsonl"),
                         max_prompt_tokens=1000, max_total_tokens=30)
    try:
        _, context = await runner.run("first turn", AgentContext())
        assert provider.calls == 1
        with pytest.raises(ContextOverflowError, match="Token budget exceeded"):
            await runner.run("second turn that adds to the history", context)
        assert provider.calls == 1
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_pre_hook_context_mutation_is_traced_and_budgeted(tmp_path):
    provider = MockProvider()
    runner = AgentRunner(provider=provider, telemetry=JsonlTelemetry(tmp_path / "events.jsonl"),
                         max_prompt_tokens=25)

    async def inject_context(context):
        context.messages.append({"role": "system", "content": "context " * 100})

    runner.register_pre_hook(inject_context)
    try:
        with pytest.raises(ContextOverflowError):
            await runner.run("short prompt")
        assert provider.calls == 0
        assert runner.hooks.names("pre").index("inject_context") < runner.hooks.names("pre").index("ContextTraceHook")
    finally:
        await runner.aclose()
