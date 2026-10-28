import pytest

from agent_lifecycle import AgentContext, AgentRunner, JsonlTelemetry, MockProvider


@pytest.mark.asyncio
async def test_multi_turn_history_usage_and_hooks(tmp_path):
    provider = MockProvider(responses=["hello back", "answer two"])
    telemetry = JsonlTelemetry(tmp_path / "events.jsonl")
    runner = AgentRunner(provider=provider, telemetry=telemetry)
    seen = []
    runner.register_pre_hook(lambda context: seen.append(("pre", context.turn_index)), priority=30)
    runner.register_post_hook(lambda context, result: seen.append(("post", result)), priority=10)
    try:
        context = AgentContext()
        first, context = await runner.run("hello", context)
        second, context = await runner.run("follow up", context)
        await telemetry.flush()
        assert (first, second) == ("hello back", "answer two")
        assert len(context.messages) == 4
        assert seen == [("pre", 1), ("post", "hello back"), ("pre", 2), ("post", "answer two")]
        assert context.token_usage.total_tokens > 0
        assert len(context.context_trace) == 2
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_context_trace_does_not_log_raw_prompt(tmp_path):
    telemetry = JsonlTelemetry(tmp_path / "events.jsonl")
    runner = AgentRunner(telemetry=telemetry)
    secret = "unique-private-phrase-123"
    try:
        _, context = await runner.run(secret)
        await telemetry.flush()
        assert context.context_trace[0]["changed"] is True
        assert secret not in str(telemetry.events)
        assert telemetry.events[0]["session_id"] == context.session_id
        assert telemetry.events[0]["span_id"]
    finally:
        await runner.aclose()
