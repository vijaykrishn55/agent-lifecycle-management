import pytest

from agent_lifecycle import AgentContext, AgentRunner, JsonlTelemetry, MockProvider, ProviderRateLimitError


@pytest.mark.asyncio
async def test_provider_error_is_logged_and_fallback_is_returned(tmp_path):
    provider = MockProvider(fail_next=ProviderRateLimitError("rate limited"))
    telemetry = JsonlTelemetry(tmp_path / "events.jsonl")
    runner = AgentRunner(provider=provider, telemetry=telemetry)
    try:
        output, context = await runner.run("please answer")
        await telemetry.flush()
        assert output == runner.fallback_response
        assert context.errors[0]["type"] == "ProviderRateLimitError"
        event = next(item for item in telemetry.events if item["event"] == "agent_error")
        post_event = next(item for item in telemetry.events if item["event"] == "agent_post_execution")
        assert event["level"] == "ERROR"
        assert event["error_type"] == "ProviderRateLimitError"
        assert "ProviderRateLimitError" in event["stack_trace"]
        assert post_event["status"] == "fallback"
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_circuit_breaker_stops_repeated_provider_calls(tmp_path):
    provider = MockProvider(fail_next=RuntimeError("temporary outage"))
    runner = AgentRunner(provider=provider, telemetry=JsonlTelemetry(tmp_path / "events.jsonl"),
                         failure_threshold=1)
    try:
        context = AgentContext()
        await runner.run("first", context)
        await runner.run("second", context)
        assert provider.calls == 1
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_post_hook_failure_is_captured_without_losing_response(tmp_path):
    telemetry = JsonlTelemetry(tmp_path / "events.jsonl")
    runner = AgentRunner(telemetry=telemetry)

    async def broken_post(context, result):
        raise RuntimeError("post hook failed")

    runner.register_post_hook(broken_post)
    try:
        output, context = await runner.run("normal prompt")
        await telemetry.flush()
        assert output.startswith("Demo response:")
        assert context.errors[0]["type"] == "RuntimeError"
        assert any(item["event"] == "agent_error" for item in telemetry.events)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_error_telemetry_redacts_user_prompt(tmp_path):
    secret = "private-error-payload-456"
    provider = MockProvider(fail_next=RuntimeError(secret))
    telemetry = JsonlTelemetry(tmp_path / "events.jsonl")
    runner = AgentRunner(provider=provider, telemetry=telemetry)
    try:
        await runner.run(secret)
        await telemetry.flush()
        event = next(item for item in telemetry.events if item["event"] == "agent_error")
        assert secret not in event["error_message"]
        assert secret not in event["stack_trace"]
    finally:
        await runner.aclose()
