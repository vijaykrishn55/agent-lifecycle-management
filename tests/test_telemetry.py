import json

import pytest

from agent_lifecycle import AgentContext, JsonlTelemetry
from agent_lifecycle.telemetry import telemetry_event


@pytest.mark.asyncio
async def test_jsonl_events_have_timestamp_session_and_span(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    sink = JsonlTelemetry(path)
    context = AgentContext()
    context.metadata["current_span_id"] = "span-test"
    await sink.emit(telemetry_event(context, "sample", hook_type="pre"))
    await sink.close()
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert event["session_id"] == context.session_id
    assert event["span_id"] == "span-test"
    assert event["event"] == "sample"
    assert event["timestamp"].endswith("+00:00")


@pytest.mark.asyncio
async def test_bounded_queue_drops_oldest_on_overflow():
    sink = JsonlTelemetry(path=None, max_queue_size=2)
    for index in range(5):
        await sink.emit({"event": "item", "index": index})
    await sink.close()
    assert sink.dropped_events >= 1
    assert sink.events[-1]["index"] == 4
