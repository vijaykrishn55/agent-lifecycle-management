import pytest

from agent_lifecycle import AgentContext, HookRegistry


@pytest.mark.asyncio
async def test_hooks_run_by_priority_then_registration_order():
    registry, seen = HookRegistry(), []

    def sync_hook(context):
        seen.append("sync")

    async def async_hook(context):
        seen.append("async")

    registry.register_pre(sync_hook, priority=10)
    registry.register_pre(async_hook, priority=1)
    registry.register_pre(lambda context: seen.append("last"), priority=10)
    await registry.dispatch_pre(AgentContext())
    assert seen == ["async", "sync", "last"]


@pytest.mark.asyncio
async def test_error_hook_failure_does_not_mask_original_error():
    registry = HookRegistry()

    async def broken(context, error):
        raise RuntimeError("hook broke")

    registry.register_error(broken)
    await registry.dispatch_error(AgentContext(), ValueError("original"))
    assert registry.dispatch_errors[0]["error"] == "RuntimeError: hook broke"
