
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from agent_lifecycle import AgentContext, AgentRunner, JsonlTelemetry, MockProvider
from agent_lifecycle.providers import OpenAICompatibleProvider


def safe_error(message: str) -> str:
    """Avoid printing API keys if an exception contains one."""
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


async def run_demo(live: bool, turns: list[str]) -> int:
    telemetry = JsonlTelemetry(ROOT / "logs" / "telemetry.jsonl")

    if live:
        model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        base_url = os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        )
        provider = OpenAICompatibleProvider(
            model=model,
            base_url=base_url,
        )
    else:
        provider = MockProvider()

    agent = AgentRunner(provider=provider, telemetry=telemetry)
    context = AgentContext()

    print("Provider mode:", "LIVE" if live else "MOCK")
    if live:
        print("Model:", provider.model)
        print("Endpoint:", os.getenv(
            "LLM_BASE_URL", "https://api.groq.com/openai/v1"
        ))
        if not (
            os.getenv("GROQ_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ):
            print("ERROR: No API key found. Check your .env file.")

    try:
        for index, prompt in enumerate(turns, start=1):
            print(f"\n--- Turn {index} ---")
            previous_errors = len(context.errors)

            try:
                output, context = await agent.run(prompt, context)
                print("Assistant:", output)

                if context.metadata.get("execution_status") == "fallback":
                    new_errors = context.errors[previous_errors:]
                    for error in new_errors:
                        print(
                            "Underlying error:",
                            safe_error(
                                f"{error.get('type', 'Error')}: "
                                f"{error.get('message', 'No details recorded')}"
                            ),
                        )
            except Exception as error:
                print(
                    f"Execution blocked: {type(error).__name__}: "
                    f"{safe_error(str(error))}"
                )
                return 1

        await telemetry.flush()

        print("\nSession summary:")
        print(json.dumps({
            "session_id": context.session_id,
            "turns": context.turn_index,
            "prompt_tokens": context.token_usage.prompt_tokens,
            "completion_tokens": context.token_usage.completion_tokens,
            "total_tokens": context.token_usage.total_tokens,
            "estimated_cost_usd": round(
                context.token_usage.estimated_cost_usd, 8
            ),
            "errors": len(context.errors),
            "trace_snapshots": len(context.context_trace),
            "telemetry_file": "logs/telemetry.jsonl",
        }, indent=2))

        print("Registered hooks:", json.dumps({
            "pre": agent.hooks.names("pre"),
            "post": agent.hooks.names("post"),
            "error": agent.hooks.names("error"),
        }, indent=2))

        return 0 if not context.errors else 1
    finally:
        await agent.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent lifecycle and observability demo"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use the configured OpenAI-compatible API",
    )
    parser.add_argument(
        "--prompt", action="append",
        help="Prompt; repeat for a multi-turn demo",
    )
    args = parser.parse_args()

    turns = args.prompt or [
        "Explain lifecycle hooks in two short sentences.",
        "Explain how token usage is tracked across turns.",
    ]
    return asyncio.run(run_demo(args.live, turns))


if __name__ == "__main__":
    raise SystemExit(main())
