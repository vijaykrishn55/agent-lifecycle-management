
# Topic 9: Agent Lifecycle Management

A modular Python project that makes AI agents easier to monitor, debug, and manage. It tracks what happens before and after each response, keeps count of token usage, prevents excessive usage, and handles errors safely.

## Features

- **Lifecycle hooks:** Run custom code before and after the agent responds, or when an error occurs.
- **Conversation history:** Maintain context across multiple turns.
- **Token tracking:** Track prompt tokens, response tokens, and cumulative usage.
- **Context tracing:** Record how the conversation changes during execution without logging raw prompts.
- **Structured logs:** Save execution details, timestamps, session IDs, response previews, and latency.
- **Usage limits:** Stop execution when configured token limits are exceeded.
- **Error handling:** Record failures and return a fallback response when appropriate.
- **Background logging:** Keep logging work separate from the main execution flow.
- **Automated tests:** Verify hook execution, token tracking, error handling, security checks, and performance.

## Quick Start

Run these commands in Windows PowerShell from the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python scripts/demo.py
```

Run the verification suite:

```powershell
python -m pytest -q
python scripts/verify.py
```

The default demo uses a mock model, so no API key or internet connection is required.

The verification script saves its results in the `reports/` folder.

## Optional Live Demo

You can also test the agent with the Groq GPT-OSS 20B model.

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-20b
LLM_TIMEOUT_SECONDS=45
```

Install the additional dependencies and run:

```powershell
python -m pip install -e ".[live,test]"
python -m pip install python-dotenv
python scripts/demo.py --live
```

Keep your API key private. Add `.env` to `.gitignore` before pushing the project to GitHub. Live requests may be subject to your provider's usage limits or charges.

## Project Structure

```text
src/agent_lifecycle/
  context.py          Conversation state and token tracking
  errors.py           Custom error types
  hooks.py            Hook registration and execution
  tokenizer.py        Token counting
  providers.py        Mock and live model providers
  builtin_hooks.py    Built-in monitoring and safety hooks
  telemetry.py        Background logging
  resilience.py       Failure protection
  runner.py           Main agent execution flow
  __init__.py         Public module exports

scripts/
  demo.py             Demonstration
  benchmark.py        Hook performance test
  verify.py           Automated verification

tests/                Automated tests
reports/              Verification and benchmark results
logs/                 Execution logs
```

The project is split into small, focused Python files to make it easier to understand, test, and maintain.

## Execution Flow

1. The user sends a prompt.
2. Pre-execution hooks inspect the conversation and check usage limits.
3. The agent calls the configured model.
4. Token usage and response details are recorded.
5. Post-execution hooks run and update the session state.
6. If something fails, error hooks record the failure and the agent handles it according to the configured policy.

## Verification

The project includes tests for:

- Hook registration and execution order.
- Token usage across multiple turns.
- Context and token budget enforcement.
- Error handling and fallback responses.
- Structured telemetry.
- Performance of hook dispatch.

Run `python -m pytest -q` to check the current test results.

## Known Limitations

- Token counting may be approximate when the tokenizer is unavailable.
- The performance benchmark measures hook overhead, not the time taken by live model requests.
- Logs are stored locally; a hosted monitoring dashboard is not included.
- The framework demonstrates production-oriented patterns but is not a complete security boundary.

## Submission

The repository contains the source code, automated tests, demo scripts, and verification reports. The final submission should also include a short Loom or YouTube walkthrough and a public GitHub repository.
