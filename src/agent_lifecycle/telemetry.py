
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlTelemetry:
    """Bounded async JSONL telemetry sink."""

    def __init__(
        self,
        path: str | Path | None = "logs/telemetry.jsonl",
        max_queue_size: int = 500,
    ):
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.path = Path(path) if path is not None else None
        self.queue: asyncio.Queue[dict[str, Any] | None] = (
            asyncio.Queue(maxsize=max_queue_size)
        )
        self.events: deque[dict[str, Any]] = deque(maxlen=max_queue_size)
        self.dropped_events = 0
        self.sink_errors = 0
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def emit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

        item = dict(event)
        item.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped_events += 1
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(item)

    async def _run(self) -> None:
        handle = None
        try:
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = self.path.open("a", encoding="utf-8")
                except OSError:
                    self.sink_errors += 1

            while True:
                event = await self.queue.get()
                try:
                    if event is None:
                        break

                    self.events.append(event)
                    # Use stdlib JSON directly; no structlog signature mismatch.
                    line = json.dumps(
                        event,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if handle is not None:
                        try:
                            handle.write(line + "\n")
                            handle.flush()
                        except (OSError, ValueError):
                            self.sink_errors += 1
                            handle.close()
                            handle = None
                except Exception:
                    # Telemetry must never crash agent execution.
                    self.sink_errors += 1
                finally:
                    self.queue.task_done()
        finally:
            if handle is not None:
                handle.close()

    async def flush(self) -> None:
        if self._worker is not None:
            await self.queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        await self.flush()
        if self._worker is not None:
            await self.queue.put(None)
            await self._worker
        self._closed = True


def telemetry_event(context, event: str, **fields: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": context.session_id,
        "span_id": context.metadata.get("current_span_id", "unknown"),
        "event": event,
        "turn_index": context.turn_index,
        **fields,
    }
