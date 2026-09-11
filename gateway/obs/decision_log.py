"""Per-request decision log and gateway state sampler.

Two JSONL streams per run, both keyed on ``request_id`` so the analysis side
can join them to the load generator's own log:

    gateway.jsonl          one line per request, decision and outcome
    gateway_samples.jsonl  fixed-interval gateway + upstream state

Writes are buffered and flushed on a timer. Per-request flushing costs more
CPU than the admission logic it records, which matters when the generator,
the gateway and vLLM's frontend share two vCPUs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, TextIO


class JsonlWriter:
    def __init__(self, path: str, buffer_lines: int = 256) -> None:
        self.path = path
        self._buffer: list[str] = []
        self._buffer_lines = buffer_lines
        self._handle: TextIO | None = None
        self.written = 0

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._handle = open(self.path, "a", buffering=1024 * 64)

    def write(self, record: dict[str, Any]) -> None:
        self._buffer.append(json.dumps(record, separators=(",", ":"), default=str))
        self.written += 1
        if len(self._buffer) >= self._buffer_lines:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._handle is None:
            return
        self._handle.write("\n".join(self._buffer) + "\n")
        self._handle.flush()
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class DecisionLog:
    def __init__(self, config, run_dir: str | None = None) -> None:
        self._config = config
        self.run_dir = run_dir or os.path.join(config.dir, config.run_id)
        self.decisions = JsonlWriter(
            os.path.join(self.run_dir, config.decisions_filename), config.buffer_lines
        )
        self.samples = JsonlWriter(
            os.path.join(self.run_dir, config.samples_filename), 16
        )
        self._task: asyncio.Task | None = None

    def open(self) -> None:
        self.decisions.open()
        self.samples.open()

    def write_config(self, payload: dict[str, Any]) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "gateway_config.json"), "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def record(self, record: dict[str, Any]) -> None:
        self.decisions.write(record)

    def sample(self, record: dict[str, Any]) -> None:
        self.samples.write(record)

    async def start_flusher(self) -> None:
        interval = self._config.flush_interval_ms / 1000.0

        async def loop() -> None:
            while True:
                await asyncio.sleep(interval)
                self.decisions.flush()
                self.samples.flush()

        self._task = asyncio.create_task(loop(), name="decision-log-flusher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.decisions.close()
        self.samples.close()


class StateSampler:
    """Writes gateway and upstream state on a fixed interval.

    Records the upstream snapshot's staleness alongside its values, so a run
    where ``/metrics`` went dark under load is visible in the data rather than
    indistinguishable from an idle backend.
    """

    def __init__(self, log: DecisionLog, controller, poller, config) -> None:
        self._log = log
        self._controller = controller
        self._poller = poller
        self._config = config
        self._task: asyncio.Task | None = None

    def snapshot(self) -> dict[str, Any]:
        report = self._controller.pressure.evaluate()
        snapshot = report.snapshot
        return {
            "ts": time.time(),
            "gateway_inflight": self._controller.inflight,
            "slots_general_held": self._controller.slots.general.held,
            "slots_reserved_held": self._controller.slots.reserved.held,
            "pressure_level": int(report.level),
            "pressure_triggers": list(report.triggers),
            "latency_ewma_ms": round(self._controller.pressure.latency_ewma_ms, 2),
            "active_clients": self._controller.limiters.active_count(),
            "upstream_running": snapshot.running,
            "upstream_waiting": snapshot.waiting,
            "upstream_waiting_capacity": snapshot.waiting_capacity,
            "upstream_stale": report.upstream_stale,
            "upstream_age_ms": round(self._poller.age_ms(), 1)
            if self._poller is not None
            else None,
            "metrics_polls": getattr(self._poller, "poll_count", None),
            "metrics_failures": getattr(self._poller, "failure_count", None),
            "signals": {k: round(v, 4) for k, v in report.signals.items()},
        }

    async def start(self) -> None:
        interval = self._config.sample_interval_ms / 1000.0

        async def loop() -> None:
            while True:
                self._log.sample(self.snapshot())
                await asyncio.sleep(interval)

        self._task = asyncio.create_task(loop(), name="state-sampler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
