"""Admission slot pools.

A plain semaphore is FIFO, which is the wrong discipline here: a cheap
request that arrives behind a wall of expensive ones waits for an expensive
one to finish, however small it is.

Capacity is therefore split into two pools. Expensive requests may only use
the general pool; cheap requests may use either, trying the general pool
first. A small reserved pool means low-cost traffic always has somewhere to
go, no matter how much expensive traffic is queued.

The key property is that this partitions on *request cost*, not on client
identity, so an attacker rotating client IDs gains nothing from it. Per-client
limits dilute under rotation — with eight identities each looks almost fair —
and this is the mechanism that does not.
"""

from __future__ import annotations

import asyncio
from collections import deque


class SlotPool:
    """A counting pool with FIFO waiters and a bounded wait."""

    def __init__(self, capacity: int, name: str = "") -> None:
        self.capacity = max(0, capacity)
        self.name = name
        self.held = 0
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.held)

    def try_acquire(self) -> bool:
        if self.held < self.capacity:
            self.held += 1
            return True
        return False

    async def acquire(self, timeout: float) -> bool:
        if self.try_acquire():
            return True
        if self.capacity == 0:
            return False
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            try:
                self._waiters.remove(waiter)
            except ValueError:
                # Handed a slot while timing out; give it straight back
                # rather than leaking capacity.
                if waiter.done() and not waiter.cancelled():
                    self.release()
            return False

    def release(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(True)
                return
        self.held = max(0, self.held - 1)


class AdmissionSlots:
    def __init__(self, total: int, reserved_cheap: int, cheap_cost_threshold: int) -> None:
        reserved = max(0, min(reserved_cheap, max(0, total - 1)))
        self.cheap_cost_threshold = cheap_cost_threshold
        self.general = SlotPool(total - reserved, "general")
        self.reserved = SlotPool(reserved, "reserved")

    @property
    def held(self) -> int:
        return self.general.held + self.reserved.held

    @property
    def capacity(self) -> int:
        return self.general.capacity + self.reserved.capacity

    def is_cheap(self, cost: int) -> bool:
        return cost <= self.cheap_cost_threshold

    async def acquire(self, cost: int, timeout: float) -> str | None:
        """Return the pool name a slot was taken from, or None on timeout."""
        if self.general.try_acquire():
            return "general"
        if self.is_cheap(cost) and self.reserved.try_acquire():
            return "reserved"
        if await self.general.acquire(timeout):
            return "general"
        return None

    def release(self, pool: str) -> None:
        (self.reserved if pool == "reserved" else self.general).release()
