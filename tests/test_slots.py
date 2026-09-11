import asyncio

import pytest

from gateway.slots import AdmissionSlots, SlotPool


@pytest.mark.asyncio
async def test_pool_hands_out_capacity_then_blocks():
    pool = SlotPool(2)
    assert pool.try_acquire()
    assert pool.try_acquire()
    assert not pool.try_acquire()
    assert await pool.acquire(timeout=0.01) is False


@pytest.mark.asyncio
async def test_released_slot_wakes_a_waiter():
    pool = SlotPool(1)
    assert pool.try_acquire()
    waiter = asyncio.create_task(pool.acquire(timeout=1.0))
    await asyncio.sleep(0.01)
    pool.release()
    assert await waiter is True
    assert pool.held == 1


@pytest.mark.asyncio
async def test_timed_out_waiter_does_not_leak_capacity():
    pool = SlotPool(1)
    pool.try_acquire()
    assert await pool.acquire(timeout=0.01) is False
    pool.release()
    assert pool.free == 1
    assert pool.try_acquire()


@pytest.mark.asyncio
async def test_zero_capacity_pool_rejects_immediately():
    pool = SlotPool(0)
    assert await pool.acquire(timeout=1.0) is False


@pytest.mark.asyncio
async def test_expensive_requests_cannot_touch_the_reserved_pool():
    slots = AdmissionSlots(total=4, reserved_cheap=2, cheap_cost_threshold=100)
    # The two general slots absorb the first two expensive requests.
    assert await slots.acquire(cost=5000, timeout=0.01) == "general"
    assert await slots.acquire(cost=5000, timeout=0.01) == "general"
    # A third expensive request must not be allowed into the reserved pool.
    assert await slots.acquire(cost=5000, timeout=0.01) is None
    assert slots.reserved.held == 0


@pytest.mark.asyncio
async def test_cheap_requests_keep_flowing_while_general_pool_is_full():
    """The property that survives identity rotation.

    Capacity is partitioned by request cost, not by client, so it does not
    matter how many client IDs the expensive traffic arrives under.
    """
    slots = AdmissionSlots(total=4, reserved_cheap=2, cheap_cost_threshold=100)
    for _ in range(2):
        assert await slots.acquire(cost=5000, timeout=0.01) == "general"
    assert await slots.acquire(cost=50, timeout=0.01) == "reserved"
    assert await slots.acquire(cost=50, timeout=0.01) == "reserved"
    assert await slots.acquire(cost=50, timeout=0.01) is None


@pytest.mark.asyncio
async def test_cheap_requests_prefer_the_general_pool():
    """Reserved slots are the fallback, so they stay available for the case
    they exist for rather than being consumed while general capacity is idle."""
    slots = AdmissionSlots(total=4, reserved_cheap=2, cheap_cost_threshold=100)
    assert await slots.acquire(cost=10, timeout=0.01) == "general"
    assert slots.reserved.held == 0


@pytest.mark.asyncio
async def test_reserved_count_is_clamped_below_total():
    slots = AdmissionSlots(total=2, reserved_cheap=10, cheap_cost_threshold=100)
    assert slots.general.capacity >= 1
    assert slots.capacity == 2


def test_release_is_idempotent_at_zero():
    pool = SlotPool(1)
    pool.release()
    assert pool.held == 0
