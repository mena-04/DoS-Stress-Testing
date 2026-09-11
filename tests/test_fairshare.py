from gateway.config import FairShareConfig
from gateway.fairshare import FairShareController
from gateway.features import FEATURE_NAMES, FeatureStore


def _store(clock):
    return FeatureStore(window_s=10.0, clock=lambda: clock[0])


def test_dominant_client_is_over_share_and_quiet_one_is_not():
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(), store)

    for _ in range(20):
        store.record_admitted("attacker", 1000, 1024, prompt_hash=1)
    store.record_admitted("legit", 40, 64, prompt_hash=2)

    assert controller.evaluate("attacker").over_share is True
    assert controller.evaluate("legit").over_share is False


def test_evenly_matched_clients_are_not_shed():
    """Strict inequality: clients splitting capacity evenly all sit at the
    threshold, and singling one out would be arbitrary."""
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(), store)
    for _ in range(10):
        store.record_admitted("a", 100, 100, prompt_hash=1)
        store.record_admitted("b", 100, 100, prompt_hash=2)
    assert controller.evaluate("a").over_share is False
    assert controller.evaluate("b").over_share is False


def test_single_client_is_never_over_share():
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(), store)
    for _ in range(50):
        store.record_admitted("only", 1000, 1024, prompt_hash=1)
    assert controller.evaluate("only").over_share is False


def test_shares_decay_out_of_the_window():
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(), store)
    for _ in range(20):
        store.record_admitted("attacker", 1000, 1024, prompt_hash=1)
    clock[0] = 5.0
    store.record_admitted("legit", 40, 64, prompt_hash=2)
    assert controller.evaluate("attacker").over_share is True

    # Past the window the old cost no longer counts against the client.
    clock[0] = 20.0
    store.record_admitted("attacker", 40, 64, prompt_hash=3)
    store.record_admitted("legit", 40, 64, prompt_hash=4)
    assert controller.evaluate("attacker").over_share is False


def test_long_running_requests_keep_being_charged():
    """A generation slower than the window must not become invisible.

    Cost is recorded at admission. Once it ages out of the window the client
    reads as idle even though it is still holding slots, and the steady cheap
    client is then measured as consuming the whole window and gets shed --
    which is precisely backwards. This is the low-and-slow profile.
    """
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(), store)

    running = []
    for i in range(8):
        store.window("attacker").concurrency += 1
        running.append(store.record_admitted("attacker", 1000, 1024, prompt_hash=i))

    # Past the 10s window, with every attacker request still in flight.
    clock[0] = 11.0
    store.record_admitted("legit", 11, 32, prompt_hash=99)

    assert controller.evaluate("attacker").over_share is True
    assert controller.evaluate("legit").over_share is False

    # The charge is released on completion, not held as a lasting penalty.
    clock[0] = 40.0
    for event in running:
        store.complete(event, latency_ms=36_000.0, failed=False)
        store.window("attacker").concurrency -= 1
    store.record_admitted("legit", 11, 32, prompt_hash=100)
    assert controller.evaluate("attacker").over_share is False
    assert store.window("attacker").expired_open_cost == 0.0


def test_disabled_controller_never_sheds():
    clock = [0.0]
    store = _store(clock)
    controller = FairShareController(FairShareConfig(enabled=False), store)
    for _ in range(20):
        store.record_admitted("attacker", 1000, 1024, prompt_hash=1)
    store.record_admitted("legit", 10, 10, prompt_hash=2)
    assert controller.evaluate("attacker").over_share is False


def test_feature_vector_layout_is_stable_and_finite():
    clock = [0.0]
    store = _store(clock)
    for i in range(5):
        clock[0] = i * 0.5
        event = store.record_admitted("c", 100, 128, prompt_hash=7)
        store.complete(event, latency_ms=250.0, failed=False)
    vector = store.vector("c")
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vector)


def test_repeat_ratio_separates_identical_from_varied_prompts():
    clock = [0.0]
    store = _store(clock)
    for _ in range(10):
        store.record_admitted("same", 100, 128, prompt_hash=1)
    for i in range(10):
        store.record_admitted("varied", 100, 128, prompt_hash=i)
    index = FEATURE_NAMES.index("repeat_ratio")
    assert store.vector("same")[index] > 0.8
    assert store.vector("varied")[index] == 0.0


def test_empty_window_yields_zero_vector():
    clock = [0.0]
    store = _store(clock)
    assert store.vector("unseen") == [0.0] * len(FEATURE_NAMES)
