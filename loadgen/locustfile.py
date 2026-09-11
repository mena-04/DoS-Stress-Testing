"""Four local-only profiles restored from the earlier runs.

"""
import itertools
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

from locust import HttpUser, LoadTestShape, between, events, task

PROFILE = os.environ.get("DOS_PROFILE", "normal")
RUN_ID = os.environ.get("DOS_RUN_ID", "demo")
RUN_DIR = Path(os.environ.get("DOS_RUN_DIR", "results/locust-demo"))
SEED = int(os.environ.get("DOS_SEED", "42"))
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TIMELINES = {
    "normal": {"end": 30, "attack_start": None, "attack_end": None, "total_users": 4},
    "spike": {"end": 30, "attack_start": 10, "attack_end": 20, "total_users": 24},
    "flood": {"end": 50, "attack_start": 10, "attack_end": 40, "total_users": 32},
    "low_slow": {"end": 50, "attack_start": 10, "attack_end": 40, "total_users": 8},
}
if PROFILE not in TIMELINES:
    raise ValueError(f"Unknown profile: {PROFILE}")
TIMELINE = TIMELINES[PROFILE]
SMALL = "Explain AI inference briefly."
MEDIUM = ("Explain how an AI inference server handles requests, batching, "
          "token generation, and resource usage. ") * 10
EXPENSIVE = ("Explain in detail the architecture of an AI inference server, "
             "including request scheduling, KV cache, GPU execution, batching, "
             "prefill, decode, memory usage, and latency. ") * 30

_started = None
_attempt_log = None
_result_log = None
_ids = {"legit": itertools.count(1), "attacker": itertools.count(1)}


def phase_at(elapsed):
    if PROFILE == "normal":
        return "normal"
    if elapsed < TIMELINE["attack_start"]:
        return "pre_attack"
    if elapsed < TIMELINE["attack_end"]:
        return "attack"
    return "recovery"


@events.test_start.add_listener
def start_logging(environment, **kwargs):
    global _started, _attempt_log, _result_log
    host = urlparse(environment.host or "")
    if host.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("This demo is restricted to your local test server.")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    _started = time.time()
    _attempt_log = (RUN_DIR / "client_attempts.jsonl").open("w", buffering=65536)
    _result_log = (RUN_DIR / "client_requests.jsonl").open("w", buffering=65536)
    (RUN_DIR / "load_profile.json").write_text(json.dumps({
        "profile": PROFILE, "run_id": RUN_ID, "start_ts": _started,
        "seed": SEED, "model": MODEL, "host": environment.host,
        "generator": "Locust closed-loop virtual users", "legitimate_users": 4,
        **TIMELINE,
    }, indent=2))


@events.test_stop.add_listener
def close_logging(environment, **kwargs):
    for handle in (_attempt_log, _result_log):
        if handle is not None and not handle.closed:
            handle.close()


@events.request.add_listener
def record_request(name, response_time, response=None, context=None,
                   exception=None, **kwargs):
    if not context or not context.get("request_id") or _result_log is None:
        return
    status = getattr(response, "status_code", 0) or 0
    usage = {}
    valid = False
    if 200 <= status < 300:
        try:
            body = response.json()
            usage = body.get("usage", {})
            valid = bool(body.get("choices")) and exception is None
        except (ValueError, AttributeError):
            pass
    headers = getattr(response, "headers", {})
    row = dict(context)
    row.update({
        "end_ts": time.time(), "status": status,
        "latency_ms": response_time, "valid_response": valid,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "gateway_reason": headers.get("X-Gateway-Reason", ""),
        "error": str(exception) if exception else "",
    })
    _result_log.write(json.dumps(row) + "\n")


class BaseTraffic(HttpUser):
    abstract = True
    traffic_label = "legit"

    def on_start(self):
        self.client_id = f"{self.traffic_label}-{next(_ids[self.traffic_label])}"
        self.sequence = 0
        self.client.trust_env = False

    def send_inference(self, prompt, max_tokens):
        self.sequence += 1
        now = time.time()
        elapsed = now - (_started or now)
        request_id = f"{RUN_ID}-{self.client_id}-{self.sequence}"
        # The nonce is deterministic per user/sequence, so paired runs use the
        # same prompt pattern. It is a documented change from the old tests.
        prompt = f"Request {self.client_id}-{self.sequence}. " + prompt
        context = {
            "request_id": request_id, "run_id": RUN_ID, "profile": PROFILE,
            "client_id": self.client_id, "traffic_label": self.traffic_label,
            "start_ts": now, "elapsed_s": elapsed, "phase": phase_at(elapsed),
            "max_tokens": max_tokens,
        }
        _attempt_log.write(json.dumps(context) + "\n")
        # No application-level retries; 429 and 503 remain real failed requests
        # in Locust, and their reasons are recorded separately for analysis.
        with self.client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0},
            headers={"X-Client-ID": self.client_id, "X-Request-ID": request_id,
                     "X-Traffic-Label": self.traffic_label},
            context=context, name=self.traffic_label,
            timeout=(3, 120), catch_response=True,
        ) as response:
            if 200 <= response.status_code < 300:
                try:
                    if not response.json().get("choices"):
                        response.failure("2xx without a completion")
                except ValueError:
                    response.failure("2xx with invalid JSON")


class LegitimateUser(BaseTraffic):
    abstract = False
    fixed_count = 4
    traffic_label = "legit"
    wait_time = between(0.5, 1.5)

    @task
    def legitimate(self):
        self.send_inference(SMALL, 32)


class AttackerUser(BaseTraffic):
    abstract = False
    weight = 1
    traffic_label = "attacker"
    wait_time = between(1.5, 2.5) if PROFILE == "low_slow" else between(0.05, 0.15)

    @task
    def attacker(self):
        if PROFILE == "low_slow":
            self.send_inference(EXPENSIVE, 256)
        else:
            self.send_inference(MEDIUM, 128)


class DemoShape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        if t >= TIMELINE["end"]:
            return None
        if (PROFILE == "normal" or t < TIMELINE["attack_start"]
                or t >= TIMELINE["attack_end"]):
            return (4, 20, [LegitimateUser])
        return (TIMELINE["total_users"], 20, [LegitimateUser, AttackerUser])
