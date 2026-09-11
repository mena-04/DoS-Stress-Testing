"""Notebook helpers: one managed gateway, real-backend checks, bounded waits.

REPO, MODEL, BACKEND_URL, GATEWAY_PORT and MAX_SEQS are set by the notebook.
The original gateway implementation is not modified.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import requests
import yaml

HTTP = requests.Session()
HTTP.trust_env = False
_gateway = None


def checked_json(url, **kwargs):
    response = HTTP.get(url, timeout=5, **kwargs)
    response.raise_for_status()
    return response.json()


def wait_ready(process, base_url, log_path, seconds=180, expected_mode=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Process exited ({process.returncode}).\n" +
                               Path(log_path).read_text(errors="replace")[-7000:])
        try:
            response = HTTP.get(base_url + "/health", timeout=2)
            if response.status_code == 200:
                if expected_mode is not None and response.json().get("mode") != expected_mode:
                    raise RuntimeError("Another gateway is answering on the selected port.")
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1)
    tail = Path(log_path).read_text(errors="replace")[-7000:] if Path(log_path).exists() else ""
    raise TimeoutError(f"Not ready after {seconds}s. Do not run the load test.\n{tail}")


def verify_real_backend():
    response = HTTP.get(BACKEND_URL + "/health", timeout=5)
    response.raise_for_status()
    # The repository's fake backend has no /version route.
    version = checked_json(BACKEND_URL + "/version")
    models = checked_json(BACKEND_URL + "/v1/models")
    ids = [m.get("id") for m in models.get("data", [])]
    if MODEL not in ids:
        raise RuntimeError(f"Expected model {MODEL}; found {ids}")
    return {"kind": "real_vllm", "version_response": version, "model": MODEL}


def wait_idle(seconds=120):
    from gateway.backpressure import parse_metric
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        response = HTTP.get(BACKEND_URL + "/metrics", timeout=5)
        response.raise_for_status()
        running = parse_metric(response.text, "vllm:num_requests_running")
        waiting = parse_metric(response.text, "vllm:num_requests_waiting")
        if running == 0 and waiting == 0:
            time.sleep(1)
            return
        time.sleep(1)
    raise RuntimeError("Backend still has work queued. No new experiment was started.")


def stop_gateway():
    global _gateway
    if _gateway is None or _gateway.poll() is not None:
        _gateway = None
        return
    _gateway.terminate()
    try:
        _gateway.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _gateway.kill()
        _gateway.wait(timeout=5)
        print("Gateway required force-stop; its final buffered records may be incomplete.")
    _gateway = None


def prepare_configs():
    paths = {}
    for mode, original in [("off", "off.yaml"), ("on", "ratelimit_queue.yaml")]:
        config = yaml.safe_load((REPO / "configs" / original).read_text())
        backpressure = config.setdefault("backpressure", {})
        backpressure.update(global_max_inflight=MAX_SEQS,
                            upstream_max_num_seqs=MAX_SEQS,
                            reserved_cheap_slots=2)
        config.setdefault("upstream", {})["base_url"] = BACKEND_URL
        # These copies preserve the other thresholds in the uploaded config.
        path = REPO / "configs" / f"demo_{mode}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths[mode] = path
    return paths


def run_locust_profile(profile, mode, pair_id):
    """Run one local profile, saving raw client + gateway logs in one folder."""
    global _gateway
    if mode not in {"off", "on"}:
        raise ValueError("mode must be off or on")
    stop_gateway()
    backend = verify_real_backend()
    wait_idle()
    # Refuse collisions rather than silently talking to an unrelated process.
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", GATEWAY_PORT)) == 0:
            raise RuntimeError(f"Port {GATEWAY_PORT} is occupied. Stop the older gateway first; "
                               "do not start several gateway copies.")
    run_id = f"{pair_id}-{profile}-{mode}"
    run_dir = REPO / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    configs = prepare_configs()
    target = f"http://127.0.0.1:{GATEWAY_PORT}"
    env = os.environ.copy()
    for key in ("GATEWAY_MODE", "GATEWAY_UPSTREAM", "GATEWAY_RUN_ID"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-u", "-m", "gateway", "--config", str(configs[mode]),
               "--upstream", BACKEND_URL, "--host", "127.0.0.1", "--port", str(GATEWAY_PORT),
               "--run-id", run_id, "--log-dir", str(REPO / "runs")]
    gateway_log = run_dir / "gateway_process.log"
    with gateway_log.open("w") as log:
        _gateway = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
    try:
        expected = "off" if mode == "off" else "ratelimit_queue"
        wait_ready(_gateway, target, gateway_log, expected_mode=expected)
        # A health response alone does not prove forwarding works.
        test = HTTP.post(target + "/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8, "temperature": 0,
        }, headers={"X-Client-ID": "warmup", "X-Traffic-Label": "warmup",
                    "X-Request-ID": uuid4().hex}, timeout=(5, 120))
        if test.status_code != 200 or not test.json().get("choices"):
            raise RuntimeError(f"Gateway forwarding failed: {test.status_code} {test.text[:300]}")
        wait_idle()
        metadata = {"run_id": run_id, "profile": profile, "mode": mode,
                    "backend": backend, "max_num_seqs": MAX_SEQS,
                    "gateway_port": GATEWAY_PORT, "backend_url": BACKEND_URL,
                    "note": "Same user schedule/seed, not identical arrival timestamps.",
                    "created_at": time.time()}
        (run_dir / "demo_manifest.json").write_text(json.dumps(metadata, indent=2))
        env.update(DOS_PROFILE=profile, DOS_RUN_ID=run_id, DOS_RUN_DIR=str(run_dir), DOS_SEED="42")
        cmd = [sys.executable, "-m", "locust", "-f", str(REPO / "loadgen" / "locustfile.py"),
               "--headless", "--host", target, "--csv", str(run_dir / "locust"),
               "--csv-full-history", "--stop-timeout", "125", "--exit-code-on-error", "0"]
        print(f"Running {profile} / {mode.upper()} -> {run_dir.name}", flush=True)
        locust_log = run_dir / "locust.log"
        with locust_log.open("w") as log:
            load = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 210
        try:
            while load.poll() is None:
                if _gateway.poll() is not None:
                    raise RuntimeError("Gateway stopped during the run.\n" +
                                       gateway_log.read_text(errors="replace")[-4000:])
                if time.monotonic() > deadline:
                    raise TimeoutError("Load test exceeded its bounded run time.")
                time.sleep(1)
        finally:
            if load.poll() is None:
                load.terminate()
                try:
                    load.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    load.kill()
                    load.wait(timeout=5)
        if load.returncode != 0:
            raise RuntimeError(locust_log.read_text(errors="replace")[-7000:])
        verify_real_backend()
        wait_idle()
        time.sleep(1)  # allow the gateway's buffered log flusher to finish
        print("Finished:", run_dir)
        return run_dir
    finally:
        stop_gateway()
