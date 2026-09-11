# DoS Stress-Testing & Auto-Mitigation for AI Inference Servers

Menna; Salama; Rawan

Mitigation gateway that sits between the load generator and a vLLM server
running `Qwen/Qwen2.5-0.5B-Instruct`.

```
load generator  ->  mitigation gateway (:8080)  ->  vLLM (:8000)
                          |                              |
                    gateway.jsonl                  vllm.log / metrics
                    gateway_samples.jsonl
```

Two required mechanisms, plus an optional third:

| Mechanism | Rejects with | Reason codes |
|---|---|---|
| Per-client dual token bucket (rate + token cost) | `429` | `rate_limit_requests`, `rate_limit_cost`, `client_concurrency`, `cost_exceeds_budget` |
| Queue-pressure shedding with fair-share ranking | `503` | `queue_pressure`, `queue_timeout`, `global_capacity` |
| Cost-partitioned admission with a reserved cheap lane | `503` | `queue_timeout` |
| VAE anomaly-adaptive tightening (optional) | `429` | `anomaly_shed` |

## Quick start

```bash
pip install -e ".[dev]"

# No GPU needed: a calibrated vLLM stand-in
python -m gateway.fake_vllm --port 8000 --max-num-seqs 16

python -m gateway --config configs/ratelimit_queue.yaml --run-id demo-01
curl -s localhost:8080/gateway/state | python -m json.tool
```

In Colab, launch it the same way vLLM is launched:

```python
gw = subprocess.Popen([
    "python", "-m", "gateway",
    "--config", "configs/ratelimit_queue.yaml",
    "--upstream", "http://127.0.0.1:8000",
    "--port", "8080",
    "--run-id", RUN_ID,
], stdout=open("/content/gateway.log", "w"), stderr=subprocess.STDOUT)
```

## Two rules that protect the result

**The gateway never reads the traffic label.** `X-Traffic-Label` is logged for
the analysis join and is invisible to admission control. Branching on ground
truth would make "attacker rejection rate" a measurement of the label rather
than of the defense.

**Mitigation-off still goes through the gateway.** `configs/off.yaml` is a
pure passthrough that still writes `gateway.jsonl`. If the baseline run
bypassed the gateway, every latency delta would include the proxy hop — which
is not free when the generator, the gateway and vLLM's frontend share two
vCPUs on a Colab box.

## Mitigation profiles

The four configs are the ablation, selectable with one flag:

| Config | Rate limit | Queue cap | Fair share | VAE |
|---|---|---|---|---|
| `configs/off.yaml` | — | — | — | — |
| `configs/ratelimit.yaml` | yes | — | — | — |
| `configs/ratelimit_queue.yaml` | yes | yes | yes | — |
| `configs/full.yaml` | yes | yes | yes | yes |

## For the load generator (Person 1)

Point `--target` at `http://127.0.0.1:8080`. Nothing else has to change: when
`X-Client-ID` is absent the gateway falls back to the peer address, so an
unmodified client works. Three headers make the evidence much stronger:

| Header | Purpose |
|---|---|
| `X-Client-ID` | the per-client identity every limit is keyed on |
| `X-Request-ID` | join key; echoed on every response and logged |
| `X-Traffic-Label` | `legit` / `attacker`, logged only |

Please also:

- **always set `max_tokens` explicitly.** An omitted value is charged
  `default_max_tokens`, because a request that generates to the context limit
  is not cheap and must not be charged zero.
- **never retry a `429` or `503`.** A retrying client turns rejections into
  amplification and makes the rejection rate meaningless.
- **record the real status code.** Collapsing everything to `"error"` erases
  the difference between a timeout, a `429` and a `503`.

## For the analysis side (Person 2)

Each run writes `runs/<run_id>/`:

- `gateway_config.json` — the full resolved config, so a run is reproducible
- `gateway.jsonl` — one record per request
- `gateway_samples.jsonl` — gateway and upstream state every 250 ms

Fields in `gateway.jsonl` that map onto the required metrics:

| Field | Use |
|---|---|
| `request_id` | join to the load generator's own log |
| `client_id`, `traffic_label` | split legitimate from attacker |
| `decision`, `reason`, `http_status` | success / error / rejection rates |
| `gateway_latency_ms`, `queue_wait_ms` | p50 / p95, and how much is queueing |
| `prompt_tokens_est`, `max_tokens`, `cost` | requests/sec versus cost/sec |
| `upstream_waiting`, `upstream_running` | queue depth over time |
| `upstream_stale` | whether that queue reading is trustworthy |
| `pressure_level`, `pressure_triggers` | which signal caused shedding |
| `cost_share`, `share_threshold` | why a specific client was shed |
| `tier`, `anomaly_score` | VAE score over time |

`upstream_stale` matters. In the first real overload run, vLLM's `/metrics`
stopped responding under load and the sampler silently dropped every sample,
so 180 consecutive samples recorded `num_requests_running = 0` while 100
requests were in flight. Any queue-depth chart needs to distinguish "the
queue was empty" from "we could not see the queue".

`/metrics` on the gateway exposes the same counters in Prometheus format
(`gateway_requests_total`, `gateway_rejections_total{reason}`,
`gateway_queue_wait_seconds`, `gateway_pressure_level`, and so on) on a
private registry that does not shadow vLLM's `vllm:*` series.

## Calibration: the thresholds are placeholders

Every number in the configs is a guess until the backend's capacity is
measured. The first real run on a T4 showed why this cannot be skipped: 100
concurrent expensive requests produced p50 5.603 s, p95 5.700 s, max 5.701 s
with 100/100 succeeding, and no queue ever formed. A 100x concurrency
increase cost about 23% more per-token latency, so the server was far from
its knee, and `num_requests_waiting` stayed at 0 because a 0.5B model with
~11 GB of KV cache admits every request into one batch.

Three consequences:

1. **Pin `--max-num-seqs`** (16–32) when launching vLLM and record it. This
   gives the backend a documented admission width and makes queue depth a
   number that actually moves. Set both `upstream_max_num_seqs` and
   `global_max_inflight` to that value. Setting `global_max_inflight` much
   higher is worse than useless: a smoke run with 48 gateway slots against a
   backend admitting 8 held pressure at level 0 for the whole run, because
   the queue formed inside vLLM where the gateway can neither see it in time
   nor reorder it, and legitimate p95 reached 4.0 s with nothing rejected.
2. **Disable prefix caching** for headline runs and randomise a prompt prefix
   per request. vLLM enables prefix caching by default, so 100 identical
   prompts mean one prefill and 99 cache hits — the "expensive" prompt is not
   expensive.
3. **Sweep concurrency** (1, 2, 4, 8, 16, 32, 64, 128) and find the knee of
   the p95 curve. That single measurement sets `global_max_inflight`, the
   shedding water marks, and the per-client cost budget.

Because batching is so efficient, the scarce resource is sequence-slot
occupancy rather than request rate. This is why the cost bucket and the
per-client in-flight cap do the real work: a client issuing
`max_num_seqs` requests at `max_tokens=1024` locks the server for tens of
seconds at a negligible request rate, which no request-rate limit can see.

## VAE anomaly detection (optional layer)

Trained on normal traffic only, so reconstruction error is a novelty signal
rather than a learned attack signature.

```bash
# 1. baseline run with feature logging on
python -m gateway --config configs/off.yaml --run-id normal-1 --log-features

# 2. train, and read the printed holdout percentiles
python -m gateway.anomaly.train runs/normal-1/gateway.jsonl -o models/vae.npz

# 3. set suspect/hostile thresholds above the baseline p99, then
python -m gateway --config configs/full.yaml --run-id mitigated-vae
```

Design notes:

- 12 rolling per-client features (`gateway/features.py`), whose order is the
  model's ABI — a model trained on a different layout is rejected at load.
- Scoring runs in a background task every 200 ms and is cached per client.
  The request path only reads a dict entry, because a mitigation layer that
  adds tail latency under load defeats its own purpose.
- The forward pass is numpy and the latent mean is used instead of a sample,
  so an identical client scores identically on consecutive ticks.
- Tiers (`normal` / `suspect` / `hostile`) scale the client's bucket rates and
  decay over `tier_decay_s`, so a client that bursts once recovers.
- Everything fails open. A missing, unreadable or stale-layout model leaves
  every client in the `normal` tier and the static limits carry the defense.

Honest framing for the report: with four known traffic profiles, the VAE does
not beat a well-tuned cost limiter at detection. Its value is that it needs
only normal traffic to train and flags shapes nobody hand-coded. Present it
as adaptive tightening with an ablation, not as the primary defense.

## Identity rotation and the reserved lane

Every per-client mechanism dilutes under identity rotation. With eight
attacker IDs, each one presents a 0.125 cost share against a 0.111 equal
share, so it looks almost fair: the token buckets see eight modest clients
and fair-share ranking finds nobody above threshold. A smoke run reproduced
exactly this — zero rejections, and legitimate p95 at 4.0 s.

The mechanism that holds up is `gateway/slots.py`, which partitions admission
capacity by *request cost* instead of by identity. Expensive requests may
only use the general pool; cheap requests try the general pool first and fall
back to a small reserved pool. Low-cost traffic therefore always has
somewhere to go, however many identities the expensive traffic arrives under,
and rotating IDs buys an attacker nothing against it.

A plain semaphore would not do this. It is FIFO, so a cheap request arriving
behind a wall of expensive ones waits for an expensive one to finish.

Calibration matters here: `cheap_cost_threshold` has to sit above the
legitimate profile's typical `prompt_tokens + max_tokens` and below the
attacker's. Set it from the cost distribution of a baseline run. If it is too
high the reserved lane admits the attack; too low and legitimate requests
never reach it.

## Known limitations

- **Identity rotation still defeats the per-client limits** themselves, as
  above. The reserved lane keeps cheap traffic flowing, but an attacker whose
  requests are individually cheap and numerous, spread across many IDs, is
  constrained only by `global_max_inflight`. Handling that properly needs a
  secondary identity key and a new-client penalty budget.
- **Cost is estimated before generation.** `max_tokens` is charged in full
  even when the model stops early, so the limiter over-charges short replies.
  This is deliberate: charging actual usage would let a client spend budget it
  has not been granted yet.
- **`max_tokens` is clamped, not rejected.** A request above
  `max_tokens_ceiling` is silently reduced so the backend cannot be asked for
  more work than was charged. Clients see a shorter completion than requested.

## Development

```bash
python -m pytest tests/ -q
```

The suite runs without a GPU against `gateway/fake_vllm.py`, a vLLM stand-in
calibrated to the measured T4 numbers (~35 ms/token at batch 1 rising to
~44 ms/token at batch 100). It models `--max-num-seqs` admission so a real
queue forms, optional prefix caching, and — with `--metrics-stall-ms` — a
`/metrics` endpoint that stops answering under load, which is how the
staleness fail-safe is tested.
