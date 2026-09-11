"""Train the VAE on a normal-traffic run.

Reads ``gateway.jsonl`` from one or more baseline runs and fits on the
per-request feature vectors the gateway recorded. Only normal traffic should
be supplied; training on a run that contains an attack teaches the model that
the attack is ordinary.

    python -m gateway.anomaly.train runs/normal-1/gateway.jsonl -o models/vae.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..features import FEATURE_NAMES
from .vae import recon_error_batch, standardize, train


def load_vectors(paths: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                vector = record.get("features")
                if isinstance(vector, list) and len(vector) == len(FEATURE_NAMES):
                    rows.append([float(v) for v in vector])
    if not rows:
        raise SystemExit(
            "no feature vectors found; run the gateway with "
            "logging.log_features enabled on a normal-traffic run first"
        )
    return np.asarray(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="gateway.jsonl files from normal runs")
    parser.add_argument("-o", "--output", default="models/vae.npz")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--latent", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_vectors(args.logs)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(data))
    split = max(1, int(len(data) * (1 - args.holdout)))
    train_data, holdout = data[order[:split]], data[order[split:]]

    weights = train(
        train_data,
        FEATURE_NAMES,
        hidden=args.hidden,
        latent=args.latent,
        epochs=args.epochs,
        seed=args.seed,
    )
    weights.save(args.output)

    print(f"trained on {len(train_data)} vectors, {len(FEATURE_NAMES)} features")
    print(f"recon mean {weights.recon_mean:.6f} std {weights.recon_std:.6f}")
    if len(holdout):
        scores = (
            recon_error_batch(standardize(holdout, weights), weights) - weights.recon_mean
        ) / weights.recon_std
        # These percentiles are the calibration for suspect/hostile: a
        # threshold below the baseline p99 will throttle normal users.
        print(
            "holdout score p50 {:.2f} p95 {:.2f} p99 {:.2f} max {:.2f}".format(
                np.percentile(scores, 50),
                np.percentile(scores, 95),
                np.percentile(scores, 99),
                scores.max(),
            )
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
