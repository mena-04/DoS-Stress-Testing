"""A small VAE in numpy.

numpy rather than torch for two reasons: the serving forward pass stays well
under a millisecond with no framework import in the gateway process, and the
whole thing trains on CPU inside the same Colab runtime that is already
holding a GPU for vLLM.

Trained on normal traffic only. Reconstruction error is therefore a novelty
signal: it reports "this client does not look like anything in the baseline",
without anyone hand-labelling attacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VAEWeights:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    W4: np.ndarray
    b4: np.ndarray
    W5: np.ndarray
    b5: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    recon_mean: float
    recon_std: float
    feature_names: tuple[str, ...]

    def save(self, path: str) -> None:
        np.savez(
            path,
            **{
                name: getattr(self, name)
                for name in (
                    "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5",
                    "feature_mean", "feature_std",
                )
            },
            recon_mean=np.array(self.recon_mean),
            recon_std=np.array(self.recon_std),
            feature_names=np.array(self.feature_names),
        )

    @classmethod
    def load(cls, path: str) -> "VAEWeights":
        data = np.load(path, allow_pickle=False)
        return cls(
            **{
                name: data[name]
                for name in (
                    "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5",
                    "feature_mean", "feature_std",
                )
            },
            recon_mean=float(data["recon_mean"]),
            recon_std=float(data["recon_std"]),
            feature_names=tuple(str(n) for n in data["feature_names"]),
        )


def init_weights(
    input_dim: int, hidden: int, latent: int, feature_names: tuple[str, ...], seed: int = 0
) -> VAEWeights:
    rng = np.random.default_rng(seed)

    def he(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, np.sqrt(2.0 / rows), size=(rows, cols))

    return VAEWeights(
        W1=he(input_dim, hidden), b1=np.zeros(hidden),
        W2=he(hidden, latent), b2=np.zeros(latent),
        W3=he(hidden, latent), b3=np.zeros(latent),
        W4=he(latent, hidden), b4=np.zeros(hidden),
        W5=he(hidden, input_dim), b5=np.zeros(input_dim),
        feature_mean=np.zeros(input_dim), feature_std=np.ones(input_dim),
        recon_mean=0.0, recon_std=1.0,
        feature_names=feature_names,
    )


def standardize(x: np.ndarray, weights: VAEWeights) -> np.ndarray:
    return (x - weights.feature_mean) / weights.feature_std


def reconstruct(x_std: np.ndarray, weights: VAEWeights) -> np.ndarray:
    """Deterministic pass: the latent mean is used instead of a sample.

    Sampling at serving time would make an identical client score differently
    on consecutive ticks, which is not a property you want in a rate limiter.
    """
    h1 = np.maximum(0.0, x_std @ weights.W1 + weights.b1)
    mu = h1 @ weights.W2 + weights.b2
    h2 = np.maximum(0.0, mu @ weights.W4 + weights.b4)
    return h2 @ weights.W5 + weights.b5


def recon_error(x: np.ndarray, weights: VAEWeights) -> np.ndarray:
    x_std = standardize(np.atleast_2d(x), weights)
    x_hat = reconstruct(x_std, weights)
    return np.mean((x_hat - x_std) ** 2, axis=1)


def anomaly_score(x: np.ndarray, weights: VAEWeights) -> float:
    """Reconstruction error expressed in baseline standard deviations.

    Reporting a z-score rather than a raw error keeps the configured
    thresholds meaningful across retrains and feature changes.
    """
    error = float(recon_error(x, weights)[0])
    return (error - weights.recon_mean) / max(weights.recon_std, 1e-9)


def train(
    data: np.ndarray,
    feature_names: tuple[str, ...],
    hidden: int = 32,
    latent: int = 4,
    epochs: int = 400,
    batch_size: int = 64,
    lr: float = 3e-3,
    beta: float = 1.0,
    seed: int = 0,
) -> VAEWeights:
    rng = np.random.default_rng(seed)
    n, d = data.shape
    weights = init_weights(d, hidden, latent, feature_names, seed)

    weights.feature_mean = data.mean(axis=0)
    std = data.std(axis=0)
    # Constant features would divide by zero; a unit std leaves them at zero
    # after centring, contributing nothing rather than exploding.
    weights.feature_std = np.where(std < 1e-6, 1.0, std)
    x_all = standardize(data, weights)

    names = ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5")
    adam_m = {k: np.zeros_like(getattr(weights, k)) for k in names}
    adam_v = {k: np.zeros_like(getattr(weights, k)) for k in names}
    step = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = x_all[order[start : start + batch_size]]
            m = batch.shape[0]
            if m == 0:
                continue

            h1 = np.maximum(0.0, batch @ weights.W1 + weights.b1)
            mu = h1 @ weights.W2 + weights.b2
            logvar = np.clip(h1 @ weights.W3 + weights.b3, -8.0, 8.0)
            eps = rng.normal(size=mu.shape)
            z = mu + np.exp(0.5 * logvar) * eps
            h2 = np.maximum(0.0, z @ weights.W4 + weights.b4)
            x_hat = h2 @ weights.W5 + weights.b5

            d_xhat = 2.0 * (x_hat - batch) / m
            g = {}
            g["W5"] = h2.T @ d_xhat
            g["b5"] = d_xhat.sum(axis=0)
            d_h2 = (d_xhat @ weights.W5.T) * (h2 > 0)
            g["W4"] = z.T @ d_h2
            g["b4"] = d_h2.sum(axis=0)
            d_z = d_h2 @ weights.W4.T

            d_mu = d_z + beta * mu / m
            d_logvar = d_z * eps * 0.5 * np.exp(0.5 * logvar)
            d_logvar += beta * -0.5 * (1.0 - np.exp(logvar)) / m

            g["W2"] = h1.T @ d_mu
            g["b2"] = d_mu.sum(axis=0)
            g["W3"] = h1.T @ d_logvar
            g["b3"] = d_logvar.sum(axis=0)
            d_h1 = (d_mu @ weights.W2.T + d_logvar @ weights.W3.T) * (h1 > 0)
            g["W1"] = batch.T @ d_h1
            g["b1"] = d_h1.sum(axis=0)

            step += 1
            for key in names:
                adam_m[key] = 0.9 * adam_m[key] + 0.1 * g[key]
                adam_v[key] = 0.999 * adam_v[key] + 0.001 * (g[key] ** 2)
                m_hat = adam_m[key] / (1 - 0.9**step)
                v_hat = adam_v[key] / (1 - 0.999**step)
                setattr(
                    weights,
                    key,
                    getattr(weights, key) - lr * m_hat / (np.sqrt(v_hat) + 1e-8),
                )

    errors = recon_error_batch(x_all, weights)
    weights.recon_mean = float(errors.mean())
    weights.recon_std = float(max(errors.std(), 1e-9))
    return weights


def recon_error_batch(x_std: np.ndarray, weights: VAEWeights) -> np.ndarray:
    x_hat = reconstruct(x_std, weights)
    return np.mean((x_hat - x_std) ** 2, axis=1)
