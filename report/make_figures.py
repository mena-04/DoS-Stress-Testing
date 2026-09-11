"""Generate the report's figures from the measured run data.

    python report/make_figures.py

Reads ``report/data/t4_runs.json`` -- the transcribed output of the two
gateway runs on the T4 -- and writes the PNGs the report embeds. Nothing here
invents a number: every value plotted is present in that file, which records
the notebook cell it came from.

Three figures, each answering a question the reader should be allowed to
check rather than take on trust:

1. Did legitimate users get a better service? Latency *and* throughput, side
   by side, because latency alone cannot distinguish protection from denial:
   a gateway that rejects most traffic has an excellent p95.
2. Did the gateway tell legitimate and attacking traffic apart? The two
   latencies in both arms; in the unmitigated arm they should be identical,
   which is the failure being fixed.
3. Where did every request actually end up? Counts, not rates, so "served"
   and "not rejected" cannot be confused.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "t4_runs.json")
OUT = os.path.join(HERE, "figures")

OFF = "#c62828"
ON = "#1565c0"
LEGIT = "#2e7d32"
ATTACKER = "#ef6c00"
GREY = "#9e9e9e"


def load() -> dict[str, dict]:
    with open(DATA) as handle:
        payload = json.load(handle)
    return {run["arm"]: run for run in payload["runs"]}


def _annotate(ax, bars, values, fmt="{:.3g}", suffix=""):
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value) + suffix,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _style(ax):
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legit_protection(runs: dict[str, dict]) -> str:
    """Latency and throughput together. Neither is honest on its own."""
    arms = ["off", "on"]
    names = ["mitigation off", "mitigation on"]
    colours = [OFF, ON]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

    p50 = [runs[a]["legit"]["p50_s"] for a in arms]
    bars = axes[0].bar(names, p50, color=colours)
    _annotate(axes[0], bars, p50, "{:.2f}", " s")
    axes[0].set_title("Legitimate p50 latency", fontsize=10)
    axes[0].set_ylabel("seconds")

    p95 = [runs[a]["legit"]["p95_s"] for a in arms]
    bars = axes[1].bar(names, p95, color=colours)
    _annotate(axes[1], bars, p95, "{:.2f}", " s")
    axes[1].set_title("Legitimate p95 latency", fontsize=10)
    axes[1].set_ylabel("seconds")

    served = [runs[a]["legit"]["requests"] for a in arms]
    bars = axes[2].bar(names, served, color=colours)
    _annotate(axes[2], bars, served, "{:.0f}")
    axes[2].set_title("Legitimate requests answered in 20 s", fontsize=10)
    axes[2].set_ylabel("requests")

    for ax in axes:
        _style(ax)
        plt.setp(ax.get_xticklabels(), fontsize=9)
    # Latency fell and throughput rose at the same time, with no legitimate
    # request rejected in either arm. Stated on the figure because a latency
    # improvement bought by shedding users would look identical otherwise.
    fig.suptitle(
        "Legitimate users during an 8-way flood  (0 legitimate requests rejected in either arm)",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig1_legit_protection.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def separation(runs: dict[str, dict]) -> str:
    """Whether cheap and expensive traffic were charged differently."""
    arms = ["off", "on"]
    names = ["mitigation off", "mitigation on"]
    width = 0.34
    positions = range(len(arms))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    legit = [runs[a]["legit"]["p95_s"] for a in arms]
    attacker = [runs[a]["attacker"]["p95_s"] for a in arms]

    bars_l = ax.bar([p - width / 2 for p in positions], legit, width,
                    label="legitimate (32-token replies)", color=LEGIT)
    bars_a = ax.bar([p + width / 2 for p in positions], attacker, width,
                    label="attacker (512-token replies)", color=ATTACKER)
    _annotate(ax, bars_l, legit, "{:.2f}", " s")
    _annotate(ax, bars_a, attacker, "{:.2f}", " s")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("p95 latency (seconds)")
    ax.set_title(
        "Unmitigated, a cheap request waits as long as an expensive one",
        fontsize=10.5,
    )
    # Headroom so the legend sits in the gap between the two groups instead
    # of over the tallest bar's label.
    ax.set_ylim(0, max(legit + attacker) * 1.32)
    ax.legend(fontsize=8.5, loc="upper center")
    _style(ax)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig2_cost_separation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def outcomes(runs: dict[str, dict]) -> str:
    """Counts, so "served" cannot be confused with "not rejected"."""
    columns = []
    for arm, arm_name in (("off", "off"), ("on", "on")):
        for label in ("legit", "attacker"):
            entry = runs[arm][label]
            served = sum(
                count for status, count in entry["statuses"].items() if status == "200"
            )
            rejected = entry["requests"] - served
            columns.append(
                (
                    f"{label}\nmitigation {arm_name}",
                    served,
                    rejected,
                    entry["reasons"],
                )
            )

    names = [c[0] for c in columns]
    served = [c[1] for c in columns]
    rejected = [c[2] for c in columns]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(names, served, label="answered 200", color=LEGIT)
    ax.bar(names, rejected, bottom=served, label="rejected 429", color=GREY)
    for index, (name, ok, bad, reasons) in enumerate(columns):
        ax.annotate(f"{ok}", (index, ok / 2), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        if bad:
            reason = ", ".join(reasons) or "rejected"
            ax.annotate(f"{bad}\n{reason}", (index, ok + bad / 2), ha="center",
                        va="center", fontsize=7.5)
    ax.axvline(1.5, color="#cccccc", lw=1)
    ax.set_ylabel("requests in 20 s")
    ax.set_title("Where every request ended up", fontsize=10.5)
    ax.legend(fontsize=8.5)
    _style(ax)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig3_outcomes.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    runs = load()
    for path in (legit_protection(runs), separation(runs), outcomes(runs)):
        print("wrote", path)


if __name__ == "__main__":
    main()
