"""Generate the figures from the measured run data.

    python report/make_figures.py            # both themes
    python report/make_figures.py --theme dark

Reads ``report/data/t4_runs.json`` -- the transcribed output of the two
gateway runs on the T4 -- and writes the PNGs that the report and the slide
deck embed. Nothing here invents a number: every value plotted is present in
that file, which records the notebook cell it came from.

Two themes, same data. The light set goes in ``figures/`` for the report; the
dark set goes in ``figures/dark/`` so the charts sit on the deck's dark grey
background without a white box around them.

Three figures, each answering a question the reader should be allowed to check
rather than take on trust:

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

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "t4_runs.json")


@dataclass(frozen=True)
class Theme:
    name: str
    out_dir: str
    bg: str
    fg: str
    muted: str
    grid: str
    off: str
    on: str
    legit: str
    attacker: str
    rejected: str
    served_text: str
    font: str


LIGHT = Theme(
    name="light",
    out_dir=os.path.join(HERE, "figures"),
    bg="#ffffff",
    fg="#1a1a1a",
    muted="#555555",
    grid="#cccccc",
    off="#c62828",
    on="#1565c0",
    legit="#2e7d32",
    attacker="#ef6c00",
    rejected="#9e9e9e",
    served_text="#ffffff",
    font="DejaVu Sans",
)

# Desaturated on purpose. On a dark slide, saturated primaries vibrate and
# read as decoration rather than data.
DARK = Theme(
    name="dark",
    out_dir=os.path.join(HERE, "figures", "dark"),
    bg="#2e3033",
    fg="#e8e9ea",
    muted="#a8adb2",
    grid="#4a4e52",
    off="#b5544a",
    on="#5b8db8",
    legit="#6f9b6f",
    attacker="#c08a4a",
    rejected="#63686d",
    served_text="#f2f3f4",
    font="Aptos",
)


def load() -> dict[str, dict]:
    with open(DATA) as handle:
        payload = json.load(handle)
    return {run["arm"]: run for run in payload["runs"]}


def _apply(theme: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": [theme.font, "DejaVu Sans"],
            "figure.facecolor": theme.bg,
            "axes.facecolor": theme.bg,
            "savefig.facecolor": theme.bg,
            "text.color": theme.fg,
            "axes.labelcolor": theme.muted,
            "axes.edgecolor": theme.grid,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "grid.color": theme.grid,
        }
    )


def _annotate(ax, bars, values, theme, fmt="{:.3g}", suffix=""):
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value) + suffix,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=theme.fg,
        )


def _style(ax):
    ax.grid(alpha=0.35, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, theme: Theme, filename: str) -> str:
    os.makedirs(theme.out_dir, exist_ok=True)
    path = os.path.join(theme.out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def legit_protection(runs: dict[str, dict], theme: Theme) -> str:
    """Latency and throughput together. Neither is honest on its own."""
    arms = ["off", "on"]
    names = ["mitigation off", "mitigation on"]
    colours = [theme.off, theme.on]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

    for ax, values, title, unit, fmt, suffix in (
        (axes[0], [runs[a]["legit"]["p50_s"] for a in arms],
         "Legitimate p50 latency", "seconds", "{:.2f}", " s"),
        (axes[1], [runs[a]["legit"]["p95_s"] for a in arms],
         "Legitimate p95 latency", "seconds", "{:.2f}", " s"),
        (axes[2], [runs[a]["legit"]["requests"] for a in arms],
         "Legitimate requests answered in 20 s", "requests", "{:.0f}", ""),
    ):
        bars = ax.bar(names, values, color=colours)
        _annotate(ax, bars, values, theme, fmt, suffix)
        ax.set_title(title, fontsize=10, color=theme.fg)
        ax.set_ylabel(unit)
        _style(ax)
        plt.setp(ax.get_xticklabels(), fontsize=9)

    # Latency fell and throughput rose at the same time, with no legitimate
    # request rejected in either arm. Stated on the figure because a latency
    # improvement bought by shedding users would look identical otherwise.
    fig.suptitle(
        "Legitimate users during an 8-way flood  "
        "(0 legitimate requests rejected in either arm)",
        fontsize=11,
        color=theme.fg,
    )
    return _save(fig, theme, "fig1_legit_protection.png")


def separation(runs: dict[str, dict], theme: Theme) -> str:
    """Whether cheap and expensive traffic were charged differently."""
    arms = ["off", "on"]
    names = ["mitigation off", "mitigation on"]
    width = 0.34
    positions = range(len(arms))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    legit = [runs[a]["legit"]["p95_s"] for a in arms]
    attacker = [runs[a]["attacker"]["p95_s"] for a in arms]

    bars_l = ax.bar([p - width / 2 for p in positions], legit, width,
                    label="legitimate (32-token replies)", color=theme.legit)
    bars_a = ax.bar([p + width / 2 for p in positions], attacker, width,
                    label="attacker (512-token replies)", color=theme.attacker)
    _annotate(ax, bars_l, legit, theme, "{:.2f}", " s")
    _annotate(ax, bars_a, attacker, theme, "{:.2f}", " s")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("p95 latency (seconds)")
    ax.set_title(
        "Unmitigated, a cheap request waits as long as an expensive one",
        fontsize=10.5,
        color=theme.fg,
    )
    # Headroom so the legend sits in the gap between the two groups instead
    # of over the tallest bar's label.
    ax.set_ylim(0, max(legit + attacker) * 1.32)
    legend = ax.legend(fontsize=8.5, loc="upper center", facecolor=theme.bg,
                       edgecolor=theme.grid)
    for text in legend.get_texts():
        text.set_color(theme.fg)
    _style(ax)
    return _save(fig, theme, "fig2_cost_separation.png")


def outcomes(runs: dict[str, dict], theme: Theme) -> str:
    """Counts, so "served" cannot be confused with "not rejected"."""
    columns = []
    for arm in ("off", "on"):
        for label in ("legit", "attacker"):
            entry = runs[arm][label]
            served = sum(
                count for status, count in entry["statuses"].items() if status == "200"
            )
            columns.append(
                (
                    f"{label}\nmitigation {arm}",
                    served,
                    entry["requests"] - served,
                    entry["reasons"],
                )
            )

    names = [c[0] for c in columns]
    served = [c[1] for c in columns]
    rejected = [c[2] for c in columns]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(names, served, label="answered 200", color=theme.legit)
    ax.bar(names, rejected, bottom=served, label="rejected 429", color=theme.rejected)
    for index, (_, ok, bad, reasons) in enumerate(columns):
        ax.annotate(f"{ok}", (index, ok / 2), ha="center", va="center",
                    fontsize=9, color=theme.served_text, fontweight="bold")
        if bad:
            reason = ", ".join(reasons) or "rejected"
            ax.annotate(f"{bad}\n{reason}", (index, ok + bad / 2), ha="center",
                        va="center", fontsize=7.5, color=theme.fg)
    ax.axvline(1.5, color=theme.grid, lw=1)
    ax.set_ylabel("requests in 20 s")
    ax.set_title("Where every request ended up", fontsize=10.5, color=theme.fg)
    legend = ax.legend(fontsize=8.5, facecolor=theme.bg, edgecolor=theme.grid)
    for text in legend.get_texts():
        text.set_color(theme.fg)
    _style(ax)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    return _save(fig, theme, "fig3_outcomes.png")


def build(theme: Theme) -> list[str]:
    _apply(theme)
    runs = load()
    return [
        legit_protection(runs, theme),
        separation(runs, theme),
        outcomes(runs, theme),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    args = parser.parse_args()

    themes = {"light": [LIGHT], "dark": [DARK], "both": [LIGHT, DARK]}[args.theme]
    for theme in themes:
        for path in build(theme):
            print("wrote", path)


if __name__ == "__main__":
    main()
