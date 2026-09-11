"""Plot actual gateway samples. Missing or stale readings never become zero."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_queue_depth(paths, output_dir):
    """paths is e.g. {'OFF': Path(...jsonl), 'ON': Path(...jsonl)}."""
    output_dir = Path(output_dir)
    missing = [str(p) for p in paths.values() if not Path(p).is_file()]
    if missing:
        print("No queue chart generated: the raw files below are missing.")
        for p in missing:
            print("  ", p)
        print("Use the original Colab runtime, or upload these JSONL files from it.")
        print("The supplied ZIP contains the summary/PNG, but not these raw logs.")
        return None

    frames, notes = [], []
    for mode, path in paths.items():
        path = Path(path)
        # Ignore only an incomplete trailing line, as a live logger may still
        # be writing it; never silently discard an interior corrupt record.
        lines = path.read_text().splitlines()
        records = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise ValueError(f"Malformed JSON inside {path}, line {i+1}")
                print(f"Ignoring incomplete trailing line in {path.name}.")
        frame = pd.DataFrame(records)
        if frame.empty or not {"ts", "upstream_waiting"}.issubset(frame.columns):
            print(f"No usable timestamp/queue records in {path}")
            return None
        frame["ts"] = pd.to_numeric(frame["ts"], errors="coerce")
        frame = frame.dropna(subset=["ts"]).sort_values("ts").copy()
        if frame.empty:
            print(f"No usable timestamps in {path}")
            return None

        # New runs have exact load-start timestamps. Old runs do not; their
        # x-axis honestly uses each file's first saved sample instead.
        load_meta = path.parent / "load_profile.json"
        metadata = json.loads(load_meta.read_text()) if load_meta.exists() else {}
        start_ts = metadata.get("start_ts", frame["ts"].iloc[0])
        if "start_ts" in metadata:
            frame = frame[frame["ts"] >= start_ts].copy()
        if frame.empty:
            print(f"No samples after load start in {path}")
            return None
        frame["elapsed_s"] = frame["ts"] - start_ts
        frame["waiting"] = pd.to_numeric(frame["upstream_waiting"], errors="coerce")
        if "upstream_stale" in frame:
            stale = frame["upstream_stale"].map(
                lambda x: True if pd.isna(x) else str(x).lower() in {"true", "1", "1.0"}
            )
            frame.loc[stale, "waiting"] = float("nan")
        else:
            stale = pd.Series(False, index=frame.index)
            print(f"{mode}: this file has no staleness flag; freshness cannot be checked.")
        frame.loc[frame["waiting"] < 0, "waiting"] = float("nan")
        frame["mode"] = mode
        frame["source"] = str(path)
        frame["stale"] = stale
        if frame["waiting"].notna().sum() == 0:
            print(f"{mode}: no fresh queue readings. Not making a misleading comparison.")
            return None
        notes.append({"mode": mode, "samples": len(frame),
                      "fresh_samples": int(frame["waiting"].notna().sum()),
                      "stale_fraction": float(stale.mean()),
                      "max_observed_queue": float(frame["waiting"].max()),
                      "time_origin": "load start" if "start_ts" in metadata else "first saved sample"})
        frames.append(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for frame in frames:
        ax.plot(frame["elapsed_s"], frame["waiting"],
                drawstyle="steps-post", label=frame["mode"].iloc[0], linewidth=1.6)
    ax.set(xlabel="Seconds from each run's origin (see summary)",
           ylabel="vLLM waiting requests",
           title="Observed backend queue depth: mitigation OFF vs ON")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = output_dir / "queue_depth_off_vs_on.png"
    csv = output_dir / "queue_depth_samples.csv"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    pd.concat(frames, ignore_index=True)[
        ["mode", "ts", "elapsed_s", "waiting", "stale", "source"]
    ].to_csv(csv, index=False)
    pd.DataFrame(notes).to_csv(output_dir / "queue_depth_summary.csv", index=False)
    print(pd.DataFrame(notes).to_string(index=False))
    print("Saved:", png, "and", csv)
    plt.show()
    plt.close(fig)
    return png
