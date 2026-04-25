#!/usr/bin/env python3
"""Parse extrinsics_and_errors.txt files and summarize per-sample RPY error statistics."""

import argparse
import math
import re
import statistics
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

SAMPLE_RE = re.compile(r"^Sample\s+(\d+)\s+\[Seq\s+(\d+)\]")
ROLL_RE = re.compile(r"Roll[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
PITCH_RE = re.compile(r"Pitch[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
YAW_RE = re.compile(r"Yaw[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class Sample(NamedTuple):
    seq: int
    roll: float
    pitch: float
    yaw: float


def parse_eval_file(path: str) -> List[Sample]:
    samples: List[Sample] = []
    seq: Optional[int] = None
    roll = pitch = yaw = None

    def flush() -> None:
        nonlocal roll, pitch, yaw, seq
        if seq is not None and roll is not None and pitch is not None and yaw is not None:
            samples.append(Sample(seq, roll, pitch, yaw))
        roll = pitch = yaw = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SAMPLE_RE.match(line.strip())
            if m:
                flush()
                seq = int(m.group(2))
                continue
            if seq is None:
                continue
            rm = ROLL_RE.search(line)
            if rm:
                roll = float(rm.group(1))
                continue
            pm = PITCH_RE.search(line)
            if pm:
                pitch = float(pm.group(1))
                continue
            ym = YAW_RE.search(line)
            if ym:
                yaw = float(ym.group(1))
                continue
    flush()
    return samples


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile p in [0, 100], like numpy default."""
    if not sorted_vals:
        return float("nan")
    x = sorted_vals
    n = len(x)
    if n == 1:
        return x[0]
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return x[f]
    return x[f] + (k - f) * (x[c] - x[f])


def axis_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ("mean", "std", "p10", "p25", "p50", "p75", "p80", "p90", "p95", "max")}
    s = sorted(values)
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(s, 10),
        "p25": percentile(s, 25),
        "p50": percentile(s, 50),
        "p75": percentile(s, 75),
        "p80": percentile(s, 80),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "max": max(values),
    }


def pct_under(values: List[float], thresh: float) -> float:
    if not values:
        return float("nan")
    c = sum(1 for v in values if v < thresh)
    return 100.0 * c / len(values)


def pct_all_axes_under(samples: List[Sample], thresh: float) -> float:
    if not samples:
        return float("nan")
    c = sum(1 for s in samples if s.roll < thresh and s.pitch < thresh and s.yaw < thresh)
    return 100.0 * c / len(samples)


def print_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None) -> None:
    if col_widths is None:
        col_widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    sep = " | "
    head = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(head)
    print("-+-".join("-" * col_widths[i] for i in range(len(headers))))
    for row in rows:
        print(sep.join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def fmt(x: float, nd: int = 6) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def fmt_pct(x: float, nd: int = 2) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}%"


def report_domain(name: str, samples: List[Sample]) -> None:
    print(f"\n{'=' * 80}")
    print(f"DOMAIN: {name}  (n = {len(samples)} samples)")
    print("=" * 80)

    rolls = [s.roll for s in samples]
    pitches = [s.pitch for s in samples]
    yaws = [s.yaw for s in samples]

    for axis, vals in ("Roll", rolls), ("Pitch", pitches), ("Yaw", yaws):
        st = axis_stats(vals)
        print(f"\n--- Per-axis statistics: {axis} (deg) ---")
        headers = ["metric", "value"]
        rows = [
            ["mean", fmt(st["mean"])],
            ["std", fmt(st["std"])],
            ["P10", fmt(st["p10"])],
            ["P25", fmt(st["p25"])],
            ["P50", fmt(st["p50"])],
            ["P75", fmt(st["p75"])],
            ["P80", fmt(st["p80"])],
            ["P90", fmt(st["p90"])],
            ["P95", fmt(st["p95"])],
            ["max", fmt(st["max"])],
        ]
        print_table(headers, rows)

    thresholds = (0.1, 0.2, 0.3, 0.5)
    print("\n--- Per-axis: % of samples with error < threshold ---")
    h = ["threshold (deg)", "Roll < t", "Pitch < t", "Yaw < t"]
    rows = []
    for t in thresholds:
        rows.append(
            [
                str(t),
                fmt_pct(pct_under(rolls, t)),
                fmt_pct(pct_under(pitches, t)),
                fmt_pct(pct_under(yaws, t)),
            ]
        )
    print_table(h, rows)

    print("\n--- ALL three axes (R, P, Y) simultaneously < threshold ---")
    h2 = ["threshold (deg)", "% samples ALL < t"]
    rows2 = [[str(t), fmt_pct(pct_all_axes_under(samples, t))] for t in thresholds]
    print_table(h2, rows2)


def report_test_sequences(samples: List[Sample]) -> None:
    print(f"\n{'=' * 80}")
    print("TEST DOMAIN — Per-sequence breakdown (12 sequences: Seq 00–11)")
    print("=" * 80)

    by_seq: Dict[int, List[Sample]] = defaultdict(list)
    for s in samples:
        by_seq[s.seq].append(s)

    h = [
        "Seq",
        "n",
        "mean_Roll",
        "mean_Pitch",
        "mean_Yaw",
        "% ALL RPY < 0.1°",
    ]
    rows = []
    for seq in range(12):
        sl = by_seq.get(seq, [])
        n = len(sl)
        if n == 0:
            rows.append([f"{seq:02d}", "0", "—", "—", "—", "—"])
            continue
        mr = statistics.mean(x.roll for x in sl)
        mp = statistics.mean(x.pitch for x in sl)
        my = statistics.mean(x.yaw for x in sl)
        pall = pct_all_axes_under(sl, 0.1)
        rows.append([f"{seq:02d}", str(n), fmt(mr, 4), fmt(mp, 4), fmt(my, 4), fmt_pct(pall)])
    print_table(h, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze RPY errors from extrinsics_and_errors.txt")
    ap.add_argument(
        "--test",
        default="/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_v2/v20-v8recipe-pitch-wt3/extrinsics_and_errors.txt",
    )
    ap.add_argument(
        "--train",
        default="/mnt/drtraining/user/dahailu/code/BEVCalib/logs/evaluations/generalization_eval_test_models2_all_data/v20-v8recipe-pitch-wt3/extrinsics_and_errors.txt",
    )
    args = ap.parse_args()

    test_samples = parse_eval_file(args.test)
    train_samples = parse_eval_file(args.train)

    print("RPY error analysis (parsed from Rotation Errors block: Roll/Pitch/Yaw in deg)")
    print(f"Test file:  {args.test}")
    print(f"Train file: {args.train}")

    report_domain("Test (generalization / test_models2_v2)", test_samples)
    report_domain("Training domain (all_training_data eval)", train_samples)
    report_test_sequences(test_samples)

    print(f"\n{'=' * 80}")
    print("SUMMARY — Is RPY ALL < 0.1° achievable?")
    print("=" * 80)
    t01 = pct_all_axes_under(test_samples, 0.1)
    tr01 = pct_all_axes_under(train_samples, 0.1)
    print(f"Test domain:   {fmt_pct(t01)} of samples have Roll, Pitch, and Yaw each < 0.1°.")
    print(f"Train domain:  {fmt_pct(tr01)} of samples have Roll, Pitch, and Yaw each < 0.1°.")
    print("=" * 80)


if __name__ == "__main__":
    main()
