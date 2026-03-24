#!/usr/bin/env python3
"""
Analyze tail distribution of Roll (LiDAR-X) errors in BEVCalib evaluation files.
"""
import re
from pathlib import Path
from collections import defaultdict

# Sequence mapping from eval_run.log: seq 00: 375 frames, 01: 142, 02: 929
SEQ_00_END = 375
SEQ_01_END = 375 + 142  # 517
# seq 02: 517 to end


def sample_to_sequence(sample_idx: int) -> str:
    if sample_idx < SEQ_00_END:
        return "00"
    if sample_idx < SEQ_01_END:
        return "01"
    return "02"


def parse_extrinsics_errors(filepath: str) -> list[dict]:
    """Parse extrinsics_and_errors.txt and return list of per-sample error dicts."""
    with open(filepath, "r") as f:
        content = f.read()

    samples = []
    # Match blocks: Sample XXXX ... Rotation Errors ... Roll/Pitch/Yaw/Total
    block_pattern = re.compile(
        r"Sample\s+(\d+)\s+.*?"
        r"Rotation Errors.*?"
        r"Total:\s+([\d.]+)\s+deg\s+"
        r"Roll\s+\(LiDAR X\):\s+([\d.]+)\s+deg\s+"
        r"Pitch \(LiDAR Y\):\s+([\d.]+)\s+deg\s+"
        r"Yaw\s+\(LiDAR Z\):\s+([\d.]+)\s+deg",
        re.DOTALL,
    )
    for m in block_pattern.finditer(content):
        sample_idx = int(m.group(1))
        total = float(m.group(2))
        roll = float(m.group(3))
        pitch = float(m.group(4))
        yaw = float(m.group(5))
        samples.append({
            "sample": sample_idx,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "total": total,
            "sequence": sample_to_sequence(sample_idx),
        })
    return samples


def analyze_file(filepath: str, label: str) -> dict:
    samples = parse_extrinsics_errors(filepath)
    if not samples:
        return {"label": label, "error": "No samples parsed", "samples": []}

    roll_gt_08 = [s for s in samples if s["roll"] > 0.8]
    roll_gt_10 = [s for s in samples if s["roll"] > 1.0]

    # Sequence breakdown for high-error samples
    seq_08 = defaultdict(list)
    seq_10 = defaultdict(list)
    for s in roll_gt_08:
        seq_08[s["sequence"]].append(s["sample"])
    for s in roll_gt_10:
        seq_10[s["sequence"]].append(s["sample"])

    # Top 10 worst Roll
    sorted_by_roll = sorted(samples, key=lambda x: x["roll"], reverse=True)
    top10 = sorted_by_roll[:10]

    return {
        "label": label,
        "filepath": filepath,
        "total_samples": len(samples),
        "roll_gt_08_count": len(roll_gt_08),
        "roll_gt_10_count": len(roll_gt_10),
        "seq_08": dict(seq_08),
        "seq_10": dict(seq_10),
        "top10_worst_roll": top10,
        "samples": samples,
    }


def main():
    files = [
        (
            "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/generalization_eval_b26a_v1/v7-opt-axis05/extrinsics_and_errors.txt",
            "v7-opt-axis05 (generalization_eval_b26a_v1)",
        ),
        (
            "/mnt/drtraining/user/dahailu/code/BEVCalib/logs/all_data_generalization_eval_v3/v5-z2-axis05/extrinsics_and_errors.txt",
            "v5-z2-axis05 (all_data_generalization_eval_v3)",
        ),
    ]

    results = []
    for filepath, label in files:
        if not Path(filepath).exists():
            results.append({"label": label, "error": f"File not found: {filepath}"})
            continue
        results.append(analyze_file(filepath, label))

    # Print report
    print("=" * 80)
    print("BEVCalib Roll (LiDAR-X) Error Tail Distribution Analysis")
    print("=" * 80)

    for r in results:
        if "error" in r:
            print(f"\n[{r['label']}] ERROR: {r['error']}")
            continue

        print(f"\n{'='*80}")
        print(f"Model: {r['label']}")
        print(f"File: {r['filepath']}")
        print(f"{'='*80}")
        print(f"Total samples: {r['total_samples']}")
        print(f"Samples with Roll > 0.8°: {r['roll_gt_08_count']} ({100*r['roll_gt_08_count']/r['total_samples']:.2f}%)")
        print(f"Samples with Roll > 1.0°: {r['roll_gt_10_count']} ({100*r['roll_gt_10_count']/r['total_samples']:.2f}%)")

        print("\nSequence breakdown (Roll > 0.8°):")
        for seq in ["00", "01", "02"]:
            indices = r["seq_08"].get(seq, [])
            print(f"  Seq {seq}: {len(indices)} samples {indices[:20]}{'...' if len(indices) > 20 else ''}")

        print("\nSequence breakdown (Roll > 1.0°):")
        for seq in ["00", "01", "02"]:
            indices = r["seq_10"].get(seq, [])
            print(f"  Seq {seq}: {len(indices)} samples {indices[:20]}{'...' if len(indices) > 20 else ''}")

        print("\nTop 10 worst Roll error samples:")
        print(f"  {'Sample':>8} {'Roll(°)':>10} {'Pitch(°)':>10} {'Yaw(°)':>10} {'Total(°)':>10} {'Seq':>4}")
        print("  " + "-" * 54)
        for s in r["top10_worst_roll"]:
            print(f"  {s['sample']:>8} {s['roll']:>10.4f} {s['pitch']:>10.4f} {s['yaw']:>10.4f} {s['total']:>10.4f} {s['sequence']:>4}")

        print("\nPerturbation pattern note: The extrinsics_and_errors.txt file does NOT contain")
        print("per-sample perturbation angles. Perturbation (5.0deg, 0.15m) is applied at eval")
        print("time with random angles per sample; these are not logged. To correlate high Roll")
        print("errors with perturbation, re-run evaluation with fixed seed and log perturbations.")

    print("\n" + "=" * 80)
    print("Summary comparison")
    print("=" * 80)
    for r in results:
        if "error" in r:
            continue
        print(f"  {r['label']}: Roll>0.8°: {r['roll_gt_08_count']}, Roll>1.0°: {r['roll_gt_10_count']}")

    # Cross-model overlap of high-Roll samples
    valid = [r for r in results if "error" not in r and "samples" in r]
    if len(valid) >= 2:
        set_08_a = {s["sample"] for s in valid[0]["samples"] if s["roll"] > 0.8}
        set_08_b = {s["sample"] for s in valid[1]["samples"] if s["roll"] > 0.8}
        overlap_08 = set_08_a & set_08_b
        print("\nCross-model overlap (Roll > 0.8°):")
        print(f"  v7-opt-axis05 high-Roll samples: {len(set_08_a)}")
        print(f"  v5-z2-axis05 high-Roll samples: {len(set_08_b)}")
        print(f"  Overlap (both models have Roll>0.8°): {len(overlap_08)} samples")
        if overlap_08:
            print(f"  Overlapping sample indices: {sorted(overlap_08)[:30]}{'...' if len(overlap_08) > 30 else ''}")


if __name__ == "__main__":
    main()
