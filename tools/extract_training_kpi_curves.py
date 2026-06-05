#!/usr/bin/env python3
"""Parse train.log for MEDW200 + Jacobian eval curves (p0ab / full / P1d)."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

MEDW_RE = re.compile(
    r"Epoch \[(\d+)/\d+\], MEDW200 \(val reuse\): Rot=([\d.]+)° "
    r"\(R:([\d.]+) P:([\d.]+) Y:([\d.]+)\)"
)
JAC_RE = re.compile(
    r"Epoch \[(\d+)/\d+\], Jacobian ±([\d.]+)° \(from val\): "
    r"Overall=([-\d.]+) \(R:([-\d.]+) P:([-\d.]+) Y:([-\d.]+)\) \[(\w+)\]"
)


def parse_log(path: Path):
    rows = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in MEDW_RE.finditer(text):
        ep = int(m.group(1))
        rows.setdefault(ep, {})["epoch"] = ep
        rows[ep].update({
            "medw_rot": float(m.group(2)),
            "medw_r": float(m.group(3)),
            "medw_p": float(m.group(4)),
            "medw_y": float(m.group(5)),
        })
    for m in JAC_RE.finditer(text):
        ep = int(m.group(1))
        rows.setdefault(ep, {})["epoch"] = ep
        rows[ep].update({
            "jac_angle": float(m.group(2)),
            "jac_overall": float(m.group(3)),
            "jac_r": float(m.group(4)),
            "jac_p": float(m.group(5)),
            "jac_y": float(m.group(6)),
            "jac_verdict": m.group(7),
        })
    return [rows[k] for k in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_files", nargs="+", help="train.log paths")
    ap.add_argument("--csv", default="", help="optional combined CSV output")
    ap.add_argument("--md", default="", help="optional markdown summary")
    args = ap.parse_args()

    all_rows = []
    for lf in args.log_files:
        p = Path(lf)
        tag = p.parent.name
        for row in parse_log(p):
            row = dict(row)
            row["run"] = tag
            all_rows.append(row)

    if not all_rows:
        print("No MEDW/Jacobian eval lines found.", file=sys.stderr)
        return 1

    hdr = ["run", "epoch", "medw_rot", "medw_r", "medw_p", "medw_y",
           "jac_overall", "jac_r", "jac_p", "jac_y", "jac_verdict"]
    print(",".join(hdr))
    for r in all_rows:
        print(",".join(str(r.get(k, "")) for k in hdr))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {out}", file=sys.stderr)

    if args.md:
        lines = ["# Training KPI Curves (MEDW200 + Jacobian)", ""]
        for run in sorted({r["run"] for r in all_rows}):
            sub = [r for r in all_rows if r["run"] == run]
            lines.append(f"## {run}")
            lines.append("")
            lines.append("| Epoch | MEDW200 | R | P | Y | Jac Overall | Jac R | Jac P | Jac Y | Verdict |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for r in sub:
                lines.append(
                    "| {epoch} | {medw_rot:.4f} | {medw_r:.4f} | {medw_p:.4f} | {medw_y:.4f} | "
                    "{jac_overall} | {jac_r} | {jac_p} | {jac_y} | {jac_verdict} |".format(
                        epoch=r.get("epoch", ""),
                        medw_rot=r.get("medw_rot", float("nan")),
                        medw_r=r.get("medw_r", float("nan")),
                        medw_p=r.get("medw_p", float("nan")),
                        medw_y=r.get("medw_y", float("nan")),
                        jac_overall=r.get("jac_overall", ""),
                        jac_r=r.get("jac_r", ""),
                        jac_p=r.get("jac_p", ""),
                        jac_y=r.get("jac_y", ""),
                        jac_verdict=r.get("jac_verdict", ""),
                    )
                )
            lines.append("")
        Path(args.md).write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
