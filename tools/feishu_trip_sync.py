#!/usr/bin/env python3
"""
Feishu Bot <-> BEVCalib trip list sync tool.

Features:
  1. Parse trip names from text (clipboard, Feishu messages, etc.)
  2. Update trips.txt with parsed trip names
  3. Push calibration results to Feishu webhook

Trip name pattern: YR-<VehicleType>-<ID>_<YYYYMMDD>_<HHMMSS>

Usage:
  # Parse trips from text and update trips.txt
  python feishu_trip_sync.py parse --text "标定任务: YR-C01-81_20260427_112840, YR-P789-56_20260506_064240"
  python feishu_trip_sync.py parse --file message.txt --output /path/to/trips.txt

  # Push calibration results to Feishu
  python feishu_trip_sync.py push --report /path/to/SUMMARY_REPORT.md
  python feishu_trip_sync.py push --report /path/to/SUMMARY_REPORT.md --webhook <url>

  # Watch stdin for trip names (pipe from clipboard, etc.)
  echo "YR-C01-81_20260427_112840" | python feishu_trip_sync.py parse --stdin
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error

TRIP_PATTERN = re.compile(r"YR-[A-Za-z0-9]+-\d+_\d{8}_\d{6}")

DEFAULT_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/35acb729-fb93-41a9-acd3-e3c5079a524f"
DEFAULT_TRIPS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "bevcalib", "bag_lists", "remote_trips.txt",
)


def extract_trip_names(text):
    """Extract all unique trip names from arbitrary text."""
    matches = TRIP_PATTERN.findall(text)
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def update_trips_file(trips, output_path, append=False):
    """Write trip names to a trips file (one per line)."""
    existing = []
    if append and os.path.isfile(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing.append(line)

    merged = list(dict.fromkeys(existing + trips))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for t in merged:
            f.write(t + "\n")

    print("[feishu] Wrote {} trips to {}".format(len(merged), output_path))
    for t in merged:
        marker = " (new)" if t not in existing else ""
        print("  - {}{}".format(t, marker))
    return merged


def push_to_feishu(webhook_url, title, content_lines):
    """Send a message to Feishu webhook (card format)."""
    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "\n".join(content_lines),
                    },
                }
            ],
        },
    }
    data = json.dumps(card).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode("utf-8"))
        if result.get("code") == 0:
            print("[feishu] Message sent successfully")
        else:
            print("[feishu] Feishu API returned: {}".format(result))
    except urllib.error.URLError as e:
        print("[feishu] Failed to send: {}".format(e))


def format_report_for_feishu(report_path):
    """Parse SUMMARY_REPORT.md and format key metrics for Feishu card."""
    if not os.path.isfile(report_path):
        return "BEVCalib Calibration Results", ["Report not found: {}".format(report_path)]

    with open(report_path, "r") as f:
        content = f.read()

    lines = []
    lines.append("**Report**: `{}`".format(os.path.basename(os.path.dirname(report_path))))

    table_lines = [l for l in content.split("\n") if l.startswith("| YR-")]
    for tl in table_lines:
        parts = [p.strip() for p in tl.split("|") if p.strip()]
        if len(parts) >= 7:
            trip = parts[0]
            status = parts[1]
            frames = parts[2]
            rot = parts[3]
            icon = "✅" if status == "ok" else "❌"
            lines.append("{} **{}**: {} frames, Δ={}°".format(icon, trip, frames, rot))

    stats_match = re.findall(r"mean_rot_delta_deg.*?([0-9.]+)", content)
    if stats_match:
        lines.append("")
        lines.append("**Mean Rot Δ**: {}°".format(stats_match[0][:8]))

    ok_match = re.findall(r"trip_ok.*?(\d+)", content)
    fail_match = re.findall(r"trip_failed.*?(\d+)", content)
    if ok_match:
        lines.append("**OK/Failed**: {}/{}".format(
            ok_match[0], fail_match[0] if fail_match else "0"))

    return "BEVCalib Calibration Results", lines


def main():
    parser = argparse.ArgumentParser(description="Feishu <-> BEVCalib trip sync")
    sub = parser.add_subparsers(dest="command")

    p_parse = sub.add_parser("parse", help="Extract trip names from text")
    p_parse.add_argument("--text", type=str, help="Text containing trip names")
    p_parse.add_argument("--file", type=str, help="File containing trip names")
    p_parse.add_argument("--stdin", action="store_true", help="Read from stdin")
    p_parse.add_argument("--output", type=str, default=DEFAULT_TRIPS_FILE,
                         help="Output trips.txt path")
    p_parse.add_argument("--append", action="store_true",
                         help="Append to existing trips (default: replace)")

    p_push = sub.add_parser("push", help="Push calibration results to Feishu")
    p_push.add_argument("--report", type=str, required=True,
                        help="Path to SUMMARY_REPORT.md")
    p_push.add_argument("--webhook", type=str, default=DEFAULT_WEBHOOK)

    args = parser.parse_args()

    if args.command == "parse":
        text = ""
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, "r") as f:
                text = f.read()
        elif args.stdin:
            text = sys.stdin.read()
        else:
            parser.error("Provide --text, --file, or --stdin")

        trips = extract_trip_names(text)
        if not trips:
            print("[feishu] No trip names found in input")
            return

        print("[feishu] Found {} trip names:".format(len(trips)))
        for t in trips:
            print("  {}".format(t))
        update_trips_file(trips, args.output, append=args.append)

    elif args.command == "push":
        title, lines = format_report_for_feishu(args.report)
        print("[feishu] Sending to Feishu:")
        print("  Title: {}".format(title))
        for l in lines:
            print("  {}".format(l))
        push_to_feishu(args.webhook, title, lines)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
