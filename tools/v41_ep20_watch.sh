#!/usr/bin/env bash
# 后台轮询，三组均完成 ep20 eval 后写报告
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/logs/v41_ep20_compare_report.md"
INTERVAL="${V41_WATCH_INTERVAL:-300}"

while true; do
    if bash "$ROOT/tools/v41_monitor.sh" ep20 2>/dev/null | grep -q "| **A**"; then
        echo "[$(date '+%F %T')] ep20 对照完成 → $REPORT"
        exit 0
    fi
    sleep "$INTERVAL"
done
