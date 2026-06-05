#!/usr/bin/env bash
# V41 ep11 里程碑监控：ep1 val+Jacobian → ep10 → ep11 train（Jac+consistency 关键验收）
#   bash tools/v41_ep11_watch.sh          # 前台轮询，三组均过 ep11 train 后退出
#   nohup bash tools/v41_ep11_watch.sh &  # 后台
set -eu
ROOT="${BEVCALIB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_BASE="$ROOT/logs/all_training_data"
REPORT="$ROOT/logs/v41_ep11_watch_report.md"
INTERVAL="${V41_EP11_INTERVAL:-120}"

declare -A EXP=(
    [a]="v41_gmp_a_v32_baseline"
    [b]="v41_gmp_b_v32_jacloss"
    [c]="v41_gmp_c_v32_jacloss_match"
)

log_path() { echo "$LOG_BASE/model_small_5deg_${1}/train.log"; }

# 返回: ep crash_msg ep1_val ep10_train ep11_train status
# status: WAIT|RUN|FAIL|PASS
check_exp() {
    local name="$1"
    local log
    log=$(log_path "$name")
    if [[ ! -f "$log" ]]; then
        echo "0 '' 0 0 0 WAIT"
        return
    fi
    local ep crash ep1 ep10 ep11
    ep=$(grep -oE "Epoch \[[0-9]+/200\]" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1 || echo "0")
    crash=""
    if grep -q "Traceback (most recent call last):" "$log"; then
        crash=$(grep -E "RuntimeError:|TypeError:" "$log" | tail -1 || echo "Traceback")
    fi
    ep1=$(grep -c "Epoch \[1/200\], Val\[±5.0°\]" "$log" || true)
    ep10=$(grep -c "Epoch \[10/200\], Train Pose Error" "$log" || true)
    ep11=$(grep -c "Epoch \[11/200\], Train Pose Error" "$log" || true)

    local st="RUN"
    [[ "$ep" == "0" ]] && st="WAIT"
    if [[ -n "$crash" ]]; then
        st="FAIL"
    elif [[ "$ep11" -ge 1 ]]; then
        st="PASS"
    elif [[ "$ep" -ge 1 && "$ep" -le 11 ]]; then
        # 无 Traceback 但进程已退出且 log 超过 5min 未更新 → STALE（常见于多组抢 GPU）
        local age proc
        age=$(($(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0)))
        proc=$(pgrep -cf "model_small_5deg_${name}" 2>/dev/null || echo 0)
        if [[ "$proc" -eq 0 && "$age" -gt 300 ]]; then
            st="STALE"
            [[ -z "$crash" ]] && crash="log stale ${age}s, no train process"
        fi
    fi
    echo "$ep|$crash|$ep1|$ep10|$ep11|$st"
}

write_report() {
    local ts="$1"
    {
        echo "# V41 ep11 监控报告"
        echo ""
        echo "更新时间: $ts"
        echo ""
        echo "关键验收: ep11 首个 batch 含 consistency + jacobian 监督，无 inplace/cuda Traceback"
        echo ""
        echo "| 组 | 当前 ep | ep1 Val | ep10 Train | ep11 Train | 状态 | 最近错误 |"
        echo "|----|---------|---------|------------|------------|------|----------|"
        for key in a b c; do
            IFS='|' read -r ep crash ep1 ep10 ep11 st <<< "$(check_exp "${EXP[$key]}")"
            local crash_short="${crash:0:60}"
            [[ -z "$crash_short" ]] && crash_short="-"
            echo "| **${key^^}** ${EXP[$key]} | ${ep}/200 | $([[ $ep1 -ge 1 ]] && echo OK || echo -) | $([[ $ep10 -ge 1 ]] && echo OK || echo -) | $([[ $ep11 -ge 1 ]] && echo OK || echo -) | **${st}** | ${crash_short} |"
        done
        echo ""
        echo "## 各组最后一行 log"
        for key in a b c; do
            local log
            log=$(log_path "${EXP[$key]}")
            echo "### ${EXP[$key]}"
            if [[ -f "$log" ]]; then
                tail -1 "$log" | sed 's/^/    /'
            else
                echo "    (missing)"
            fi
            echo ""
        done
    } > "$REPORT"
}

all_pass=0
while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    write_report "$ts"
    all_pass=1
    for key in a b c; do
        IFS='|' read -r _ _ _ _ _ st <<< "$(check_exp "${EXP[$key]}")"
        if [[ "$st" != "PASS" ]]; then
            all_pass=0
        fi
        if [[ "$st" == "FAIL" || "$st" == "STALE" ]]; then
            echo "[$ts] ${st} ${EXP[$key]} — 见 $REPORT"
        fi
    done
    echo "[$ts] ep11 watch → $REPORT (A/B/C pass=$all_pass)"
    bash "$ROOT/tools/v41_monitor.sh" 2>/dev/null | head -20 || true
    if [[ "$all_pass" -eq 1 ]]; then
        echo "[$ts] 三组均已通过 ep11 train，监控结束"
        exit 0
    fi
    sleep "$INTERVAL"
done
