#!/usr/bin/env bash
# V41 三组训练收敛监控
#   bash tools/v41_monitor.sh          # 当前进度
#   bash tools/v41_monitor.sh ep20     # ep20 首批 KPI 对照（三组均到 ep20 后才有完整表）
set -eu
ROOT="${BEVCALIB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_BASE="$ROOT/logs/all_training_data"
REPORT="$ROOT/logs/v41_ep20_compare_report.md"

declare -A EXP_NAMES=(
    [a]="v41_gmp_a_v32_baseline"
    [b]="v41_gmp_b_v32_jacloss"
    [c]="v41_gmp_c_v32_jacloss_match"
)

log_path() { echo "$LOG_BASE/model_small_5deg_${1}/train.log"; }

extract_ep20() {
    local log="$1"
    local medw jac dual gate
    medw=$(grep -F "Epoch [20/200], MEDW200" "$log" | head -1 || true)
    jac=$(grep -F "Epoch [20/200], Jacobian" "$log" | head -1 || true)
    dual=$(grep -F "Dual gate" "$log" | grep -F "ep 20" | head -1 || true)
    if [[ -z "$dual" ]]; then
        dual=$(grep -E "Dual gate (PASS|FAIL)" "$log" | awk '/Epoch \[20\/200\]|ep 20/ || NR==0' | head -1 || true)
    fi
    if [[ -z "$dual" ]]; then
        # 取 ep20 附近第一条 dual gate（MEDW/Jacobian 之后）
        dual=$(grep -E "★ Dual gate PASS|○ Dual gate FAIL" "$log" | sed -n '1p' || true)
        if [[ $(grep -c "Epoch \[20/200\], MEDW200" "$log") -eq 0 ]]; then
            dual=""
        fi
    fi
    gate="WAIT"
    if echo "$dual" | grep -q "PASS"; then gate="PASS"; fi
    if echo "$dual" | grep -q "FAIL"; then gate="FAIL"; fi
    printf '%s\n%s\n%s\n%s\n' "$medw" "$jac" "$dual" "$gate"
}

print_exp() {
    local name="$1"
    local log
    log=$(log_path "$name")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ ! -f "$log" ]]; then
        echo "  [MISSING] $log"
        return
    fi
    if grep -q "训练启动, 参数配置:" "$log"; then
        grep -o "correspondence_loss_weight=[^,]*" "$log" | tail -1 | sed 's/^/  /' || true
        grep -o "consistency_loss_start_epoch=[0-9]*" "$log" | tail -1 | sed 's/^/  /' || true
        grep -o "jacobian_loss_start_epoch=[0-9]*" "$log" | tail -1 | sed 's/^/  /' || true
        grep -o "jacobian_loss_probe_deg=[0-9.]*" "$log" | tail -1 | sed 's/^/  /' || true
        grep -o "pretrain_ckpt=[^,]*" "$log" | tail -1 | sed 's/^/  /' || true
    fi
    echo "  架构:"
    grep "GeoMatchProjCalib" "$log" | tail -1 | sed 's/^/  /' || true
    echo "  进度:"
    tail -1 "$log" | sed 's/^/  /'
    echo "  最近 KPI:"
    grep -E "MEDW200 \(val reuse\)|Jacobian ±|Dual gate (PASS|FAIL)|★ Dual gate|○ Dual gate|New best MEDW|New best dual" "$log" | tail -5 | sed 's/^/  /' || echo "  (尚无 eval — 首次 KPI @ ep20)"
    echo
}

ep20_compare() {
    local ready=0
    echo "# V41 ep20 首批 KPI 对照"
    echo ""
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "红线: max(R,P,Y) MEDW200 < 0.10° 且 Jacobian R/P/Y/overall 均 > 0.85"
    echo ""

    for key in a b c; do
        local log ep
        log=$(log_path "${EXP_NAMES[$key]}")
        ep=$(grep -oE "Epoch \[[0-9]+/200\]" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1 || echo "0")
        if [[ ! -f "$log" ]]; then
            echo "- **${EXP_NAMES[$key]}**: log 不存在"
            continue
        fi
        if [[ "$ep" -lt 20 ]]; then
            echo "- **${EXP_NAMES[$key]}**: 当前 ep${ep}/200，未到 ep20"
            continue
        fi
        if ! grep -q "Epoch \[20/200\], MEDW200" "$log"; then
            echo "- **${EXP_NAMES[$key]}**: ep≥20 但 ep20 MEDW 行未找到（eval 进行中？）"
            continue
        fi
        ready=$((ready + 1))
    done

    if [[ "$ready" -lt 3 ]]; then
        echo ""
        echo "> 三组尚未全部完成 ep20 eval，完整对照表待补齐。"
        return 1
    fi

    echo "| 组 | max(R,P,Y)° | R | P | Y | Jac overall | Jac R | P | Y | Dual gate |"
    echo "|----|-------------|---|---|---|-------------|-------|---|---|-----------|"

    for key in a b c; do
        local log medw jac dual gate label
        log=$(log_path "${EXP_NAMES[$key]}")
        label="${key^^}"
        mapfile -t lines < <(extract_ep20 "$log")
        medw="${lines[0]}"
        jac="${lines[1]}"
        dual="${lines[2]}"
        gate="${lines[3]}"

        # parse MEDW: max(R,P,Y)=X (R:a P:b Y:c
        local mx r p y jo jr jp jy
        mx=$(echo "$medw" | grep -oE 'max\(R,P,Y\)=[0-9.]+' | cut -d= -f2 || echo "?")
        r=$(echo "$medw" | grep -oE 'R:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")
        p=$(echo "$medw" | grep -oE 'P:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")
        y=$(echo "$medw" | grep -oE 'Y:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")
        jo=$(echo "$jac" | grep -oE 'Overall=[0-9.]+' | cut -d= -f2 || echo "?")
        jr=$(echo "$jac" | grep -oE 'R:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")
        jp=$(echo "$jac" | grep -oE 'P:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")
        jy=$(echo "$jac" | grep -oE 'Y:[0-9.]+' | head -1 | cut -d: -f2 || echo "?")

        echo "| **${label}** ${EXP_NAMES[$key]} | ${mx} | ${r} | ${p} | ${y} | ${jo} | ${jr} | ${jp} | ${jy} | **${gate}** |"
    done

    echo ""
    echo "## 原始 log 摘录"
    for key in a b c; do
        local log
        log=$(log_path "${EXP_NAMES[$key]}")
        echo "### ${EXP_NAMES[$key]}"
        grep -E "Epoch \[20/200\], MEDW200|Epoch \[20/200\], Jacobian|Dual gate" "$log" | head -3 | sed 's/^/    /'
        echo ""
    done
    return 0
}

if [[ "${1:-}" == "ep20" ]]; then
    if ep20_compare | tee "$REPORT"; then
        echo ""
        echo "报告已写入: $REPORT"
    else
        echo ""
        echo "部分实验未到 ep20，报告为进度摘要: $REPORT"
    fi
    exit 0
fi

echo "V41 训练监控  $(date '+%Y-%m-%d %H:%M:%S')"
echo
print_exp "v41_gmp_a_v32_baseline"
print_exp "v41_gmp_b_v32_jacloss"
print_exp "v41_gmp_c_v32_jacloss_match"
echo "里程碑: ep10 V32+jacloss | ep20 首批 KPI | ep200 CONVERGENCE_REPORT"
echo "ep20 对照: bash tools/v41_monitor.sh ep20"
