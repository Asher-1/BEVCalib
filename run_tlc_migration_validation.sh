#!/bin/bash
# TLC 迁移效果验证流水线
# 1) 对已有 v54a_lsp vs v54a_nolsp ckpt 做泛化评估
# 2) 训练 v54a_lsp_smoke（15ep，验证 LSP 可稳定训练）
# 3) 输出对比摘要
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bevcalib310

LOG_ROOT="logs/tlc_migration_validation"
mkdir -p "$LOG_ROOT"

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(_ts)] === Step 1/2: V54 TLC ablation generalization eval (LSP vs no-LSP) ===" | tee "$LOG_ROOT/pipeline.log"
bash run_v54_eval.sh --gate-eval 2>&1 | tee "$LOG_ROOT/step1_v54_eval.log"

echo "" | tee -a "$LOG_ROOT/pipeline.log"
echo "[$(_ts)] === Step 2/2: v54a_lsp_smoke training (15ep, LSP+MGDA+RigC) ===" | tee -a "$LOG_ROOT/pipeline.log"
bash batch_train.sh --skip-pattern "full|nolsp" configs/v54a_lsp_cf_bev_r.yaml 2>&1 | tee "$LOG_ROOT/step2_v54a_smoke_train.log"

echo "" | tee -a "$LOG_ROOT/pipeline.log"
echo "[$(_ts)] === Pipeline complete ===" | tee -a "$LOG_ROOT/pipeline.log"
echo "  Eval report: logs/evaluations/generalization_v54/GENERALIZATION_REPORT.md" | tee -a "$LOG_ROOT/pipeline.log"
echo "  Gate acceptance: logs/evaluations/generalization_v54/DEPLOY_GATE_ACCEPTANCE.md" | tee -a "$LOG_ROOT/pipeline.log"
echo "  Smoke train log: logs/all_training_data/model_small_5deg_v54a_lsp_smoke/train.log" | tee -a "$LOG_ROOT/pipeline.log"
