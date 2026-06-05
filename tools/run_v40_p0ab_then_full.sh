#!/usr/bin/env bash
# Run V40 GMP p0ab then full calibration sequentially (MEDW200, inject 2,2,2).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export no_proxy="*"
export https_proxy=""
export http_proxy=""

echo "[chain] $(date -Is) start p0ab"
python run_bag_calibration.py --config configs/calibration_v40_gmp_p0ab.yaml 2>&1 | tee /tmp/v40_p0ab_calib.log
echo "[chain] $(date -Is) p0ab done, start full"
python run_bag_calibration.py --config configs/calibration_v40_gmp_full.yaml 2>&1 | tee /tmp/v40_full_calib.log
echo "[chain] $(date -Is) all done"
