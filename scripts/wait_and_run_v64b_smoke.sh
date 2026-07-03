#!/bin/bash
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
CONFIG="configs/v64b_clean_mount_pitch_cf_bev_r.yaml"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SEC="${POLL_SEC:-120}"

cd "$ROOT"

while true; do
  ts="$(date '+%F %T')"
  gpu="$(
    /opt/conda/envs/bevcalib310/bin/python - <<PY
import subprocess

min_free = int("${MIN_FREE_MB}")
max_util = int("${MAX_UTIL}")
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
for line in out.strip().splitlines():
    idx_s, free_s, util_s = [part.strip() for part in line.split(",")]
    if int(free_s) >= min_free and int(util_s) <= max_util:
        print(idx_s)
        break
PY
  )"

  if [ -n "$gpu" ]; then
    echo "[$ts] selected GPU $gpu for V64b smoke (free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%)"
    CUDA_VISIBLE_DEVICES="$gpu" bash batch_train.sh --force --skip-pattern full "$CONFIG"
    exit $?
  fi

  echo "[$ts] waiting for free GPU (need free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%)"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
  sleep "$POLL_SEC"
done
