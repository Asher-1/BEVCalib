#!/bin/bash
set -euo pipefail

ROOT="/mnt/drtraining/user/dahailu/code/BEVCalib"
CONFIG="configs/v64b_clean_mount_pitch_cf_bev_r.yaml"
NEEDED_GPUS="${NEEDED_GPUS:-8}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SEC="${POLL_SEC:-180}"

cd "$ROOT"

while true; do
  ts="$(date '+%F %T')"
  gpus="$(
    /opt/conda/envs/bevcalib310/bin/python - <<PY
import subprocess

needed = int("${NEEDED_GPUS}")
min_free = int("${MIN_FREE_MB}")
max_util = int("${MAX_UTIL}")
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
chosen = []
for line in out.strip().splitlines():
    idx_s, free_s, util_s = [part.strip() for part in line.split(",")]
    if int(free_s) >= min_free and int(util_s) <= max_util:
        chosen.append(idx_s)
if len(chosen) >= needed:
    print(",".join(chosen[:needed]))
PY
  )"

  if [ -n "$gpus" ]; then
    echo "[$ts] selected GPUs $gpus for V64b full (need ${NEEDED_GPUS}, free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%)"
    CUDA_VISIBLE_DEVICES="$gpus" bash batch_train.sh --force --skip-pattern smoke "$CONFIG"
    exit $?
  fi

  echo "[$ts] waiting for ${NEEDED_GPUS} free GPUs (free>=${MIN_FREE_MB}MB util<=${MAX_UTIL}%)"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
  sleep "$POLL_SEC"
done
