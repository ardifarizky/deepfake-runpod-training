#!/usr/bin/env bash
# Run from the RunPod project directory after the data volume is mounted.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/full_training.json"
DATA=/workspace/data
OUT=/workspace/outputs
mkdir -p "$OUT" /workspace/logs

python "$ROOT/scripts/validate_data.py" \
  --index "$DATA/index_v2.csv" \
  --df40-external "$DATA/df40_external.csv" \
  --sdxl-external "$DATA/sdxl_external.csv" \
  --check-files

for MODEL in efficientnet swin; do
  echo "START $MODEL $(date -Is)"
  python "$ROOT/scripts/train_full.py" --config "$CONFIG" --model "$MODEL" --resume \
    2>&1 | tee -a "/workspace/logs/${MODEL}.log"
  echo "COMPLETE $MODEL $(date -Is)"
done

python "$ROOT/scripts/build_report.py" --results "$OUT"
echo "ALL_COMPLETE $(date -Is)"
