#!/usr/bin/env bash
# SAM LoRA — train + evaluate with a YAML config.
#
# Usage:
#   ./train.sh configs/potsdam.yaml
#   ./train.sh configs/vaihingen.yaml
#
# Optional overrides:
#   ./train.sh configs/potsdam.yaml training.epochs=100 training.batch_size=4

set -e
cd "$(dirname "$0")"

CONFIG="${1:?Usage: $0 CONFIG_YAML [overrides...]}"
shift
OVERRIDES="$@"

echo "=== SAM LoRA Training ==="
echo "Config: $CONFIG"
[ -n "$OVERRIDES" ] && echo "Overrides: $OVERRIDES"
echo ""

python3 train.py --config "$CONFIG" --override $OVERRIDES

# Extract experiment name from config for eval
EXP_DIR=$(python3 -c "
from utils.config import load_config
cfg = load_config('$CONFIG')
print(f'{cfg.experiment.output_dir}/{cfg.experiment.name}')
")

echo ""
echo "=== Evaluation ==="
python3 test.py --config "$CONFIG" --checkpoint "${EXP_DIR}/best.pth" --save_preds
