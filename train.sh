#!/usr/bin/env bash
# SAM3 PEFT — train + evaluate with a YAML config.
#
# Usage:
#   ./train.sh configs/sam3_lora_potsdam.yaml
#   ./train.sh configs/sam3_linear_probing_potsdam.yaml

set -e
cd "$(dirname "$0")"

CONFIG="${1:?Usage: $0 CONFIG_YAML}"

echo "=== SAM3 PEFT Training ==="
echo "Config: $CONFIG"
echo ""

python3 train.py --config "$CONFIG"

EXP_DIR=$(python3 -c "
from utils.config import load_config
cfg = load_config('$CONFIG')
print(f'{cfg.experiment.output_dir}/{cfg.experiment.name}')
")

echo ""
echo "=== Evaluation ==="
python3 test.py --config "$CONFIG" --checkpoint "${EXP_DIR}/best.pth" --save_preds
