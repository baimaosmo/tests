#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_data.py
RUN_DIR="runs/demo_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python train.py "$@" --output-dir "$RUN_DIR"
.venv/bin/python predict.py --checkpoint "$RUN_DIR/best.pt" --output-dir "$RUN_DIR/predictions"
echo "完成！模型、指标和预测图位于 $RUN_DIR"
