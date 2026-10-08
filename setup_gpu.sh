#!/usr/bin/env bash
# 兼容旧命令；实际安装流程已经迁移到 Python。
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-python3}" setup_gpu.py --cuda "${1:-cu126}"
