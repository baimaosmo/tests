#!/usr/bin/env bash
# 兼容旧命令；实际入口已经迁移到 Python。
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON_BIN:-python}" run_gpu.py "$@"
