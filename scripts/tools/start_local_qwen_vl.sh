#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv_qwen_vl/bin/python"
MODEL="${QWEN_VL_LOCAL_MODEL:-$ROOT/models/Qwen2.5-VL-3B-Instruct}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing local Qwen-VL environment: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing local Qwen-VL model: $MODEL" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/scripts/tools/local_qwen_vl_server.py" \
  --model "$MODEL" \
  --host "${QWEN_VL_HOST:-127.0.0.1}" \
  --port "${QWEN_VL_PORT:-8000}" \
  --max-new-tokens "${QWEN_VL_MAX_NEW_TOKENS:-512}" \
  --device "${QWEN_VL_DEVICE:-cuda}" \
  --gpu-memory-gib "${QWEN_VL_GPU_MEMORY_GIB:-0}"
