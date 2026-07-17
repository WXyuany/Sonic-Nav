#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/.venv_qwen_vl/bin/python"
MODEL="${GROUNDING_DINO_LOCAL_MODEL:-$ROOT/models/grounding-dino-tiny}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing local vision environment: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "Missing local Grounding DINO model: $MODEL" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/scripts/tools/local_grounding_dino_server.py" \
  --model "$MODEL" \
  --host "${GROUNDING_DINO_HOST:-127.0.0.1}" \
  --port "${GROUNDING_DINO_PORT:-8001}"
