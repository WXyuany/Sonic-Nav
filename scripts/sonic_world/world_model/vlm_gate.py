from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_passing_vlm_gate(path: str | Path) -> dict[str, Any]:
    """Load an anchor-evaluation report and reject anything that did not pass."""

    report_path = Path(path).expanduser()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read VLM gate report {report_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "sonic_vlm_anchor_eval_report_v0":
        raise ValueError(f"{report_path} is not a Sonic VLM anchor-evaluation report")
    gate = payload.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        failed = gate.get("failed_checks") if isinstance(gate, dict) else None
        detail = ", ".join(str(item) for item in failed) if isinstance(failed, list) else "unknown checks"
        raise ValueError(f"VLM gate did not pass: {detail}")
    return payload
