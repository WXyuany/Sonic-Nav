#!/usr/bin/env python3
"""Gate visual-policy candidates before they can enter physical A/B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen/RGB-D quality and visual transition volume for policy A/B eligibility.")
    parser.add_argument("--vlm-report", required=True)
    parser.add_argument("--visual-summary", required=True)
    parser.add_argument("--min-visual-transitions", type=int, default=100)
    parser.add_argument("--output", default="reports/policy_data/visual_training_gate.json")
    parser.add_argument("--strict", action="store_true", help="Return non-zero while visual data remains shadow-only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vlm, summary = _read(_path(args.vlm_report)), _read(_path(args.visual_summary))
    report = evaluate_gate(vlm, summary, min_visual_transitions=max(1, int(args.min_visual_transitions)))
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"visual_training_gate={report['decision']} checks={sum(report['checks'].values())}/{len(report['checks'])}")
    print(_relative(output))
    return 0 if report["decision"] == "eligible_for_ab" or not args.strict else 2


def evaluate_gate(vlm: dict[str, Any], summary: dict[str, Any], *, min_visual_transitions: int) -> dict[str, Any]:
    gate = vlm.get("gate") if isinstance(vlm.get("gate"), dict) else {}
    count = int(summary.get("visual_transition_count") or 0)
    checks = {"vlm_anchor_quality": bool(gate.get("passed")), "visual_transition_volume": count >= min_visual_transitions}
    return {
        "schema": "sonic_world_model_visual_training_gate_v0",
        "decision": "eligible_for_ab" if all(checks.values()) else "shadow_training_only",
        "checks": checks,
        "minimum_visual_transitions": min_visual_transitions,
        "visual_transition_count": count,
        "vlm_failed_checks": list(gate.get("failed_checks") or []),
        "sources": {"vlm_report_schema": vlm.get("schema"), "visual_summary_schema": summary.get("schema")},
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
