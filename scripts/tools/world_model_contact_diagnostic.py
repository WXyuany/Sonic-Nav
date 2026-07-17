#!/usr/bin/env python3
"""Summarize physical grasp-contact evidence for controlled residual sweeps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a contact/IK diagnostic report from executor-backed episode logs.")
    parser.add_argument("--input", action="append", required=True, help="Episode JSONL file or directory; repeatable.")
    parser.add_argument("--skill", default="manip.side_grasp")
    parser.add_argument("--output", default="reports/diagnostics/contact_latest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for path in _paths(args.input) for row in _rows(path, str(args.skill))]
    report = summarize(rows, skill=str(args.skill))
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"contact_diagnostic=skill={args.skill} attempts={report['summary']['attempt_count']} passed={report['summary']['passed_count']}")
    print(_relative(output))
    return 0


def summarize(rows: list[dict[str, Any]], *, skill: str) -> dict[str, Any]:
    passed = [row for row in rows if row["passed"]]
    failed = [row for row in rows if not row["passed"]]
    return {
        "schema": "sonic_world_model_contact_diagnostic_v0",
        "skill_name": skill,
        "summary": {
            "attempt_count": len(rows), "passed_count": len(passed), "failed_count": len(failed),
            "success_rate": round(len(passed) / len(rows), 4) if rows else 0.0,
            "contact_count_median": _median(rows, "contact_count"),
            "ik_error_median": _median(rows, "servo_ik_error"),
        },
        "by_outcome": {"passed": _group(passed), "failed": _group(failed)},
        "samples": rows,
    }


def _rows(path: Path, skill: str) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") != "primitive_status" or event.get("skill_name") != skill or event.get("status") != "success":
            continue
        evidence = event.get("effect_evidence") if isinstance(event.get("effect_evidence"), dict) else {}
        effects = evidence.get("effects") if isinstance(evidence.get("effects"), dict) else {}
        contact = effects.get("object_contact_ready") if isinstance(effects.get("object_contact_ready"), dict) else {}
        metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
        ik = metrics.get("ik") if isinstance(metrics.get("ik"), dict) else {}
        params = metrics.get("command_params") if isinstance(metrics.get("command_params"), dict) else {}
        runtime = metrics.get("runtime_overrides") if isinstance(metrics.get("runtime_overrides"), dict) else {}
        out.append({
            "source_log": _relative(path), "stamp": event.get("stamp"), "passed": bool(evidence.get("passed")),
            "contact_count": _number(contact.get("contact_count")), "servo_ik_error": _number(ik.get("servo_ik_error")),
            "close_ratio": _number(runtime.get("close_ratio")),
            "contact_x_delta_m": _number(params.get("contact_x_delta_m")),
            "contact_z_delta_m": _number(params.get("contact_z_delta_m")),
        })
    return out


def _group(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return {"count": len(rows), "contact_count_median": _median(rows, "contact_count"), "ik_error_median": _median(rows, "servo_ik_error"), "close_ratio_median": _median(rows, "close_ratio")}


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return round(float(median(values)), 6) if values else None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paths(inputs: list[str]) -> list[Path]:
    paths = []
    for raw in inputs:
        path = _path(raw)
        if path.is_file() and path.suffix == ".jsonl":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
    return list(dict.fromkeys(paths))


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
