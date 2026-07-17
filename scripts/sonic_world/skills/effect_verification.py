from __future__ import annotations

from typing import Any


def verify_backend_effects(
    action: dict[str, Any],
    status: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Verify a backend result against the effects declared by a dispatch action."""

    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    effects = [str(item) for item in metadata.get("effects", []) if str(item)]
    evidence = status.get("effect_evidence")
    if not isinstance(evidence, dict):
        return False, "terminal success did not include structured effect_evidence", {"effects": effects}

    records = evidence.get("effects") if isinstance(evidence.get("effects"), dict) else {}
    failed: list[str] = []
    missing: list[str] = []
    details: dict[str, Any] = {}
    for effect in effects:
        record = records.get(effect)
        if isinstance(record, dict):
            passed = bool(record.get("passed"))
            details[effect] = record
        elif isinstance(record, bool):
            passed = record
            details[effect] = {"passed": passed}
        else:
            missing.append(effect)
            continue
        if not passed:
            failed.append(effect)

    overall = evidence.get("passed")
    if effects:
        passed = not missing and not failed and overall is not False
    else:
        passed = bool(overall)
    reasons = []
    if missing:
        reasons.append(f"missing effects: {', '.join(missing)}")
    if failed:
        reasons.append(f"failed effects: {', '.join(failed)}")
    if not reasons and not passed:
        reasons.append(str(evidence.get("reason") or "backend effect verification failed"))
    return passed, "; ".join(reasons), {
        "effect_verified": passed,
        "declared_effects": effects,
        "effect_details": details,
    }


def effect_evidence(
    effects: dict[str, bool | dict[str, Any]],
    *,
    source: str,
    reason: str = "",
) -> dict[str, Any]:
    normalized: dict[str, dict[str, Any]] = {}
    for name, record in effects.items():
        normalized[str(name)] = dict(record) if isinstance(record, dict) else {"passed": bool(record)}
    passed = bool(normalized) and all(bool(record.get("passed")) for record in normalized.values())
    return {
        "schema": "sonic_skill_effect_evidence_v0",
        "passed": passed,
        "source": source,
        "reason": reason,
        "effects": normalized,
    }
