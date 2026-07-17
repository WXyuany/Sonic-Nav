#!/usr/bin/env python3
"""Advantage-weighted offline refinement of the task-space residual policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from sonic_world.rl.hybrid_ppo import HybridRecurrentActorCritic


RECOVERY = {None: 0, "perception_reobserve": 1, "navigation_micro_adjust": 2, "runtime_replan": 3, "manual_review": 4, "affordance_repair": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AWR-style residual candidate from physical episode transitions.")
    parser.add_argument("--dataset", default="reports/policy_data/physical_episode_residual_features_v0.jsonl")
    parser.add_argument("--checkpoint", default="reports/policy_models/world_model_hybrid_ppo_v0.pt")
    parser.add_argument("--output", default="reports/policy_models/world_model_hybrid_ppo_physical_aw_v0.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--visual-context", action="store_true", help="Enable Qwen/RGB-D feature slots after training on visual-anchor transitions.")
    parser.add_argument("--visual-gate-report", default="", help="Training gate report or VLM evaluation report used to label visual deployment eligibility.")
    parser.add_argument("--component", choices=("joint", "residual", "recovery"), default="joint", help="Train both heads or isolate one policy component.")
    parser.add_argument("--skill", action="append", default=[], help="Only train transitions from this primitive skill; repeatable.")
    parser.add_argument(
        "--effect-source",
        action="append",
        default=[],
        help="Only keep transitions with one of these terminal effect evidence sources; repeatable.",
    )
    parser.add_argument(
        "--positive-effect-only",
        action="store_true",
        help="Train only terminal primitives with verified passed effect evidence. Recommended for sparse teacher-assisted skill data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = _path(args.dataset)
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if isinstance(row.get("observation"), dict) and isinstance((row.get("policy") or {}).get("metadata"), dict)]
    skills = {str(item) for item in args.skill if str(item)}
    if skills:
        rows = [row for row in rows if str((row.get("primitive") or {}).get("skill_name") or "") in skills]
    effect_sources = {str(item) for item in args.effect_source if str(item)}
    if effect_sources:
        rows = [row for row in rows if str((row.get("outcome") or {}).get("effect_source") or "") in effect_sources]
    candidate_count = len(rows)
    if args.positive_effect_only:
        rows = [row for row in rows if _positive_effect(row)]
    if len(rows) < args.min_samples:
        selection = "positive effect" if args.positive_effect_only else "feature"
        raise SystemExit(f"need at least {args.min_samples} {selection} transitions, found {len(rows)}")
    entity = torch.tensor([row["observation"]["entity"] for row in rows], dtype=torch.float32).unsqueeze(1)
    context = torch.tensor([row["observation"]["context"] for row in rows], dtype=torch.float32).unsqueeze(1)
    action = torch.tensor([row["policy"]["metadata"]["residual"] for row in rows], dtype=torch.float32).unsqueeze(1)
    recovery = torch.tensor([RECOVERY.get((row["outcome"] or {}).get("recovery_handler"), 0) for row in rows], dtype=torch.long).unsqueeze(1)
    reward = torch.tensor([float((row["outcome"] or {}).get("reward") or 0.0) for row in rows], dtype=torch.float32).unsqueeze(1)
    source = torch.load(_path(args.checkpoint), map_location="cpu", weights_only=False)
    model = HybridRecurrentActorCritic(); model.load_state_dict(source["state_dict"]); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    weights = torch.softmax(reward * 2.0, dim=0).detach() * len(rows)
    for _ in range(max(1, args.epochs)):
        normal, categorical, value, _hidden = model(entity, context)
        action_loss = (((normal.mean.tanh() - action) ** 2).mean(-1) * weights).mean()
        recovery_loss = F.cross_entropy(categorical.logits.reshape(-1, 5), recovery.reshape(-1), reduction="none").reshape_as(reward)
        value_loss = F.mse_loss(value, reward)
        residual_term = action_loss if args.component in {"joint", "residual"} else torch.zeros((), dtype=action_loss.dtype)
        recovery_term = (recovery_loss * weights).mean() if args.component in {"joint", "recovery"} else torch.zeros((), dtype=action_loss.dtype)
        loss = residual_term + 0.25 * recovery_term + 0.1 * value_loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    output = _path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    visual_gate = _visual_gate(args.visual_gate_report, visual_context=bool(args.visual_context))
    training = {
        "method": "advantage_weighted_offline_physical", "component": args.component,
        "source_checkpoint": str(_path(args.checkpoint)), "dataset": str(dataset_path), "dataset_sha256": _sha256(dataset_path),
        "sample_count": len(rows), "epochs": args.epochs, "mean_reward": float(reward.mean()),
        "skills": sorted(skills) if skills else sorted({str((row.get("primitive") or {}).get("skill_name") or "unknown") for row in rows}),
        "sample_selection": {
            "positive_effect_only": bool(args.positive_effect_only),
            "effect_sources": sorted(effect_sources),
            "candidate_count": candidate_count,
            "selected_count": len(rows),
            "teacher_assisted_count": sum(bool((row.get("outcome") or {}).get("teacher_assisted")) for row in rows),
        },
        "recovery_label_counts": {name: int((recovery == index).sum()) for index, name in enumerate(("continue", "reobserve", "micro_adjust", "replan", "abort"))},
    }
    torch.save({"schema": "sonic_world_model_hybrid_ppo_v0", "state_dict": model.state_dict(), "observation": "entity12x2+context24", "continuous_actions": 8, "recovery_actions": 5, "visual_context": bool(args.visual_context), "visual_deployment": visual_gate, "training": training}, output)
    print(json.dumps({"output": str(output), "sample_count": len(rows), "component": args.component, "visual_deployment": visual_gate["status"], "mean_reward": round(float(reward.mean()), 4), "final_loss": round(float(loss.detach()), 6)}, sort_keys=True))
    return 0


def _positive_effect(row: dict) -> bool:
    """Return true only for a successful primitive with passed effect evidence."""
    primitive = row.get("primitive") if isinstance(row.get("primitive"), dict) else {}
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    return str(primitive.get("status") or "").lower() == "success" and bool(outcome.get("effect_passed"))


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visual_gate(raw: str, *, visual_context: bool) -> dict[str, object]:
    if not visual_context:
        return {"status": "not_requested", "eligible_for_ab": True, "report": None}
    report_path = _path(raw) if raw else None
    payload: dict[str, object] = {}
    if report_path is not None and report_path.is_file():
        value = json.loads(report_path.read_text(encoding="utf-8"))
        payload = value if isinstance(value, dict) else {}
    decision = str(payload.get("decision") or "")
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    passed = decision == "eligible_for_ab" or bool(gate.get("passed"))
    return {
        "status": "eligible_for_ab" if passed else "shadow_training_only",
        "eligible_for_ab": passed,
        "report": str(report_path) if report_path is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
