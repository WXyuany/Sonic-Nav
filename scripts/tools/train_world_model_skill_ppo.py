#!/usr/bin/env python3
"""Train the hybrid residual/recovery policy in the fast MuJoCo curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from sonic_world.rl.hybrid_ppo import HybridPPO, HybridRecurrentActorCritic, RolloutBatch, generalized_advantage
from sonic_world.rl.mujoco_skill_env import MujocoSkillLevelEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train world-model residual/recovery PPO in a vectorized MuJoCo skill curriculum.")
    parser.add_argument("--init-checkpoint", default="reports/policy_models/world_model_hybrid_ppo_curriculum_stage1_v4.pt")
    parser.add_argument("--output", default="reports/policy_models/world_model_hybrid_ppo_mujoco_skill_v0.pt")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = MujocoSkillLevelEnv(num_envs=max(1, args.num_envs), max_steps=max(4, args.horizon), seed=args.seed)
    device = torch.device(args.device)
    model = HybridRecurrentActorCritic().to(device)
    init = _path(args.init_checkpoint)
    if init.is_file():
        model.load_state_dict(torch.load(init, map_location=device, weights_only=False)["state_dict"])
    ppo = HybridPPO(model, learning_rate=args.learning_rate)
    entity, context = env.observe()
    history: list[dict[str, float]] = []
    for iteration in range(1, max(1, args.iterations) + 1):
        rollout = _collect(env, model, entity, context, horizon=args.horizon, device=device)
        entity, context = rollout.pop("next_observation")
        advantage, returns = generalized_advantage(rollout["reward"], rollout["value"], rollout["done"])
        batch = RolloutBatch(
            entity=rollout["entity"], context=rollout["context"], continuous_action=rollout["continuous"],
            recovery_action=rollout["recovery"], old_log_prob=rollout["log_prob"], advantage=advantage,
            returns=returns, valid=torch.ones_like(rollout["done"], dtype=torch.bool),
        )
        metrics = {}
        for _ in range(max(1, args.ppo_epochs)): metrics = ppo.update(batch)
        metrics.update({"iteration": float(iteration), "mean_reward": float(rollout["reward"].mean()), "success_rate": float(rollout["success"].float().mean()), "lift_rate": float(rollout["lifted"].float().mean())})
        history.append(metrics)
        if iteration == 1 or iteration % 10 == 0: print(json.dumps(metrics, sort_keys=True))
    output = _path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "sonic_world_model_hybrid_ppo_v0", "state_dict": model.state_dict(), "observation": "entity12x2+context24", "continuous_actions": 8, "recovery_actions": 5, "visual_context": False, "visual_deployment": {"status": "sim_skill_curriculum_only", "eligible_for_ab": False}, "training": {"method": "vectorized_mujoco_skill_ppo", "visual_token_source": "simulated_low_frequency_anchor_token", "init_checkpoint": str(init), "num_envs": args.num_envs, "horizon": args.horizon, "iterations": args.iterations, "history": history[-10:]}}, output)
    print(json.dumps({"output": str(output), **history[-1]}, sort_keys=True)); return 0


def _collect(env: MujocoSkillLevelEnv, model: HybridRecurrentActorCritic, entity: np.ndarray, context: np.ndarray, *, horizon: int, device: torch.device) -> dict[str, torch.Tensor | tuple[np.ndarray, np.ndarray]]:
    items: dict[str, list[torch.Tensor]] = {key: [] for key in ("entity", "context", "continuous", "recovery", "log_prob", "value", "reward", "done", "success", "lifted")}
    for _ in range(max(1, horizon)):
        e = torch.as_tensor(entity, dtype=torch.float32, device=device).unsqueeze(1); c = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad(): continuous, recovery, log_prob, value, _ = model.act(e, c)
        next_obs, reward, done, info = env.step(continuous[:, 0].cpu().numpy(), recovery[:, 0].cpu().numpy())
        for key, value_tensor in (("entity", e[:, 0]), ("context", c[:, 0]), ("continuous", continuous[:, 0]), ("recovery", recovery[:, 0]), ("log_prob", log_prob[:, 0]), ("value", value[:, 0]), ("reward", torch.as_tensor(reward, device=device)), ("done", torch.as_tensor(done, device=device)), ("success", torch.as_tensor(info.success, device=device)), ("lifted", torch.as_tensor(info.lifted, device=device))): items[key].append(value_tensor)
        entity, context = next_obs
    out: dict[str, torch.Tensor | tuple[np.ndarray, np.ndarray]] = {key: torch.stack(value, dim=1) for key, value in items.items()}
    out["next_observation"] = (entity, context); return out


def _path(value: str) -> Path:
    path = Path(value).expanduser(); return path if path.is_absolute() else REPO / path


if __name__ == "__main__":
    raise SystemExit(main())
