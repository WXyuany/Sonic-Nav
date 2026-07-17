"""Vectorized task-space residual curriculum for custom Hybrid PPO.

It is deliberately a fast world-model curriculum, not a replacement for the
frozen SONIC controller. Rewards and terminal conditions mirror the task
oracle; candidates must still pass the held-out MuJoCo/SONIC evaluation stack.
"""
from __future__ import annotations

import torch

from .hybrid_ppo import CONTEXT_DIM, ENTITY_DIM


class WorldModelResidualEnv:
    def __init__(self, num_envs: int = 256, *, device: str = "cpu", max_steps: int = 64, seed: int = 7):
        self.num_envs, self.device, self.max_steps = num_envs, torch.device(device), max_steps
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.robot = torch.zeros(num_envs, 3, device=self.device)  # x, y, yaw in map
        self.object = torch.zeros(num_envs, 3, device=self.device)
        self.target = torch.zeros(num_envs, 3, device=self.device)
        self.held = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.recovery_count = torch.zeros(num_envs, device=self.device)
        self.previous_distance = torch.zeros(num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        count = len(ids)
        self.robot[ids] = 0.0
        self.robot[ids, 1] = torch.empty(count, device=self.device).uniform_(-0.18, 0.18, generator=self.generator)
        self.object[ids, 0] = torch.empty(count, device=self.device).uniform_(0.85, 1.35, generator=self.generator)
        self.object[ids, 1] = torch.empty(count, device=self.device).uniform_(-0.30, 0.14, generator=self.generator)
        self.object[ids, 2] = 0.84
        self.target[ids, 0] = self.object[ids, 0] + torch.empty(count, device=self.device).uniform_(-0.05, 0.12, generator=self.generator)
        self.target[ids, 1] = torch.empty(count, device=self.device).uniform_(-0.08, 0.30, generator=self.generator)
        self.target[ids, 2] = 0.84
        self.held[ids] = False
        self.recovery_count[ids] = 0.0
        self.step_count[ids] = 0
        self.previous_distance[ids] = torch.linalg.vector_norm(self.object[ids, :2] - self.robot[ids, :2], dim=-1)
        return self.observe()

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        entity = torch.zeros(self.num_envs, 2, ENTITY_DIM, device=self.device)
        entity[:, 0, :3] = self.object - torch.cat((self.robot[:, :2], torch.zeros_like(self.robot[:, :1])), dim=-1)
        entity[:, 1, :3] = self.target - torch.cat((self.robot[:, :2], torch.zeros_like(self.robot[:, :1])), dim=-1)
        entity[:, 0, 3] = 0.045
        entity[:, 0, 4:7] = torch.tensor([0.09, 0.09, 0.09], device=self.device)
        entity[:, 0, 7] = self.held.float()
        context = torch.zeros(self.num_envs, CONTEXT_DIM, device=self.device)
        rel = self.object[:, :2] - self.robot[:, :2]
        context[:, :2] = rel
        context[:, 2] = torch.atan2(rel[:, 1], rel[:, 0]) - self.robot[:, 2]
        context[:, 3] = torch.linalg.vector_norm(rel, dim=-1)
        context[:, 4] = self.held.float()
        context[:, 5] = self.recovery_count / 3.0
        context[:, 6] = self.step_count.float() / self.max_steps
        context[:, 7:12] = torch.nn.functional.one_hot((self.held.long() * 3).clamp_max(4), 5).float()
        return entity, context

    def step(self, continuous: torch.Tensor, recovery: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        action = continuous.tanh()
        old_distance = torch.linalg.vector_norm(self.object[:, :2] - self.robot[:, :2], dim=-1)
        self.robot[:, 0] += action[:, 0] * 0.08
        self.robot[:, 1] += action[:, 1] * 0.08
        self.robot[:, 2] += action[:, 2] * 0.12
        near = torch.linalg.vector_norm(self.object[:, :2] - self.robot[:, :2], dim=-1) < 0.72
        contact = near & (action[:, 4].abs() < 0.35) & (action[:, 5].abs() < 0.35)
        self.held |= contact
        self.object[self.held, :2] = self.robot[self.held, :2] + torch.stack((torch.full_like(self.robot[self.held, 0], 0.48), torch.zeros_like(self.robot[self.held, 0])), dim=-1)
        self.object[self.held, 2] = 0.84 + (0.10 + action[self.held, 7] * 0.06).clamp_min(0.04)
        release = self.held & (torch.linalg.vector_norm(self.target[:, :2] - self.robot[:, :2], dim=-1) < 0.62) & (action[:, 6] < -0.25)
        self.object[release] = self.target[release]
        self.held[release] = False
        self.recovery_count += (recovery != 0).float()
        new_distance = torch.linalg.vector_norm(self.object[:, :2] - self.robot[:, :2], dim=-1)
        placed = (~self.held) & (torch.linalg.vector_norm(self.object[:, :2] - self.target[:, :2], dim=-1) < 0.14)
        dropped = self.object[:, 2] < 0.60
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        done = placed | dropped | timeout
        reward = (old_distance - new_distance).clamp(-0.2, 0.2) + contact.float() * 0.5 + self.held.float() * 0.04 + placed.float() * 10.0 - dropped.float() * 6.0 - (recovery != 0).float() * 0.05
        extras = {"success": placed, "dropped": dropped, "timeouts": timeout, "mean_distance": new_distance.mean()}
        finished = done.nonzero(as_tuple=False).squeeze(-1)
        if len(finished):
            self.reset(finished)
        return self.observe(), reward, done, extras
