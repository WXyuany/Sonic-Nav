"""Custom hybrid recurrent PPO for task-space world-model control.

This implementation owns the policy contract instead of inheriting an RL
library's continuous-only action assumptions. It trains a Gaussian residual
controller and a categorical recovery controller under one PPO objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal


ENTITY_DIM = 12
CONTEXT_DIM = 24
CONTINUOUS_DIM = 8
RECOVERY_DIM = 5


@dataclass(frozen=True)
class RolloutBatch:
    entity: torch.Tensor          # [batch, time, 2, 12]: object and target
    context: torch.Tensor         # [batch, time, 24]: robot, skill, evidence
    continuous_action: torch.Tensor
    recovery_action: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor


class HybridRecurrentActorCritic(nn.Module):
    """Entity-aware recurrent actor critic with separate residual/recovery heads."""

    def __init__(self, entity_dim: int = ENTITY_DIM, context_dim: int = CONTEXT_DIM, hidden_dim: int = 192, memory_dim: int = 128):
        super().__init__()
        self.entity_encoder = nn.Sequential(nn.Linear(entity_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.context_encoder = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU())
        self.memory = nn.GRU(hidden_dim, memory_dim, batch_first=True)
        self.residual_mean = nn.Linear(memory_dim, CONTINUOUS_DIM)
        self.residual_log_std = nn.Parameter(torch.full((CONTINUOUS_DIM,), -1.2))
        self.recovery_logits = nn.Linear(memory_dim, RECOVERY_DIM)
        self.value_head = nn.Linear(memory_dim, 1)

    def forward(self, entity: torch.Tensor, context: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[Normal, Categorical, torch.Tensor, torch.Tensor]:
        object_latent = self.entity_encoder(entity[:, :, 0])
        target_latent = self.entity_encoder(entity[:, :, 1])
        context_latent = self.context_encoder(context)
        latent = self.fusion(torch.cat((object_latent, target_latent, context_latent), dim=-1))
        features, next_hidden = self.memory(latent, hidden)
        normal = Normal(self.residual_mean(features), self.residual_log_std.exp().view(1, 1, -1))
        categorical = Categorical(logits=self.recovery_logits(features))
        return normal, categorical, self.value_head(features).squeeze(-1), next_hidden

    def act(self, entity: torch.Tensor, context: torch.Tensor, hidden: torch.Tensor | None = None, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, categorical, value, next_hidden = self(entity, context, hidden)
        continuous = normal.mean if deterministic else normal.rsample()
        recovery = categorical.probs.argmax(dim=-1) if deterministic else categorical.sample()
        log_prob = normal.log_prob(continuous).sum(-1) + categorical.log_prob(recovery)
        return continuous, recovery, log_prob, value, next_hidden


class HybridPPO:
    def __init__(self, policy: HybridRecurrentActorCritic, *, learning_rate: float = 3e-4, clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01, max_grad_norm: float = 1.0):
        self.policy = policy
        self.clip_ratio, self.value_coef, self.entropy_coef, self.max_grad_norm = clip_ratio, value_coef, entropy_coef, max_grad_norm
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        normal, categorical, value, _ = self.policy(batch.entity, batch.context)
        log_prob = normal.log_prob(batch.continuous_action).sum(-1) + categorical.log_prob(batch.recovery_action.long())
        ratio = torch.exp(log_prob - batch.old_log_prob)
        clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        mask = batch.valid.float()
        denom = mask.sum().clamp_min(1.0)
        policy_loss = -(torch.minimum(ratio * batch.advantage, clipped * batch.advantage) * mask).sum() / denom
        value_loss = (((value - batch.returns) ** 2) * mask).sum() / denom
        entropy = ((normal.entropy().sum(-1) + categorical.entropy()) * mask).sum() / denom
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm))
        self.optimizer.step()
        return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()), "grad_norm": grad_norm, "clip_fraction": float((((ratio - 1.0).abs() > self.clip_ratio).float() * mask).sum().detach() / denom)}


def generalized_advantage(reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor, *, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized advantages and bootstrapped returns for [batch,time] tensors."""
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[:, 0])
    for index in range(reward.shape[1] - 1, -1, -1):
        next_value = value[:, index + 1] if index + 1 < reward.shape[1] else torch.zeros_like(running)
        nonterminal = 1.0 - done[:, index].float()
        delta = reward[:, index] + gamma * next_value * nonterminal - value[:, index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantage[:, index] = running
    normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    return normalized, advantage + value
