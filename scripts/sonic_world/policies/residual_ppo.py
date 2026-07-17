"""Hybrid recurrent PPO network for world-model residual control.

The policy deliberately emits task-space residuals and a recovery mode, never
joint torques or WBC targets.  It is runner-agnostic so the same checkpoint
shape can be used by a local PyTorch trainer or an RSL-RL adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal


OBSERVATION_DIM = 48
CONTINUOUS_ACTION_DIM = 8
RECOVERY_ACTION_DIM = 5


@dataclass(frozen=True)
class HybridAction:
    continuous: torch.Tensor
    recovery_mode: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class HybridRecurrentActorCritic(nn.Module):
    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 256, recurrent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.memory = nn.GRU(hidden_dim, recurrent_dim, batch_first=True)
        self.mean = nn.Linear(recurrent_dim, CONTINUOUS_ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((CONTINUOUS_ACTION_DIM,), -1.2))
        self.recovery_logits = nn.Linear(recurrent_dim, RECOVERY_ACTION_DIM)
        self.value_head = nn.Linear(recurrent_dim, 1)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = observation.unsqueeze(1) if observation.ndim == 2 else observation
        encoded = self.encoder(sequence)
        features, hidden = self.memory(encoded, hidden)
        features = features[:, -1]
        return self.mean(features), self.log_std.expand_as(self.mean(features)), self.recovery_logits(features), self.value_head(features).squeeze(-1), hidden

    def act(self, observation: torch.Tensor, hidden: torch.Tensor | None = None, deterministic: bool = False) -> tuple[HybridAction, torch.Tensor]:
        mean, log_std, logits, value, next_hidden = self.forward(observation, hidden)
        continuous_dist = Normal(mean, log_std.exp())
        recovery_dist = Categorical(logits=logits)
        continuous = mean if deterministic else continuous_dist.rsample()
        recovery = logits.argmax(dim=-1) if deterministic else recovery_dist.sample()
        log_prob = continuous_dist.log_prob(continuous).sum(-1) + recovery_dist.log_prob(recovery)
        return HybridAction(continuous=continuous, recovery_mode=recovery, log_prob=log_prob, value=value), next_hidden
