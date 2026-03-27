from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class TokenCapPolicy(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def action_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def sample_action(self, obs: torch.Tensor):
        dist = self.action_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return torch.argmax(logits, dim=-1)
