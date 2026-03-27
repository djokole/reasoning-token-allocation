from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from openreward import OpenReward
from tqdm import trange

from token_env import CAP_CHOICES, TokenCapPolicy, task_features


def _meta_value(metadata, key, default):
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def train(args: argparse.Namespace) -> None:
    client = OpenReward()
    env = client.environments.get(name=args.env_name, base_url=args.base_url)
    tasks = env.list_tasks(split="train")
    if args.max_examples > 0:
        tasks = tasks[: min(args.max_examples, len(tasks))]
    if len(tasks) == 0:
        raise ValueError("No tasks found in train split.")

    cap_choices = CAP_CHOICES
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = TokenCapPolicy(
        obs_dim=3,
        num_actions=len(cap_choices),
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0
    baseline_momentum = 0.97

    running_reward = 0.0
    running_acc = 0.0
    running_spent = 0.0

    for step in trange(1, args.steps + 1, desc="training"):
        task = tasks[int(np.random.randint(0, len(tasks)))]
        obs = task_features(task)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t, log_prob_t = policy.sample_action(obs_t)
        cap = cap_choices[int(action_t.item())]

        with env.session(task=task) as session:
            session.call_tool("set_token_cap", {"cap": cap})
            result = session.call_tool("answer", {"answer": ""})

        reward = float(result.reward or 0.0)
        metadata = result.metadata
        correct = int(_meta_value(metadata, "correct", 0))
        spent_tokens = float(_meta_value(metadata, "allocated_tokens", 0.0))

        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
        advantage = reward - baseline
        loss = -log_prob_t * advantage

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_reward += reward
        running_acc += correct
        running_spent += spent_tokens

        if step % args.log_every == 0:
            denom = float(args.log_every)
            print(
                json.dumps(
                    {
                        "step": step,
                        "avg_reward": running_reward / denom,
                        "avg_acc": running_acc / denom,
                        "avg_spent_tokens": running_spent / denom,
                    }
                )
            )
            running_reward = 0.0
            running_acc = 0.0
            running_spent = 0.0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "cap_choices": cap_choices,
            "obs_dim": 3,
            "num_actions": len(cap_choices),
            "hidden_dim": args.hidden_dim,
        },
        out / "policy.pt",
    )
    print(f"saved policy to {out / 'policy.pt'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--base-url", type=str, default="http://localhost:8080")
    p.add_argument("--env-name", type=str, default="gsm8ktokencapenvironment")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
