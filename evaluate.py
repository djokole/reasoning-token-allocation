from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from openreward import OpenReward

from token_env import CAP_CHOICES, TokenCapPolicy, task_features


def _meta_value(metadata, key, default):
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _load_tasks(args: argparse.Namespace):
    client = OpenReward()
    env = client.environments.get(name=args.env_name, base_url=args.base_url)
    tasks = env.list_tasks(split=args.split)
    if args.max_examples > 0:
        tasks = tasks[: min(args.max_examples, len(tasks))]
    return env, tasks


def evaluate_policy(args: argparse.Namespace) -> None:
    env, tasks = _load_tasks(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy = TokenCapPolicy(
        obs_dim=ckpt["obs_dim"],
        num_actions=ckpt["num_actions"],
        hidden_dim=ckpt["hidden_dim"],
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    n = args.episodes
    acc = 0
    total_reward = 0.0
    spent = []
    picked_caps = []

    with torch.no_grad():
        for _ in range(n):
            task = tasks[int(np.random.randint(0, len(tasks)))]
            obs = task_features(task)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = int(policy.greedy_action(obs_t).item())
            cap = ckpt["cap_choices"][action]
            with env.session(task=task) as session:
                session.call_tool("set_token_cap", {"cap": int(cap)})
                result = session.call_tool("answer", {"answer": ""})
            reward = float(result.reward or 0.0)
            metadata = result.metadata
            correct = int(_meta_value(metadata, "correct", 0))
            spent_tokens = float(_meta_value(metadata, "allocated_tokens", 0.0))
            acc += correct
            total_reward += reward
            spent.append(spent_tokens)
            picked_caps.append(cap)

    report = {
        "episodes": n,
        "accuracy": acc / n,
        "avg_reward": total_reward / n,
        "avg_spent_tokens": float(np.mean(spent)),
        "median_spent_tokens": float(np.median(spent)),
        "avg_cap_choice": float(np.mean(picked_caps)),
    }
    print(json.dumps(report, indent=2))


def evaluate_fixed_caps(args: argparse.Namespace) -> None:
    env, tasks = _load_tasks(args)
    cap_choices = CAP_CHOICES
    rows = []
    for cap in cap_choices:
        total_reward = 0.0
        acc = 0
        spent = []
        for _ in range(args.episodes):
            task = tasks[int(np.random.randint(0, len(tasks)))]
            with env.session(task=task) as session:
                session.call_tool("set_token_cap", {"cap": int(cap)})
                result = session.call_tool("answer", {"answer": ""})
            reward = float(result.reward or 0.0)
            metadata = result.metadata
            correct = int(_meta_value(metadata, "correct", 0))
            spent_tokens = float(_meta_value(metadata, "allocated_tokens", 0.0))
            total_reward += reward
            acc += correct
            spent.append(spent_tokens)
        rows.append(
            {
                "cap": cap,
                "accuracy": acc / args.episodes,
                "avg_reward": total_reward / args.episodes,
                "avg_spent_tokens": float(np.mean(spent)),
            }
        )
    print(json.dumps(rows, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/policy.pt")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", type=str, choices=["policy", "fixed"], default="policy")
    p.add_argument("--base-url", type=str, default="http://localhost:8080")
    p.add_argument("--env-name", type=str, default="gsm8ktokencapenvironment")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.mode == "policy":
        evaluate_policy(args)
    else:
        evaluate_fixed_caps(args)
