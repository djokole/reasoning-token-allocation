from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from openreward.environments import Environment, JSONObject, Server, Split, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

from token_env import CAP_CHOICES
from token_env.spec import task_features


def _dataset_paths(split: str) -> str:
    data_dir = os.environ.get("ORWD_DATA_DIR", ".")
    if split == "train":
        return os.path.join(data_dir, "train-00000-of-00001.parquet")
    if split == "test":
        return os.path.join(data_dir, "test-00000-of-00001.parquet")
    raise ValueError(f"Unknown split: {split}")


def _load_tasks(split: str) -> list[dict[str, Any]]:
    path = _dataset_paths(split)
    tasks = pd.read_parquet(path).to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        out.append(
            {
                "id": str(idx),
                "question": str(task["question"]),
                "answer": str(task["answer"]),
            }
        )
    return out


def _sample_needed_reasoning_tokens(obs: np.ndarray, rng: np.random.Generator) -> int:
    q_norm, a_norm, complexity = obs.tolist()
    mean = 120.0 + 1700.0 * complexity + 250.0 * q_norm + 120.0 * a_norm
    noise = rng.normal(0.0, 120.0)
    return int(np.clip(mean + noise, 80.0, 4096.0))


def _correctness_probability(cap: int, needed: int) -> float:
    margin = (cap - needed) / 320.0
    saturation = 1.0 / (1.0 + np.exp(-margin))
    p = 0.10 + 0.86 * saturation
    return float(np.clip(p, 0.02, 0.97))


class SetTokenCapParams(BaseModel):
    cap: int = Field(description="Reasoning token cap for this episode")


class AnswerParams(BaseModel):
    answer: str = Field(default="", description="Final answer text")


class GSM8KTokenCapEnvironment(Environment):
    _train_tasks: list[JSONObject] | None = None
    _test_tasks: list[JSONObject] | None = None

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.task_spec = {
            "id": str(task_spec.get("id", "")),
            "question": str(task_spec.get("question", "")),
            "answer": str(task_spec.get("answer", "")),
        }
        self.secrets = secrets

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str):
        if split == "train":
            if cls._train_tasks is None:
                cls._train_tasks = _load_tasks("train")
            return cls._train_tasks
        if split == "test":
            if cls._test_tasks is None:
                cls._test_tasks = _load_tasks("test")
            return cls._test_tasks
        raise ValueError(f"Unknown split: {split}")

    def setup(self):
        self.rng = np.random.default_rng()
        self.cap = CAP_CHOICES[0]
        self.submitted = False

    def get_prompt(self):
        return [TextBlock(type="text", text=self.task_spec["question"])]

    @tool
    def set_token_cap(self, params: SetTokenCapParams) -> ToolOutput:
        if params.cap not in CAP_CHOICES:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Invalid cap: {params.cap}")],
                reward=0.0,
                finished=False,
                metadata={"cap_choices": CAP_CHOICES},
            )
        self.cap = int(params.cap)
        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Token cap set to {self.cap}.")],
            reward=0.0,
            finished=False,
            metadata={"cap": self.cap},
        )

    @tool
    def answer(self, params: AnswerParams) -> ToolOutput:
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="Episode already finished.")],
                reward=0.0,
                finished=True,
            )
        obs = task_features(self.task_spec)
        needed = _sample_needed_reasoning_tokens(obs, self.rng)
        p_correct = _correctness_probability(self.cap, needed)
        correct = int(self.rng.random() < p_correct)
        allocated_tokens = int(self.cap)
        reward = float(correct * (1.0 - allocated_tokens / 4096.0))
        self.submitted = True
        return ToolOutput(
            blocks=[TextBlock(type="text", text="Correct!" if correct else "Wrong!")],
            reward=reward,
            finished=True,
            metadata={
                "correct": correct,
                "cap": self.cap,
                "allocated_tokens": allocated_tokens,
                "needed_tokens_estimate": needed,
                "p_correct": p_correct,
                "provided_answer": params.answer,
                "gold_answer": self.task_spec["answer"],
            },
        )


if __name__ == "__main__":
    Server([GSM8KTokenCapEnvironment]).run()
