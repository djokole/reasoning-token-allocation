from __future__ import annotations

from typing import Any

import numpy as np

CAP_CHOICES = [128, 512, 1024, 2048, 4096]
MAX_CAP = 4096


def task_features(task_spec: dict[str, Any]) -> np.ndarray:
    question = str(task_spec["question"])
    answer = str(task_spec.get("answer", ""))
    q_words = len(question.split())
    a_words = len(answer.split())
    q_norm = min(1.0, q_words / 140.0)
    a_norm = min(1.0, a_words / 80.0)
    complexity = min(1.0, 0.7 * q_norm + 0.3 * a_norm)
    return np.array([q_norm, a_norm, complexity], dtype=np.float32)
