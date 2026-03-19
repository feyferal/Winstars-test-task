from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_spacy_style(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list")

    for i, item in enumerate(data[:5]):
        if not (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], dict)
            and "entities" in item[1]
        ):
            raise ValueError(f"Invalid format at index {i}: {item}")

    return data


def compute_metrics(p):
    from sklearn.metrics import precision_recall_fscore_support

    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids

    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    if labels.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }

    pr, rc, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    acc = (preds == labels).mean()

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(pr),
        "macro_recall": float(rc),
    }