from __future__ import annotations

import numpy as np
import torch


def collect_entity_labels(data: list[tuple[str, dict]]) -> list[str]:
    types = sorted({ent[2] for _, ann in data for ent in ann.get("entities", [])})
    labels = ["O"]
    for t in types:
        labels += [f"B-{t}", f"I-{t}"]
    return labels


def char_spans_to_bio(
    text: str,
    entities: list[tuple[int, int, str]],
    tokenizer,
    max_len: int
):
    tags_char = ["O"] * len(text)

    for start, end, label in entities:
        if start < 0 or end > len(text) or start >= end:
            continue

        tags_char[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags_char[i] = f"I-{label}"

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True
    )

    labels = []

    for (s, e) in enc["offset_mapping"]:
        if s == e:
            labels.append(-100)
            continue

        if s >= len(tags_char):
            labels.append("O")
            continue

        span_tags = tags_char[s:e]

        if not span_tags:
            labels.append("O")
            continue

        if any(t.startswith("B-") for t in span_tags):
            tag = next(t for t in span_tags if t.startswith("B-"))
        elif any(t.startswith("I-") for t in span_tags):
            tag = next(t for t in span_tags if t.startswith("I-"))
        else:
            tag = "O"

        labels.append(tag)

    return enc, labels


class NERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: list[tuple[str, dict]],
        tokenizer,
        label2id: dict[str, int],
        max_len: int = 256
    ):
        self.items = []

        for text, ann in data:
            enc, tags = char_spans_to_bio(
                text,
                ann.get("entities", []),
                tokenizer,
                max_len
            )

            lab_ids = []
            for t in tags:
                if t == -100:
                    lab_ids.append(-100)
                else:
                    lab_ids.append(label2id.get(t, label2id["O"]))

            enc.pop("offset_mapping")
            enc["labels"] = lab_ids

            self.items.append(enc)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: torch.tensor(v) for k, v in self.items[i].items()}


def compute_class_weights(
    data,
    label2id,
    tokenizer,
    max_len
) -> torch.Tensor:
    counts = np.zeros(len(label2id), dtype=np.float64)

    for text, ann in data:
        _, tags = char_spans_to_bio(
            text,
            ann.get("entities", []),
            tokenizer,
            max_len
        )

        for t in tags:
            if t == -100:
                continue
            counts[label2id.get(t, label2id["O"])] += 1

    counts[counts == 0] = 1.0

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)