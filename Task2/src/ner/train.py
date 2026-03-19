from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from inspect import signature

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)

from ..paths import NER_MODEL_PATH, TRAIN_NER
from .dataset import (
    NERDataset,
    compute_class_weights,
    collect_entity_labels,
)
from .utils import set_seed, load_spacy_style, compute_metrics


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights, ignore_index=-100
        )

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def make_training_args(outdir: Path, lr: float, bs: int, epochs: int) -> TrainingArguments:
    params = signature(TrainingArguments).parameters

    kwargs = dict(
        output_dir=str(outdir),
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
    )

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "macro_f1"
        kwargs["greater_is_better"] = True
    if "report_to" in params:
        kwargs["report_to"] = []

    return TrainingArguments(**kwargs)


def train_ner(
    train_data: List[Tuple[str, Dict[str, Any]]],
    iterations: int = 5,
    hf_model_name: str = "distilbert-base-uncased",
    lr: float = 3e-5,
    batch_size: int = 8,
    max_len: int = 128,
    seed: int = 42,
) -> None:
    set_seed(seed)

    labels = collect_entity_labels(train_data)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    tr, va = train_test_split(
        train_data,
        test_size=0.2,
        random_state=seed,
        shuffle=True
    )

    ds_tr = NERDataset(tr, tokenizer, label2id, max_len=max_len)
    ds_va = NERDataset(va, tokenizer, label2id, max_len=max_len)

    model = AutoModelForTokenClassification.from_pretrained(
        hf_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weights(
        tr, label2id, tokenizer, max_len
    ).to(device)

    os.makedirs(NER_MODEL_PATH, exist_ok=True)
    args = make_training_args(Path(NER_MODEL_PATH), lr, batch_size, iterations)

    trainer = WeightedTokenTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(Path(NER_MODEL_PATH)))
    tokenizer.save_pretrained(str(Path(NER_MODEL_PATH)))

    print(f"[OK] Saved transformer NER to {NER_MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)
    args = ap.parse_args()

    data = load_spacy_style(TRAIN_NER)

    train_ner(
        data,
        iterations=args.iterations,
    )