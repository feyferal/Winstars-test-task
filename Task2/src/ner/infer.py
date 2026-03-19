from __future__ import annotations
import argparse
import json
import torch
from transformers import pipeline
from ..paths import NER_MODEL_PATH

_clf = None


def _build_pipeline(device: int):
    return pipeline(
        "token-classification",
        model=str(NER_MODEL_PATH),
        tokenizer=str(NER_MODEL_PATH),
        aggregation_strategy="simple",
        device=device,
    )


def get_pipeline():
    global _clf
    if _clf is None:
        device = 0 if torch.cuda.is_available() else -1
        _clf = _build_pipeline(device)
    return _clf


def infer_ner(text: str, threshold: float = 0.55, debug: bool = False) -> list[str]:
    clf = get_pipeline()
    spans = clf(text)

    if debug:
        print("\n[DEBUG] RAW SPANS:")
        for s in spans:
            print(s)

    entities: list[str] = []

    for s in spans:
        label = s.get("entity_group") or s.get("entity") or ""
        score = float(s.get("score", 0.0))

        if "ANIMAL" not in label:
            continue

        if score < threshold:
            continue

        start = int(s.get("start", -1))
        end = int(s.get("end", -1))

        if 0 <= start < end <= len(text):
            entity = text[start:end].strip().lower()
            if entity:
                entities.append(entity)

    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = infer_ner(
        args.text,
        threshold=args.threshold,
        debug=args.debug
    )

    print(json.dumps(result, ensure_ascii=False))