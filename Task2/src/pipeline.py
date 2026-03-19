from __future__ import annotations

from src.ner.infer import infer_ner
from src.img.infer import predict_image


def normalize_label(label: str) -> str:
    return label.lower().strip()


def run_pipeline(text: str, image_path: str) -> bool:
    entities = infer_ner(text)

    if not entities:
        return False

    text_animals = {normalize_label(e) for e in entities}

    image_label = predict_image(image_path)
    image_label = normalize_label(image_label)

    return image_label in text_animals

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    result = run_pipeline(args.text, args.image)

    print(result)