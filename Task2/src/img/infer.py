from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.img.model import get_model
from src.img.dataset import get_class_names

BASE_DIR = Path(__file__).resolve().parents[2]

def predict_image(
    image_path: str | Path,
    model_path: str | Path = BASE_DIR / "models/img/model.pth",
    data_dir: str | Path = BASE_DIR / "data/cv/raw",
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = get_class_names(data_dir)

    model = get_model(num_classes=len(class_names))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return class_names[pred]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    pred = predict_image(args.image)
    print(f"Prediction: {pred}")