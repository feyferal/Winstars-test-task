from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.img.dataset import AnimalDataset
from src.img.model import get_model


def train(
    data_dir: str = "data/cv/raw",
    model_path: str = "models/img/model.pth",
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 7,
    samples_per_class: int = 300,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnimalDataset(data_dir)

    class_samples: dict[int, list[tuple]] = defaultdict(list)

    for path, label in dataset.samples:
        class_samples[label].append((path, label))

    balanced_samples: list[tuple] = []

    for label, items in class_samples.items():
        k = min(len(items), samples_per_class)
        balanced_samples.extend(random.sample(items, k))

    dataset.samples = balanced_samples

    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    print("Class distribution:", counter)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = get_model(num_classes=len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(dataloader)}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()