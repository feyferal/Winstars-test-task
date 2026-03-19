from __future__ import annotations

import random
from collections import Counter, defaultdict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from src.img.dataset import AnimalDataset
from src.img.model import get_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def train(
    data_dir: str = "data/cv/raw",
    model_path: str = "models/img/model.pth",
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 7,
    seed: int = 42,
) -> None:
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnimalDataset(data_dir)

    class_samples: dict[int, list[tuple]] = defaultdict(list)

    for path, label in dataset.samples:
        class_samples[label].append((path, label))

    min_class_size = min(len(items) for items in class_samples.values())
    print(f"Using {min_class_size} samples per class")

    balanced_samples: list[tuple] = []

    for label, items in class_samples.items():
        balanced_samples.extend(random.sample(items, min_class_size))

    dataset.samples = balanced_samples

    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    print("Class distribution:", counter)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = get_model(num_classes=len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)}")

        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": dataset.classes,
        },
        model_path,
    )

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()