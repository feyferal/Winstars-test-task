from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..interface import MnistClassifierInterface
from ..data import add_channel_dim, make_loaders

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


class FFNN(nn.Module):
    """Tiny MLP: Flatten -> 256 -> 128 -> logits(10)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                  # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardNNMNIST(MnistClassifierInterface):
    """MLP classifier with optional external validation usage."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 128,
        device: Optional[str] = None,
        val_split: float = 0.1,
        print_val: bool = True,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.print_val = print_val
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FFNN().to(self.device)

    def _fit_epoch(
        self,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
    ) -> float:
        """One training epoch; returns mean loss."""
        self.model.train()
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _val_accuracy(self, loader: DataLoader) -> float:
        """Compute accuracy on validation loader."""
        self.model.eval()
        correct, n = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self.model(xb).argmax(dim=1)
            correct += int((preds == yb).sum())
            n += xb.size(0)
        return (correct / n) if n else 0.0

    def train(
        self,
        X_train: ArrayF,
        y_train: ArrayI,
        X_val: Optional[ArrayF] = None,
        y_val: Optional[ArrayI] = None,
    ) -> None:
        tr_loader, val_loader = make_loaders(
            X_train,
            y_train,
            batch_size=self.batch_size,
            X_val=X_val,
            y_val=y_val,
            val_split=self.val_split,
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs + 1):
            train_loss = self._fit_epoch(tr_loader, opt)
            if self.print_val:
                val_acc = self._val_accuracy(val_loader)
                print(f"[MLP] epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    @torch.no_grad()
    def predict(self, X: ArrayF) -> ArrayI:
        self.model.eval()
        xb = add_channel_dim(X).to(self.device)
        logits = self.model(xb)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        return preds
