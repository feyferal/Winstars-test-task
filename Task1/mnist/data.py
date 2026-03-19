from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


def add_channel_dim(x: ArrayF) -> torch.Tensor:
    return torch.tensor(x[:, None, :, :], dtype=torch.float32)


def numpy_to_flat(X: ArrayF) -> ArrayF:
    return X.reshape(len(X), -1).astype(np.float32)


def make_loaders(
    X_train: ArrayF,
    y_train: ArrayI,
    *,
    batch_size: int = 128,
    seed: int = 42,
    X_val: Optional[ArrayF] = None,
    y_val: Optional[ArrayI] = None,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    X_train_t = add_channel_dim(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    if X_val is not None and y_val is not None:
        X_val_t = add_channel_dim(X_val)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        ds_tr = TensorDataset(X_train_t, y_train_t)
        ds_val = TensorDataset(X_val_t, y_val_t)
    else:
        ds = TensorDataset(X_train_t, y_train_t)
        n_val = int(len(ds) * val_split)
        n_tr = len(ds) - n_val
        g = torch.Generator().manual_seed(seed)
        ds_tr, ds_val = random_split(ds, [n_tr, n_val], generator=g)

    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return tr_loader, val_loader


def load_mnist_numpy(
    train_split: float = 0.9,
    seed: int = 42,
) -> Tuple[ArrayF, ArrayI, ArrayF, ArrayI, ArrayF, ArrayI]:
    dtrain = datasets.MNIST(root="./data", train=True, download=True)
    dtest = datasets.MNIST(root="./data", train=False, download=True)

    X_all: ArrayF = (dtrain.data.numpy().astype(np.float32)) / 255.0
    y_all: ArrayI = dtrain.targets.numpy().astype(np.int64)
    X_test: ArrayF = (dtest.data.numpy().astype(np.float32)) / 255.0
    y_test: ArrayI = dtest.targets.numpy().astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        train_size=train_split,
        random_state=seed,
        stratify=y_all,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
