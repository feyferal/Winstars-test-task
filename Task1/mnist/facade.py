from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .models import RandomForestMNIST, FeedForwardNNMNIST, CNNMNIST

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


class MnistClassifier:

    def __init__(self, algorithm: Literal["rf", "nn", "cnn"], **kwargs) -> None:
        algo = algorithm.lower()
        if algo == "rf":
            self.impl = RandomForestMNIST(**kwargs)
        elif algo == "nn":
            self.impl = FeedForwardNNMNIST(**kwargs)
        elif algo == "cnn":
            self.impl = CNNMNIST(**kwargs)
        else:
            raise ValueError("algorithm must be one of: 'rf', 'nn', 'cnn'")

    def train(
        self,
        X_train: ArrayF,
        y_train: ArrayI,
        X_val: Optional[ArrayF] = None,
        y_val: Optional[ArrayI] = None,
    ) -> None:
        self.impl.train(X_train, y_train, X_val, y_val)

    def predict(self, X: ArrayF) -> ArrayI:
        return self.impl.predict(X)
