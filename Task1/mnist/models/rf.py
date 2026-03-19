from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from ..interface import MnistClassifierInterface
from ..data import numpy_to_flat

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


class RandomForestMNIST(MnistClassifierInterface):
    """Sklearn baseline on flattened pixels."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def train(
        self,
        X_train: ArrayF,
        y_train: ArrayI,
        X_val: Optional[ArrayF] = None,
        y_val: Optional[ArrayI] = None,
    ) -> None:
        Xf = numpy_to_flat(X_train)
        self.model.fit(Xf, y_train)
        # Note: external val ignored by design (no early stopping in RF)

    def predict(self, X: ArrayF) -> ArrayI:
        Xf = numpy_to_flat(X)
        return self.model.predict(Xf).astype(np.int64)
