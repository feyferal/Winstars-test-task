from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float32]
ArrayI = NDArray[np.int64]


class MnistClassifierInterface(abc.ABC):

    @abc.abstractmethod
    def train(
        self,
        X_train: ArrayF,
        y_train: ArrayI,
        X_val: Optional[ArrayF] = None,
        y_val: Optional[ArrayI] = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: ArrayF) -> ArrayI:
        raise NotImplementedError
