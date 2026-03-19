from __future__ import annotations

import argparse
from sklearn.metrics import accuracy_score

from Task1.mnist.data import load_mnist_numpy
from Task1.mnist.facade import MnistClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate MNIST classifiers (RF, NN, CNN).")
    parser.add_argument(
        "--algo",
        type=str,
        default="cnn",
        choices=["rf", "nn", "cnn"],
        help="Algorithm to use: 'rf' (Random Forest), 'nn' (Feed-Forward Neural Net), or 'cnn' (Convolutional Neural Net).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (for NN/CNN).")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size (for NN/CNN).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (for NN/CNN).")
    parser.add_argument("--no_val_print", action="store_true", help="Suppress validation accuracy output.")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_numpy()

    if args.algo == "rf":
        clf = MnistClassifier("rf")
    elif args.algo == "nn":
        clf = MnistClassifier(
            "nn",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            print_val=not args.no_val_print,
        )
    else:
        clf = MnistClassifier(
            "cnn",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            print_val=not args.no_val_print,
        )

    clf.train(X_train, y_train, X_val, y_val)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Algorithm: {args.algo} | Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
