"""Même MLP (2 couches cachées, ReLU, sortie sigmoïde, perte MSE) réimplémenté avec
torch.nn et l'autograd, sur le même dataset — pour comparer au forward/backward
codés à la main dans mlp_numpy.py et vérifier que les deux implémentations
convergent vers des résultats similaires.

Usage :
    python mlp_pytorch.py --dataset moons
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split


class MLPTorch(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1), nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2), nn.ReLU(),
            nn.Linear(n_hidden2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(name, seed):
    if name == "moons":
        X, y = make_moons(n_samples=500, noise=0.2, random_state=seed)
    else:
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["moons", "circles"], default="moons")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    X, y = make_dataset(args.dataset, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).reshape(-1, 1)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).reshape(-1, 1)

    model = MLPTorch(2, args.hidden, args.hidden)
    # SGD "brut", sans momentum, pour rester comparable à la mise à jour manuelle
    # w -= lr * dW de mlp_numpy.py.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            with torch.no_grad():
                train_acc = ((y_pred > 0.5).float() == y_train).float().mean().item()
            print(f"[PyTorch] époque {epoch:4d} | perte {loss.item():.4f} | acc train {train_acc:.3f}")

    with torch.no_grad():
        y_test_pred = model(X_test)
        test_acc = ((y_test_pred > 0.5).float() == y_test).float().mean().item()
    print(f"\n[PyTorch] précision finale sur le test : {test_acc:.3f}")
    print("Compare cette précision à celle affichée par train_numpy.py sur le même dataset :")
    print(f"  python train_numpy.py --dataset {args.dataset} --epochs {args.epochs} --lr {args.lr} --hidden {args.hidden}")


if __name__ == "__main__":
    main()
