"""Entraîne le MLP codé à la main (mlp_numpy.py) sur un dataset 2D (moons ou cercles
concentriques) et visualise la frontière de décision qui se déforme au fil de
l'entraînement — l'illustration concrète de la distorsion de l'espace d'entrée
évoquée dans l'article pour rendre les classes linéairement séparables.

Usage :
    python train_numpy.py --dataset moons
    python train_numpy.py --dataset circles --epochs 4000 --lr 0.3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

from mlp_numpy import MLP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def make_dataset(name, seed):
    if name == "moons":
        X, y = make_moons(n_samples=500, noise=0.2, random_state=seed)
    else:
        X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=seed)
    return X.astype(np.float64), y.reshape(-1, 1).astype(np.float64)


def decision_boundary_frame(model, X, y, epoch, loss):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, proba, levels=50, cmap="RdBu_r", alpha=0.8, vmin=0, vmax=1)
    ax.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu_r", edgecolors="black", linewidths=0.5, s=20)
    ax.set_title(f"Époque {epoch} — perte {loss:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(frame).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["moons", "circles"], default="moons")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = make_dataset(args.dataset, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    model = MLP(n_in=2, n_hidden1=args.hidden, n_hidden2=args.hidden,
                hidden_activation=args.activation, seed=args.seed)

    frames = []
    losses = []

    for epoch in range(args.epochs + 1):
        y_pred = model.forward(X_train)
        loss = float(0.5 * np.mean((y_pred - y_train) ** 2))
        losses.append(loss)

        # backward() et step() doivent utiliser le cache (x, z) laissé par le forward
        # ci-dessus : tout autre appel à forward() (ex. decision_boundary_frame, qui
        # prédit sur la grille de visualisation) écraserait ce cache avant coup.
        model.backward(y_pred, y_train)
        model.step(args.lr)

        if epoch % args.snapshot_every == 0:
            frames.append(decision_boundary_frame(model, X_train, y_train, epoch, loss))
            train_acc = float(((y_pred > 0.5) == y_train).mean())
            print(f"époque {epoch:4d} | perte {loss:.4f} | acc train {train_acc:.3f}")

    y_test_pred = model.predict_proba(X_test)
    test_acc = float(((y_test_pred > 0.5) == y_test).mean())
    print(f"\nprécision finale sur le test : {test_acc:.3f}")

    gif_path = os.path.join(OUTPUT_DIR, f"decision_boundary_{args.dataset}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"animation sauvegardée dans {gif_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Époque")
    plt.ylabel("Perte (MSE)")
    plt.title(f"Courbe d'apprentissage — {args.dataset}")
    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, f"loss_curve_{args.dataset}.png")
    plt.savefig(loss_path, dpi=150)
    print(f"courbe de perte sauvegardée dans {loss_path}")


if __name__ == "__main__":
    main()
