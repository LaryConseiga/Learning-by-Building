"""Bonus : entraîne deux fois le même ConvNet sur un petit sous-ensemble de
Fashion-MNIST — une fois "nu", une fois avec dropout + augmentation de données
(rotation, translation légères) — et compare les courbes de précision en
validation. Reproduit à petite échelle les deux ingrédients cités dans
l'article comme responsables du succès d'ImageNet 2012, avec le sous-ensemble
volontairement réduit pour bien faire apparaître le surapprentissage du modèle
"nu" en peu d'exemples.

Usage :
    python dropout_augmentation.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from cnn_model import SmallCNN
from convnet_pytorch import DATA_DIR, evaluate, get_device

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

AUGMENTATION = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])
NO_AUGMENTATION = transforms.ToTensor()


def get_loaders(subset, batch_size, seed, augment):
    transform = AUGMENTATION if augment else NO_AUGMENTATION
    train_full = FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    # Une deuxième vue du même sous-ensemble, sans augmentation, pour mesurer une
    # précision d'entraînement propre (comparable entre les deux configurations).
    train_full_clean = FashionMNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test_full = FashionMNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_full), generator=g)[:subset]
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(Subset(train_full_clean, train_idx), batch_size=256, shuffle=False)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False)
    return train_loader, train_eval_loader, test_loader


def run(label, dropout, augment, subset, epochs, lr, batch_size, seed, device):
    torch.manual_seed(seed)
    train_loader, train_eval_loader, test_loader = get_loaders(subset, batch_size, seed, augment)
    model = SmallCNN(n_classes=10, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_eval_loader, device)
        val_acc = evaluate(model, test_loader, device)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"[{label}] époque {epoch + 1}/{epochs} | acc train {train_acc:.3f} | acc validation {val_acc:.3f}")

    return train_accuracies, val_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--subset", type=int, default=800,
                         help="petit sous-ensemble volontaire, pour bien voir le surapprentissage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()

    train_baseline, val_baseline = run(
        "sans dropout/augmentation", dropout=0.0, augment=False,
        subset=args.subset, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed, device=device)

    train_regularized, val_regularized = run(
        "avec dropout + augmentation", dropout=0.5, augment=True,
        subset=args.subset, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].plot(train_baseline, label="train")
    axes[0].plot(val_baseline, label="validation")
    axes[0].set_title("Sans dropout / sans augmentation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Précision")
    axes[0].legend()

    axes[1].plot(train_regularized, label="train")
    axes[1].plot(val_regularized, label="validation")
    axes[1].set_title("Avec dropout (0.5) + augmentation")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    fig.suptitle(f"Effet du dropout + de l'augmentation de données ({args.subset} exemples d'entraînement)")
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "dropout_augmentation.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nfigure sauvegardée dans {fig_path}")
    print(f"écart train-val final — sans régularisation : {train_baseline[-1] - val_baseline[-1]:.3f} "
          f"| avec régularisation : {train_regularized[-1] - val_regularized[-1]:.3f}")


if __name__ == "__main__":
    main()
