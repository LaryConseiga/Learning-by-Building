"""Entraîne le petit ConvNet (cnn_model.py) sur Fashion-MNIST, puis visualise les
feature maps apprises à chaque couche pour une image de test — on y retrouve
concrètement la hiérarchie de représentations de l'article 1 : chaque étage
convolution-ReLU-pooling produit des cartes de plus en plus abstraites et de plus
en plus petites (28x28 -> 14x14 -> 7x7 -> 3x3).

Usage :
    python convnet_pytorch.py
    python convnet_pytorch.py --epochs 5 --subset 10000
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLASSES = ["T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
           "Sandale", "Chemise", "Basket", "Sac", "Bottine"]


def get_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).cpu()
            return "cuda"
        except RuntimeError:
            print("GPU détecté mais incompatible avec ce build de PyTorch — utilisation du CPU.")
    return "cpu"


def get_dataloaders(subset, batch_size, seed):
    transform = transforms.ToTensor()
    train_full = FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_full = FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

    if subset:
        g = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_full), generator=g)[:subset]
        train_full = Subset(train_full, train_idx)

    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def plot_feature_maps(model, image, label, device):
    model.eval()
    with torch.no_grad():
        _, features = model(image.unsqueeze(0).to(device), return_features=True)

    fig, axes = plt.subplots(1, len(features) + 1, figsize=(4 * (len(features) + 1), 4))
    axes[0].imshow(image.squeeze(0), cmap="gray")
    axes[0].set_title(f"entrée ({CLASSES[label]})\n28x28")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for i, fmap in enumerate(features, start=1):
        fmap = fmap.squeeze(0).cpu()
        n_show = min(8, fmap.shape[0])
        n_cols = 4
        n_rows = -(-n_show // n_cols)  # arrondi au-dessus

        h, w = fmap.shape[1:]
        pad = 1
        mosaic = torch.zeros(n_rows * (h + pad), n_cols * (w + pad))
        for k in range(n_show):
            r, c = divmod(k, n_cols)
            mosaic[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = fmap[k]

        axes[i].imshow(mosaic, cmap="viridis")
        axes[i].set_title(f"conv{i} ({fmap.shape[0]} feature maps)\n{tuple(fmap.shape[1:])} chacune")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "feature_maps.png")
    fig.savefig(fig_path, dpi=150)
    print(f"figure des feature maps sauvegardée dans {fig_path}")
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", type=int, default=0, help="0 = dataset complet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    device = get_device()

    train_loader, test_loader = get_dataloaders(args.subset, args.batch_size, args.seed)

    model = SmallCNN(n_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        test_acc = evaluate(model, test_loader, device)
        print(f"époque {epoch + 1}/{args.epochs} | perte {loss.item():.4f} | précision test {test_acc:.3f}")

    # Feature maps pour la première image du jeu de test
    test_dataset = test_loader.dataset
    image, label = test_dataset[0]
    plot_feature_maps(model, image, label, device)


if __name__ == "__main__":
    main()
