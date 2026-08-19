"""Bonus : illustre le vanishing gradient décrit dans l'article. Un MLP profond
(10 couches cachées) entraîné avec des activations sigmoïdes voit sa perte stagner,
car le gradient s'écrase en traversant de nombreuses dérivées proches de zéro. Le
même réseau entraîné avec ReLU (dérivée 0 ou 1, jamais "écrasante") converge
normalement.

Usage :
    python vanishing_gradient.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def build_deep_mlp(n_layers, hidden, activation_cls):
    """Construit un MLP de n_layers couches cachées + une sortie sigmoïde.

    L'initialisation des poids est adaptée à l'activation utilisée (Xavier pour
    sigmoïde, He pour ReLU) : c'est l'initialisation "recommandée" pour chacune,
    celle qui préserve le mieux la variance du signal à la traversée des couches.
    Sans ça, un réseau de 10 couches peine à apprendre quelle que soit
    l'activation, et la comparaison sigmoïde vs ReLU perd son sens — la
    stagnation de la sigmoïde doit venir de l'activation elle-même (sa dérivée
    plafonne à 0.25), pas d'une mauvaise initialisation.
    """
    layers = []
    n_in = 2
    for _ in range(n_layers):
        linear = nn.Linear(n_in, hidden)
        if activation_cls is nn.ReLU:
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers += [linear, activation_cls()]
        n_in = hidden
    output_linear = nn.Linear(n_in, 1)
    nn.init.xavier_normal_(output_linear.weight)
    nn.init.zeros_(output_linear.bias)
    layers += [output_linear, nn.Sigmoid()]
    return nn.Sequential(*layers)


def train(model, X, y, epochs, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=0)
    X_np = StandardScaler().fit_transform(X_np)  # centre/normalise : évite de biaiser l'activation dès la 1re couche
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)

    n_layers, hidden, epochs, lr = 10, 16, 3000, 0.5

    torch.manual_seed(0)
    model_sigmoid = build_deep_mlp(n_layers, hidden, nn.Sigmoid)
    print("Entraînement du MLP à 10 couches, activation sigmoïde...")
    losses_sigmoid = train(model_sigmoid, X, y, epochs, lr)

    torch.manual_seed(0)
    model_relu = build_deep_mlp(n_layers, hidden, nn.ReLU)
    print("Entraînement du MLP à 10 couches, activation ReLU...")
    losses_relu = train(model_relu, X, y, epochs, lr)

    print(f"\nPerte finale — sigmoïde : {losses_sigmoid[-1]:.4f} | ReLU : {losses_relu[-1]:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(losses_sigmoid, label="Sigmoïde (10 couches) — le gradient s'écrase")
    plt.plot(losses_relu, label="ReLU (10 couches) — converge normalement")
    plt.xlabel("Époque")
    plt.ylabel("Perte (MSE)")
    plt.title("Vanishing gradient : sigmoïde vs ReLU sur un MLP à 10 couches")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "vanishing_gradient.png")
    plt.savefig(fig_path, dpi=150)
    print(f"figure sauvegardée dans {fig_path}")


if __name__ == "__main__":
    main()
