"""MLP à deux couches cachées, implémenté à la main : forward pass, rétropropagation,
mise à jour des poids. Aucune bibliothèque de deep learning n'est utilisée ici.

Les formules suivent exactement la dérivation de la rétropropagation détaillée dans
le premier article de la série (LeCun, Bengio, Hinton, 2015, Fig. 1), avec une perte
quadratique (MSE) en sortie :

    z_j = sum_i w_ij * x_i + b_j        (somme pondérée, "pre-activation")
    y_j = f(z_j)                        (activation)

    dE/dy_L = y_L - t_L                 (erreur en sortie, directement calculable)
    dE/dz_l = dE/dy_l * f'(z_l)         (passage à travers la non-linéarité, chain rule)
    dE/dy_j = sum_l w_jl * dE/dz_l      (propagation vers l'arrière, mêmes poids que le forward)
    dE/dw_jk = y_j * dE/dz_k            (gradient d'un poids)
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


ACTIVATIONS = {
    "relu": (relu, relu_prime),
    "sigmoid": (sigmoid, sigmoid_prime),
}


class Dense:
    """Une couche entièrement connectée : z = x @ W + b, y = f(z)."""

    def __init__(self, n_in, n_out, activation, rng):
        # Initialisation "He", adaptée à ReLU
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self.f, self.f_prime = ACTIVATIONS[activation]

        # Caches remplis par forward(), utilisés par backward()
        self.x = None
        self.z = None
        self.y = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.y = self.f(self.z)
        return self.y

    def backward(self, dE_dy):
        """Reçoit dE/dy (gradient de la perte par rapport à la sortie de CETTE couche),
        stocke dE/dW et dE/db, renvoie dE/dy pour la couche précédente."""
        dE_dz = dE_dy * self.f_prime(self.z)           # dE/dz_l = dE/dy_l * f'(z_l)
        self.dW = self.x.T @ dE_dz / self.x.shape[0]    # dE/dw_jk = y_j * dE/dz_k (moyenné sur le batch)
        self.db = dE_dz.mean(axis=0)
        dE_dx = dE_dz @ self.W.T                        # dE/dy_j = sum_l w_jl * dE/dz_l
        return dE_dx

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class MLP:
    """MLP à 2 couches cachées + une couche de sortie sigmoïde, perte MSE.

    La perte quadratique est un choix pédagogique délibéré : elle permet d'utiliser
    directement dE/dy_L = y_L - t_L comme point de départ de la rétropropagation,
    exactement comme dans la dérivation générale de l'article — puis de laisser
    ce gradient traverser la non-linéarité de sortie comme n'importe quelle autre
    couche, sans raccourci algébrique (contrairement à la combinaison
    sigmoïde + cross-entropy, qui simplifie différemment).
    """

    def __init__(self, n_in, n_hidden1, n_hidden2, hidden_activation="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [
            Dense(n_in, n_hidden1, hidden_activation, rng),
            Dense(n_hidden1, n_hidden2, hidden_activation, rng),
            Dense(n_hidden2, 1, "sigmoid", rng),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true):
        dE_dy = y_pred - y_true  # dE/dy_L = y_L - t_L
        for layer in reversed(self.layers):
            dE_dy = layer.backward(dE_dy)

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def predict_proba(self, x):
        return self.forward(x)
