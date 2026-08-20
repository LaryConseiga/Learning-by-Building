"""RNN "vanilla" codé à la main : forward pass et rétropropagation dans le temps
(BPTT, backpropagation through time), sans framework de deep learning.

Notation, cohérente avec l'article :
    s_t = tanh(Wxh @ x_t + Whh @ s_{t-1} + bh)   état caché au pas de temps t
    y_t = Why @ s_t + by                          logits (avant softmax)

Les mêmes matrices Wxh, Whh, Why sont réutilisées à chaque pas de temps — c'est
le partage de poids décrit dans l'article, appliqué dans la dimension temporelle.

Le "dépliage dans le temps" décrit dans l'article correspond très concrètement à
la boucle `for t in range(T)` du forward, et à la boucle inversée du backward :
chaque pas de temps agit comme une couche d'un réseau profond, sauf que toutes
ces "couches" partagent les mêmes poids — leurs gradients s'additionnent donc au
lieu d'être indépendants.
"""
import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class VanillaRNN:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.01
        self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * scale
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.Why = rng.standard_normal((vocab_size, hidden_size)) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def params(self):
        return {"Wxh": self.Wxh, "Whh": self.Whh, "Why": self.Why, "bh": self.bh, "by": self.by}

    def forward(self, inputs, s0=None):
        """inputs : liste d'indices de caractères. Renvoie les caches nécessaires
        au backward (états cachés et probabilités à chaque pas de temps)."""
        T = len(inputs)
        s_prev = np.zeros((self.hidden_size, 1)) if s0 is None else s0
        states = {-1: s_prev}
        probs = {}
        for t in range(T):
            x_t = np.zeros((self.vocab_size, 1))
            x_t[inputs[t]] = 1.0
            z_t = self.Wxh @ x_t + self.Whh @ states[t - 1] + self.bh
            states[t] = np.tanh(z_t)
            logits_t = self.Why @ states[t] + self.by
            probs[t] = softmax(logits_t)
        return states, probs

    def loss(self, inputs, targets, s0=None):
        _, probs = self.forward(inputs, s0)
        return float(np.sum([-np.log(probs[t][targets[t], 0] + 1e-12) for t in range(len(inputs))]))

    def backward(self, inputs, targets, states, probs):
        """Rétropropagation dans le temps. Renvoie les gradients des poids et,
        pour chaque pas de temps (dans l'ordre chronologique), la norme de
        dL/ds_t : c'est cette norme, en remontant vers t=0, qui illustre le
        vanishing gradient."""
        T = len(inputs)
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        ds_next = np.zeros((self.hidden_size, 1))
        gradient_norms = [None] * T

        for t in reversed(range(T)):
            x_t = np.zeros((self.vocab_size, 1))
            x_t[inputs[t]] = 1.0

            dy = probs[t].copy()
            dy[targets[t]] -= 1.0  # dL/dlogits_t : gradient combiné softmax + cross-entropy

            grads["Why"] += dy @ states[t].T
            grads["by"] += dy

            ds = self.Why.T @ dy + ds_next  # dL/ds_t : contribution directe + héritée de t+1
            gradient_norms[t] = float(np.linalg.norm(ds))

            dz = (1 - states[t] ** 2) * ds  # chain rule à travers tanh
            grads["bh"] += dz
            grads["Wxh"] += dz @ x_t.T
            grads["Whh"] += dz @ states[t - 1].T

            ds_next = self.Whh.T @ dz  # propagation vers le pas de temps précédent

        return grads, gradient_norms

    def step(self, grads, lr, clip=5.0):
        for name, value in self.params().items():
            np.clip(grads[name], -clip, clip, out=grads[name])
            value -= lr * grads[name]
