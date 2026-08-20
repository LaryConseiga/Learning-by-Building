"""Cellule LSTM codée à la main : forward pass et rétropropagation dans le temps,
sans framework de deep learning.

La cellule mémoire c_t est mise à jour par :

    c_t = f_t * c_{t-1} + i_t * g_t

C'est exactement le mécanisme décrit dans l'article : une connexion à soi-même
(c_{t-1} -> c_t) dont le poids est implicitement 1 (aucune matrice ne multiplie
c_{t-1} dans ce terme), régulée uniquement par la porte d'oubli f_t — "multiplicatively
gated by another unit that learns to decide when to clear the content of the memory."
C'est cette absence de matrice de poids sur la connexion c_{t-1} -> c_t (donc pas de
répétition d'un facteur < 1 à chaque pas de temps) qui évite le vanishing gradient
du RNN vanilla.

    f_t = sigmoid(Wf [s_{t-1}, x_t] + bf)   porte d'oubli
    i_t = sigmoid(Wi [s_{t-1}, x_t] + bi)   porte d'entrée
    o_t = sigmoid(Wo [s_{t-1}, x_t] + bo)   porte de sortie
    g_t = tanh(Wg [s_{t-1}, x_t] + bg)      nouvelles valeurs candidates
    c_t = f_t * c_{t-1} + i_t * g_t          cellule mémoire (la connexion à soi-même)
    s_t = o_t * tanh(c_t)                    état caché / sortie
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class LSTM:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.01
        concat_size = hidden_size + vocab_size

        self.Wf = rng.standard_normal((hidden_size, concat_size)) * scale
        self.Wi = rng.standard_normal((hidden_size, concat_size)) * scale
        self.Wo = rng.standard_normal((hidden_size, concat_size)) * scale
        self.Wg = rng.standard_normal((hidden_size, concat_size)) * scale
        self.bf = np.ones((hidden_size, 1))   # biais d'oubli initialisé à 1 : par défaut, la cellule mémorise
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))

        self.Why = rng.standard_normal((vocab_size, hidden_size)) * scale
        self.by = np.zeros((vocab_size, 1))

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def params(self):
        return {
            "Wf": self.Wf, "Wi": self.Wi, "Wo": self.Wo, "Wg": self.Wg,
            "bf": self.bf, "bi": self.bi, "bo": self.bo, "bg": self.bg,
            "Why": self.Why, "by": self.by,
        }

    def forward(self, inputs, s0=None, c0=None):
        T = len(inputs)
        s_prev = np.zeros((self.hidden_size, 1)) if s0 is None else s0
        c_prev = np.zeros((self.hidden_size, 1)) if c0 is None else c0

        cache = {-1: {"s": s_prev, "c": c_prev}}
        probs = {}

        for t in range(T):
            x_t = np.zeros((self.vocab_size, 1))
            x_t[inputs[t]] = 1.0
            concat = np.vstack([cache[t - 1]["s"], x_t])

            f_t = sigmoid(self.Wf @ concat + self.bf)
            i_t = sigmoid(self.Wi @ concat + self.bi)
            o_t = sigmoid(self.Wo @ concat + self.bo)
            g_t = np.tanh(self.Wg @ concat + self.bg)

            c_t = f_t * cache[t - 1]["c"] + i_t * g_t
            tanh_c_t = np.tanh(c_t)
            s_t = o_t * tanh_c_t

            logits_t = self.Why @ s_t + self.by
            probs[t] = softmax(logits_t)

            cache[t] = {"x": x_t, "concat": concat, "f": f_t, "i": i_t, "o": o_t, "g": g_t,
                        "c": c_t, "tanh_c": tanh_c_t, "s": s_t}

        return cache, probs

    def loss(self, inputs, targets, s0=None, c0=None):
        _, probs = self.forward(inputs, s0, c0)
        return float(np.sum([-np.log(probs[t][targets[t], 0] + 1e-12) for t in range(len(inputs))]))

    def backward(self, inputs, targets, cache, probs):
        T = len(inputs)
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}
        ds_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        gradient_norms = [None] * T

        for t in reversed(range(T)):
            c = cache[t]
            c_prev = cache[t - 1]["c"]

            dy = probs[t].copy()
            dy[targets[t]] -= 1.0
            grads["Why"] += dy @ c["s"].T
            grads["by"] += dy

            ds_total = self.Why.T @ dy + ds_next  # dL/ds_t
            gradient_norms[t] = float(np.linalg.norm(ds_total))

            dc_total = ds_total * c["o"] * (1 - c["tanh_c"] ** 2) + dc_next  # dL/dc_t

            do_raw = (ds_total * c["tanh_c"]) * c["o"] * (1 - c["o"])
            di_raw = (dc_total * c["g"]) * c["i"] * (1 - c["i"])
            dg_raw = (dc_total * c["i"]) * (1 - c["g"] ** 2)
            df_raw = (dc_total * c_prev) * c["f"] * (1 - c["f"])

            grads["Wo"] += do_raw @ c["concat"].T
            grads["Wi"] += di_raw @ c["concat"].T
            grads["Wg"] += dg_raw @ c["concat"].T
            grads["Wf"] += df_raw @ c["concat"].T
            grads["bo"] += do_raw
            grads["bi"] += di_raw
            grads["bg"] += dg_raw
            grads["bf"] += df_raw

            dconcat = (self.Wo.T @ do_raw + self.Wi.T @ di_raw
                       + self.Wg.T @ dg_raw + self.Wf.T @ df_raw)

            ds_next = dconcat[: self.hidden_size]
            dc_next = dc_total * c["f"]  # dc_t/dc_{t-1} = f_t : la connexion à soi-même, gatée

        return grads, gradient_norms

    def step(self, grads, lr, clip=5.0):
        for name, value in self.params().items():
            np.clip(grads[name], -clip, clip, out=grads[name])
            value -= lr * grads[name]
