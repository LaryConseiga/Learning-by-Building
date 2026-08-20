"""Visualise le vanishing gradient de ses propres yeux : on calcule la perte au
DERNIER pas de temps d'une séquence, on rétropropage dans le temps, et on trace
la norme de dL/ds_t à chaque pas de temps en remontant vers t=0. Comparé sur une
séquence courte et une séquence longue, pour montrer que le problème s'aggrave
avec la distance à parcourir — exactement le mécanisme décrit dans l'article :
"the backpropagated gradients either grow or shrink at each time step, so over
many time steps they typically explode or vanish."

Le réseau n'est volontairement PAS entraîné : on veut isoler l'effet de
l'architecture (la répétition de Whh et de la dérivée de tanh à chaque pas de
temps), pas celui d'un entraînement particulier.

Usage :
    python gradient_vanishing.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rnn_numpy import VanillaRNN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def gradient_norms_for_length(model, length, seed):
    rng = np.random.default_rng(seed)
    inputs = list(rng.integers(0, model.vocab_size, size=length))
    # La perte n'est calculée qu'au DERNIER pas de temps : on force ainsi le
    # gradient à parcourir toute la séquence pour atteindre les premiers pas.
    targets = [0] * (length - 1) + [int(rng.integers(0, model.vocab_size))]

    states, probs = model.forward(inputs)
    # On veut isoler la perte du dernier pas : les "cibles" des pas précédents
    # ne contribuent donc pas — on force leur gradient de sortie à zéro en
    # construisant des probabilités qui coïncident avec la cible (dérivée nulle).
    probs_only_last = {t: probs[t].copy() for t in range(length)}
    for t in range(length - 1):
        probs_only_last[t] = np.zeros_like(probs[t])
        probs_only_last[t][targets[t]] = 1.0  # dy = probs - onehot(target) = 0

    _, gradient_norms = model.backward(inputs, targets, states, probs_only_last)
    return gradient_norms


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vocab_size, hidden_size = 4, 32
    model = VanillaRNN(vocab_size, hidden_size, seed=0)
    # Échelle de poids volontairement "réaliste" (proche de ce que produit un
    # entraînement), pour que le phénomène soit visible : une échelle trop
    # petite masquerait le vanishing gradient sous le bruit numérique.
    rng = np.random.default_rng(1)
    model.Whh = rng.standard_normal((hidden_size, hidden_size)) * 0.9 / np.sqrt(hidden_size)

    short_len, long_len = 15, 80
    norms_short = gradient_norms_for_length(model, short_len, seed=2)
    norms_long = gradient_norms_for_length(model, long_len, seed=2)

    plt.figure(figsize=(8, 5))
    steps_back_short = [short_len - 1 - t for t in range(short_len)]
    steps_back_long = [long_len - 1 - t for t in range(long_len)]
    plt.semilogy(steps_back_short, norms_short, marker="o", markersize=3,
                 label=f"séquence courte ({short_len} pas)")
    plt.semilogy(steps_back_long, norms_long, marker="o", markersize=3,
                 label=f"séquence longue ({long_len} pas)")
    plt.xlabel("Nombre de pas de temps en arrière depuis la perte")
    plt.ylabel("Norme du gradient ||dL/ds_t|| (échelle log)")
    plt.title("Vanishing gradient dans un RNN vanilla, pendant la BPTT")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "gradient_vanishing.png")
    plt.savefig(fig_path, dpi=150)
    print(f"figure sauvegardée dans {fig_path}")

    print(f"\nnorme du gradient au pas le plus ancien — séquence courte : {norms_short[0]:.2e}")
    print(f"norme du gradient au pas le plus ancien — séquence longue : {norms_long[0]:.2e}")


if __name__ == "__main__":
    main()
