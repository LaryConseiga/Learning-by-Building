"""Compare le RNN vanilla et le LSTM sur la tâche à dépendance longue : le
réseau doit se souvenir d'un marqueur vu au tout premier pas de temps pour le
restituer au tout dernier, après une longue série de caractères de remplissage.
C'est l'argument le plus direct de l'article : le LSTM réussit là où le RNN
vanilla échoue, grâce à sa connexion à soi-même de poids 1.

Usage :
    python compare_rnn_lstm.py
    python compare_rnn_lstm.py --length 60 --epochs 300
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lstm_numpy import LSTM
from long_range_task import VOCAB_SIZE, make_batch
from rnn_numpy import VanillaRNN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def masked_probs(probs, targets, length):
    """Ne garde le vrai signal de gradient qu'au DERNIER pas de temps : ailleurs,
    probs = onehot(target) => dy = probs - onehot(target) = 0. Sans ce masquage,
    le réseau apprendrait à prédire le marqueur à CHAQUE pas de temps de
    remplissage, ce qui ne demande aucune mémoire longue distance."""
    masked = {}
    for t in range(length):
        if t == length - 1:
            masked[t] = probs[t]
        else:
            masked[t] = np.zeros_like(probs[t])
            masked[t][targets[t]] = 1.0
    return masked


def accuracy_at_last_step(model, examples):
    correct = 0
    for inputs, targets, marker in examples:
        _, probs = model.forward(inputs)
        pred = int(np.argmax(probs[len(inputs) - 1]))
        correct += int(pred == marker)
    return correct / len(examples)


def train(model, length, epochs, lr, n_train, seed, label):
    train_examples = make_batch(n_train, length, seed=seed)
    test_examples = make_batch(200, length, seed=seed + 1000)

    accuracies = []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train_examples))
        for idx in order:
            inputs, targets, _ = train_examples[idx]
            states, probs = model.forward(inputs)
            grads, _ = model.backward(inputs, targets, states, masked_probs(probs, targets, length))
            model.step(grads, lr)

        acc = accuracy_at_last_step(model, test_examples)
        accuracies.append(acc)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  [{label}] époque {epoch:4d} | précision test {acc:.3f}")

    return accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Tâche : se souvenir d'un marqueur sur {args.length} pas de temps\n")

    rnn = VanillaRNN(VOCAB_SIZE, args.hidden_size, seed=args.seed)
    print("Entraînement du RNN vanilla...")
    acc_rnn = train(rnn, args.length, args.epochs, args.lr, args.n_train, args.seed, "RNN")

    lstm = LSTM(VOCAB_SIZE, args.hidden_size, seed=args.seed)
    print("\nEntraînement du LSTM...")
    acc_lstm = train(lstm, args.length, args.epochs, args.lr, args.n_train, args.seed, "LSTM")

    print(f"\nprécision finale — RNN vanilla : {acc_rnn[-1]:.3f} | LSTM : {acc_lstm[-1]:.3f}")

    plt.figure(figsize=(7, 5))
    # Avant d'apprendre, les deux réseaux oscillent entre les deux seules
    # stratégies "faciles" (toujours prédire la même classe) et tombent souvent
    # exactement sur les mêmes valeurs d'une époque à l'autre — d'où des styles
    # de trait différents, sans quoi les deux courbes se superposeraient.
    plt.plot(acc_rnn, label="RNN vanilla", linestyle="--", alpha=0.85)
    plt.plot(acc_lstm, label="LSTM", linestyle="-", alpha=0.85)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="hasard (2 classes)")
    plt.xlabel("Époque")
    plt.ylabel("Précision (test)")
    plt.title(f"RNN vs LSTM — se souvenir d'un marqueur sur {args.length} pas de temps")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "rnn_vs_lstm.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nfigure sauvegardée dans {fig_path}")


if __name__ == "__main__":
    main()
