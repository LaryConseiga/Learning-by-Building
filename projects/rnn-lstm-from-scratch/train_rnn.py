"""Entraîne le RNN vanilla (rnn_numpy.py) à prédire le caractère suivant dans une
séquence répétitive ("abcabcabc..."). Sert à vérifier que l'implémentation
apprend correctement une dépendance courte, avant d'étudier le vanishing
gradient (gradient_vanishing.py) et la comparaison avec le LSTM sur une tâche à
dépendance longue (compare_rnn_lstm.py).

Usage :
    python train_rnn.py
"""
import argparse

import numpy as np

from char_dataset import make_repeating_sequence, make_training_pairs
from rnn_numpy import VanillaRNN


def accuracy(model, inputs, targets):
    _, probs = model.forward(inputs)
    preds = [int(np.argmax(probs[t])) for t in range(len(inputs))]
    return np.mean([p == t for p, t in zip(preds, targets)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="abc")
    parser.add_argument("--n-repeats", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    indices, char_to_idx, idx_to_char = make_repeating_sequence(args.pattern, args.n_repeats)
    pairs = make_training_pairs(indices, args.seq_len)

    model = VanillaRNN(vocab_size=len(char_to_idx), hidden_size=args.hidden_size, seed=args.seed)

    for epoch in range(args.epochs):
        total_loss, total_acc = 0.0, 0.0
        for inputs, targets in pairs:
            states, probs = model.forward(inputs)
            total_loss += model.loss(inputs, targets)
            grads, _ = model.backward(inputs, targets, states, probs)
            model.step(grads, args.lr)
            total_acc += accuracy(model, inputs, targets)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"époque {epoch:4d} | perte moyenne {total_loss / len(pairs):.4f} "
                  f"| précision moyenne {total_acc / len(pairs):.3f}")

    # Génère une courte séquence à partir du modèle entraîné, pour vérifier
    # qualitativement qu'il a bien appris le motif périodique.
    seed_char = args.pattern[0]
    s = np.zeros((args.hidden_size, 1))
    idx = char_to_idx[seed_char]
    generated = seed_char
    for _ in range(30):
        x = np.zeros((len(char_to_idx), 1))
        x[idx] = 1.0
        z = model.Wxh @ x + model.Whh @ s + model.bh
        s = np.tanh(z)
        logits = model.Why @ s + model.by
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = int(np.argmax(probs))
        generated += idx_to_char[idx]

    print(f"\nséquence générée à partir de '{seed_char}' : {generated}")


if __name__ == "__main__":
    main()
