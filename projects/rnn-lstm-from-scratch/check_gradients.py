"""Vérifie, par différences finies, que les gradients analytiques calculés par
backward() (RNN et LSTM) correspondent aux gradients numériques — la BPTT est
facile à coder avec un bug subtil, donc mieux vaut vérifier empiriquement que de
se fier uniquement à la dérivation.

Usage :
    python check_gradients.py
"""
import numpy as np

from lstm_numpy import LSTM
from rnn_numpy import VanillaRNN


def numerical_gradient(loss_fn, param, epsilon=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + epsilon
        loss_plus = loss_fn()

        param[idx] = original - epsilon
        loss_minus = loss_fn()

        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        it.iternext()
    return grad


def check_model(model, inputs, targets, name):
    states, probs = model.forward(inputs)
    grads, _ = model.backward(inputs, targets, states, probs)

    # Deux gradients quasi nuls (ex. 4.5e-9 vs 4.7e-9) donnent une erreur RELATIVE
    # énorme sans que ce soit un bug — l'erreur qui compte alors est l'écart
    # ABSOLU. On n'échoue que si les deux critères sont mauvais à la fois.
    print(f"\n--- {name} ---")
    all_ok = True
    for pname, pvalue in model.params().items():
        numeric = numerical_gradient(lambda: model.loss(inputs, targets), pvalue)
        analytic = grads[pname]
        abs_error = np.abs(numeric - analytic)
        rel_error = abs_error / np.maximum(np.abs(numeric) + np.abs(analytic), 1e-8)
        bad = (abs_error > 1e-6) & (rel_error > 1e-4)
        ok = not bad.any()
        all_ok &= ok
        print(f"  {pname:6s} erreur abs max = {abs_error.max():.2e} "
              f"| erreur relative max = {rel_error.max():.2e}  [{'OK' if ok else 'ECHEC'}]")

    assert all_ok, f"{name} : gradients incorrects"
    print(f"  -> tous les gradients de {name} sont corrects.")


def main():
    rng = np.random.default_rng(0)
    vocab_size, hidden_size, seq_len = 5, 6, 6
    inputs = list(rng.integers(0, vocab_size, size=seq_len))
    targets = list(rng.integers(0, vocab_size, size=seq_len))

    check_model(VanillaRNN(vocab_size, hidden_size, seed=1), inputs, targets, "VanillaRNN")
    check_model(LSTM(vocab_size, hidden_size, seed=1), inputs, targets, "LSTM")


if __name__ == "__main__":
    main()
