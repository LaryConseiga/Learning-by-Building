"""Tâche à dépendance longue, où un RNN vanilla échoue clairement sur de longues
séquences : la première "lettre" de la séquence est un marqueur A ou B, suivi
d'une longue série de caractères de remplissage. Le réseau ne doit produire sa
réponse (le marqueur vu au tout début) qu'au DERNIER pas de temps.

C'est le test classique utilisé dans la littérature originale du LSTM (Hochreiter
& Schmidhuber) pour mesurer la capacité d'un réseau récurrent à préserver 1 bit
d'information sur une longue distance — exactement ce que la connexion à
soi-même de poids 1 de la cellule mémoire est censée permettre (voir l'article).

Vocabulaire : 0 = marqueur A, 1 = marqueur B, 2 = caractère de remplissage.
"""
import numpy as np

VOCAB_SIZE = 3
MARKER_A, MARKER_B, FILLER = 0, 1, 2


def make_batch(n_examples, length, seed):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n_examples):
        marker = int(rng.integers(0, 2))  # 0 ou 1
        inputs = [marker] + [FILLER] * (length - 1)
        # la cible n'importe qu'au dernier pas de temps : les pas précédents
        # sont ignorés par les scripts d'entraînement (voir compare_rnn_lstm.py)
        targets = [marker] * length
        examples.append((inputs, targets, marker))
    return examples
