"""Tâche jouet : prédire le caractère suivant dans une séquence courte et
répétitive, ex. "abcabcabc...". Une tâche volontairement simple (dépendance
d'ordre 1 : il suffit de connaître le caractère précédent), utilisée pour
valider que le RNN s'entraîne correctement, et comme séquence de base pour
visualiser le vanishing gradient de la BPTT."""
import numpy as np


def make_repeating_sequence(pattern="abc", n_repeats=20):
    text = pattern * n_repeats
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    indices = [char_to_idx[ch] for ch in text]
    return indices, char_to_idx, idx_to_char


def make_training_pairs(indices, seq_len):
    """Découpe la séquence encodée en paires (entrées, cibles) de longueur
    seq_len : la cible au pas t est le caractère qui suit l'entrée au pas t."""
    pairs = []
    for start in range(0, len(indices) - seq_len - 1, seq_len):
        inputs = indices[start:start + seq_len]
        targets = indices[start + 1:start + seq_len + 1]
        pairs.append((inputs, targets))
    return pairs
