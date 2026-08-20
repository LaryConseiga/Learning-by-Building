# rnn-lstm-from-scratch

Code accompagnant le troisième article de la série [Learning by Building](../../README.md) :
*"Comment un réseau lit une phrase : représentations distribuées, RNN et LSTM"*.

Objectif : montrer concrètement le vanishing/exploding gradient dans un RNN, et
pourquoi le LSTM le corrige — pas juste l'affirmer.

## Contenu

| Fichier | Rôle |
|---|---|
| `rnn_numpy.py` | RNN "vanilla" from scratch : forward pass et rétropropagation dans le temps (BPTT). |
| `lstm_numpy.py` | Cellule LSTM from scratch : mêmes principes, avec la connexion à soi-même de la cellule mémoire. |
| `check_gradients.py` | Vérifie par différences finies que les gradients analytiques du RNN et du LSTM sont corrects. |
| `char_dataset.py` | Tâche jouet : prédire le caractère suivant dans "abcabcabc...". |
| `train_rnn.py` | Entraîne le RNN sur cette tâche courte — vérifie qu'il apprend correctement. |
| `gradient_vanishing.py` | Calcule et trace la norme du gradient à chaque pas de temps en remontant dans une séquence courte vs longue. |
| `long_range_task.py` | Tâche à dépendance longue : se souvenir d'un marqueur vu au premier pas de temps. |
| `compare_rnn_lstm.py` | Entraîne RNN et LSTM sur cette tâche longue et compare leurs courbes de précision. |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Vérifie d'abord que les deux implémentations de rétropropagation sont correctes
python check_gradients.py

# Version minimale : le RNN apprend une tâche courte, puis visualisation du vanishing gradient
python train_rnn.py
python gradient_vanishing.py

# Extension : RNN vs LSTM sur une tâche à dépendance longue (~3 min sur CPU)
python compare_rnn_lstm.py
```

Les figures sont écrites dans `outputs/`.

## Ce que montre chaque expérience

**`gradient_vanishing.py`** — la perte n'est calculée qu'au tout dernier pas de
temps d'un RNN non entraîné (poids fixes, pour isoler l'effet de l'architecture).
En remontant la rétropropagation dans le temps, la norme de dL/ds_t décroît de
façon exponentielle. Sur une séquence de 80 pas, elle devient environ 250 fois
plus petite qu'au pas le plus proche de la perte — exactement le phénomène
décrit dans l'article.

**`compare_rnn_lstm.py`** — la tâche (inspirée des expériences originales de
Hochreiter & Schmidhuber) : le réseau voit un marqueur (A ou B) suivi d'une
longue série de caractères de remplissage, et doit restituer ce marqueur au tout
dernier pas de temps. Sur 12 pas de temps, le RNN vanilla ne dépasse jamais le
niveau du hasard (~50 %) en 500 époques — le gradient qui devrait lui apprendre
à utiliser le tout premier caractère s'est évanoui avant d'y arriver. Le LSTM,
lui, décolle brusquement vers 100 % de précision après quelques centaines
d'époques : la connexion à soi-même de sa cellule mémoire (poids 1, gatée par la
porte d'oubli) laisse le gradient traverser la séquence sans s'atténuer.

## Note sur la tâche "abcabc..." vs la tâche à dépendance longue

La tâche `char_dataset.py` (prédire le caractère suivant dans "abcabcabc...")
sert uniquement à valider que le RNN s'entraîne correctement : c'est une
dépendance d'ordre 1 (il suffit de connaître le caractère précédent), qui
n'exige aucune mémoire longue distance. Pour comparer RNN et LSTM de façon
probante, `compare_rnn_lstm.py` utilise volontairement une tâche différente
(`long_range_task.py`), conçue spécifiquement pour exiger une mémoire sur toute
la longueur de la séquence — sans quoi le RNN vanilla n'aurait aucune raison
d'échouer, et la comparaison ne démontrerait rien.
