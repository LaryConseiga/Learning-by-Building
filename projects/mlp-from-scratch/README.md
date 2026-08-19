# mlp-from-scratch

Code accompagnant le premier article de la série [Learning by Building](../../README.md) :
*"Comment un réseau de neurones apprend vraiment"*.

Objectif : prouver avec du code — pas `model.fit()` — la compréhension du forward
pass et de la rétropropagation détaillés dans l'article, à partir des formules de
la Figure 1 de LeCun, Bengio & Hinton (*Nature*, 2015).

## Contenu

| Fichier | Rôle |
|---|---|
| `mlp_numpy.py` | Le MLP from scratch : forward pass, rétropropagation (chain rule couche par couche), mise à jour des poids. Aucune bibliothèque de deep learning. |
| `train_numpy.py` | Entraîne ce MLP sur `moons` ou `circles` (scikit-learn) et anime la frontière de décision qui se déforme au fil de l'entraînement. |
| `mlp_pytorch.py` | Le même MLP réimplémenté avec `torch.nn` et l'autograd, sur le même dataset — pour comparer au code manuel. |
| `vanishing_gradient.py` | Bonus : un MLP à 10 couches cachées entraîné en sigmoïde vs en ReLU, pour visualiser le vanishing gradient. |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Version from scratch (NumPy) : produit une animation de la frontière de décision
python train_numpy.py --dataset moons
python train_numpy.py --dataset circles --epochs 4000 --lr 0.3

# Comparaison avec PyTorch, sur le même dataset
python mlp_pytorch.py --dataset moons

# Bonus : vanishing gradient, sigmoïde vs ReLU sur 10 couches
python vanishing_gradient.py
```

Les résultats (courbes de perte, GIF de la frontière de décision, figure du
vanishing gradient) sont écrits dans `outputs/`.

## Ce que `mlp_numpy.py` implémente exactement

Un MLP à 2 couches cachées (ReLU) + une couche de sortie (sigmoïde), entraîné avec
une perte quadratique (MSE) par descente de gradient stochastique. Les formules du
forward et du backward suivent directement celles de l'article :

```
z_j = sum_i w_ij * x_i + b_j        forward : somme pondérée
y_j = f(z_j)                        forward : activation

dE/dy_L = y_L - t_L                 backward : erreur de sortie, directement calculable
dE/dz_l = dE/dy_l * f'(z_l)         backward : chain rule à travers la non-linéarité
dE/dy_j = sum_l w_jl * dE/dz_l      backward : propagation vers l'arrière (mêmes poids que le forward)
dE/dw_jk = y_j * dE/dz_k            backward : gradient d'un poids
```

La perte MSE (plutôt que la cross-entropy) est un choix délibéré : elle permet
d'utiliser `dE/dy_L = y_L - t_L` tel quel, sans le raccourci algébrique propre à la
combinaison sigmoïde + cross-entropy, ce qui garde le code fidèle, couche par
couche, à la dérivation générale de l'article.
