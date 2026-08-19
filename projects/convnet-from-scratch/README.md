# convnet-from-scratch

Code accompagnant le deuxième article de la série [Learning by Building](../../README.md) :
*"Comment un réseau apprend à voir : les réseaux de neurones convolutifs"*.

Objectif : montrer le mécanisme de convolution et de pooling à la main — pas
juste appeler `nn.Conv2d`.

## Contenu

| Fichier | Rôle |
|---|---|
| `conv2d.py` | Convolution 2D from scratch (version boucles + version vectorisée), max pooling from scratch, et 4 filtres classiques (Sobel x/y, flou, sharpen). |
| `test_image.py` | Image de test synthétique (carré, cercle, diagonale), générée sans dépendance externe. |
| `run_filters_and_pooling.py` | Applique les 4 filtres puis le pooling à l'image de test, vérifie que la convolution en boucles et la version vectorisée donnent le même résultat, produit une figure récapitulative. |
| `cnn_model.py` | Petit ConvNet PyTorch (3 couches conv+ReLU+pooling), partagé par les deux scripts suivants. |
| `convnet_pytorch.py` | Entraîne ce ConvNet sur Fashion-MNIST et visualise les feature maps apprises à chaque couche. |
| `dropout_augmentation.py` | Bonus : compare, sur un petit sous-ensemble d'exemples, les courbes train/validation avec et sans dropout + augmentation de données. |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Version from scratch (NumPy) : filtres + pooling codés à la main
python run_filters_and_pooling.py

# ConvNet PyTorch sur Fashion-MNIST + visualisation des feature maps
python convnet_pytorch.py
python convnet_pytorch.py --epochs 5 --subset 10000   # plus rapide, sous-ensemble

# Bonus : effet du dropout + de l'augmentation de données sur le surapprentissage
python dropout_augmentation.py
```

Les résultats (figures) sont écrits dans `outputs/`. Fashion-MNIST est téléchargé
automatiquement dans `data/` au premier lancement des scripts PyTorch.

## Ce que `conv2d.py` implémente exactement

`conv2d_naive` glisse littéralement un filtre sur l'image, position après
position, exactement comme décrit dans l'article : à chaque position, le même
filtre (mêmes poids) est appliqué, d'où le nom de convolution. `conv2d_fast`
fait la même chose de façon vectorisée (`sliding_window_view` + produit
scalaire), et `run_filters_and_pooling.py` vérifie que les deux donnent un
résultat identique.

`max_pool2d` découpe la carte de sortie en blocs et ne garde que le maximum de
chaque bloc — une opération fixe, sans aucun poids appris, qui réduit la taille
spatiale et apporte l'invariance aux petits décalages décrite dans l'article.

## Ce que montre `dropout_augmentation.py`

Sur un petit sous-ensemble d'entraînement (800 exemples par défaut) et 40
époques, le réseau "nu" surapprend nettement : l'écart entre précision
d'entraînement et précision de validation se creuse progressivement. Avec
dropout (0.5) et augmentation de données (rotation/translation légères), cet
écart reste beaucoup plus faible — la régularisation fait exactement ce que
l'article attribue à ces deux techniques dans le succès d'ImageNet 2012.
