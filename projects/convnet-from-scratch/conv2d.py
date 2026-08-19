"""Convolution 2D et max pooling codés à la main, sans framework de deep learning.

Deux implémentations de la convolution sont fournies, pour le même résultat :
- conv2d_naive : boucles explicites, la plus proche du mécanisme décrit dans
  l'article — un petit filtre qui glisse, position après position, sur l'image.
- conv2d_fast : vectorisée avec `sliding_window_view`, beaucoup plus rapide.

`check_naive_vs_fast` (dans run_filters_and_pooling.py) vérifie que les deux
donnent exactement le même résultat.
"""
import numpy as np


def conv2d_naive(image, kernel):
    """Convolution 2D "valide" (pas de padding, pas de stride), par boucles."""
    kh, kw = kernel.shape
    ih, iw = image.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = image[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def conv2d_fast(image, kernel):
    """Même opération, vectorisée : toutes les fenêtres glissantes sont extraites
    d'un coup (sliding_window_view), puis chacune est réduite par un produit
    scalaire avec le filtre."""
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(image, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def max_pool2d(feature_map, size=2, stride=None):
    """Max pooling : découpe la carte en blocs de `size` x `size` et ne garde que
    le maximum de chaque bloc. Aucun poids appris — une opération fixe, purement
    mécanique, qui réduit la taille spatiale et apporte l'invariance aux petits
    décalages décrite dans l'article."""
    stride = stride or size
    ih, iw = feature_map.shape
    oh = (ih - size) // stride + 1
    ow = (iw - size) // stride + 1
    out = np.zeros((oh, ow), dtype=feature_map.dtype)
    for i in range(oh):
        for j in range(ow):
            block = feature_map[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = block.max()
    return out


# Quatre filtres "classiques", faits main — aucun poids appris, juste des valeurs
# connues qui détectent chacune un motif précis.
KERNELS = {
    "sobel_x": np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64),
    "sobel_y": np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float64),
    "blur": np.ones((3, 3), dtype=np.float64) / 9.0,
    "sharpen": np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float64),
}
