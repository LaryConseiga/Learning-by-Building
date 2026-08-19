"""Image de test synthétique en niveaux de gris — un carré, un cercle et une
diagonale, avec des contours nets et bien visibles par les filtres de Sobel.
Générée à la main pour ne dépendre d'aucun fichier image externe."""
import numpy as np


def make_test_image(size=128):
    img = np.zeros((size, size), dtype=np.float64)

    # carré
    img[15:55, 15:55] = 1.0

    # cercle
    yy, xx = np.mgrid[0:size, 0:size]
    circle_mask = (xx - 95) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    img[circle_mask] = 1.0

    # bande diagonale, en bas de l'image
    diag_mask = np.abs((yy - 90) - (xx - size // 2)) <= 4
    diag_mask &= (yy > 70)
    img[diag_mask] = 1.0

    return img
