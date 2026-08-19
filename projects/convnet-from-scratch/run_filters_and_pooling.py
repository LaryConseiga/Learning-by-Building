"""Applique des filtres codés à la main (Sobel horizontal/vertical, flou, sharpen)
à une image de test, puis un max pooling — pour voir concrètement ce que
"glisser un filtre" et "réduire par pooling" veulent dire, avant même de parler
d'apprentissage. Vérifie aussi que la convolution par boucles et la convolution
vectorisée donnent exactement le même résultat.

Usage :
    python run_filters_and_pooling.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conv2d import KERNELS, conv2d_fast, conv2d_naive, max_pool2d
from test_image import make_test_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def check_naive_vs_fast(image):
    kernel = KERNELS["sobel_x"]
    out_naive = conv2d_naive(image, kernel)
    out_fast = conv2d_fast(image, kernel)
    max_diff = np.abs(out_naive - out_fast).max()
    assert np.allclose(out_naive, out_fast), "conv2d_naive et conv2d_fast divergent !"
    print(f"vérification : conv2d_naive et conv2d_fast donnent un résultat identique "
          f"(écart max {max_diff:.2e})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image = make_test_image()

    check_naive_vs_fast(image)

    n_rows = len(KERNELS) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title(f"image d'origine\n{image.shape}")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for row, (name, kernel) in enumerate(KERNELS.items(), start=1):
        convolved = conv2d_fast(image, kernel)
        pooled = max_pool2d(convolved, size=2)

        axes[row, 0].imshow(kernel, cmap="gray")
        axes[row, 0].set_title(f"filtre : {name} ({kernel.shape[0]}x{kernel.shape[1]})")

        axes[row, 1].imshow(convolved, cmap="gray")
        axes[row, 1].set_title(f"après convolution\n{convolved.shape}")

        axes[row, 2].imshow(pooled, cmap="gray")
        axes[row, 2].set_title(f"après max pooling 2x2\n{pooled.shape}")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "filters_and_pooling.png")
    fig.savefig(fig_path, dpi=150)
    print(f"figure sauvegardée dans {fig_path}")


if __name__ == "__main__":
    main()
