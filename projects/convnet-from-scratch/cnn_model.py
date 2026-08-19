"""Petit ConvNet PyTorch : 3 couches conv+ReLU+pooling puis une couche de sortie,
exactement l'architecture décrite dans l'article (empilement de plusieurs étages
convolution → non-linéarité → pooling, suivi d'une couche fully-connected).

Utilisé à la fois par convnet_pytorch.py (visualisation des feature maps) et
dropout_augmentation.py (comparaison avec/sans dropout).
"""
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # 28x28 -> pool -> 14x14 -> pool -> 7x7 -> pool -> 3x3, avec 32 feature maps
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(32 * 3 * 3, n_classes)

    def forward(self, x, return_features=False):
        f1 = self.pool(self.relu(self.conv1(x)))
        f2 = self.pool(self.relu(self.conv2(f1)))
        f3 = self.pool(self.relu(self.conv3(f2)))
        out = self.fc(self.dropout(f3.flatten(1)))
        if return_features:
            return out, [f1, f2, f3]
        return out
