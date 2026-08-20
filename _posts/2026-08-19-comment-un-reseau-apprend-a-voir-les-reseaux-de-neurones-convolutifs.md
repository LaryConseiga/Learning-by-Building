---
title: "Comment un réseau apprend à voir : les réseaux de neurones convolutifs"
date: 2026-08-19 09:00:00 +0000
categories: [Deep Learning, Vision par ordinateur]
tags: [deep-learning, cnn, convnets, computer-vision, pooling]
math: true
---

*🇬🇧 [English version]({{ '/posts/how-a-network-learns-to-see-convolutional-neural-networks/' | relative_url }})*

## Où on en était

Dans le [premier article]({{ '/posts/comment-un-reseau-de-neurones-apprend-vraiment/' | relative_url }}) de cette série, on a vu comment un réseau de neurones apprend : le forward pass, la descente de gradient, et la rétropropagation, le mécanisme qui permet de calculer l'effet de chaque poids sur l'erreur finale, à travers toutes les couches d'un réseau.

Ce qu'on n'avait pas encore abordé, c'est **l'architecture** elle-même. Un perceptron multicouche classique, dit *fully-connected* (entièrement connecté), applique le même schéma générique quelle que soit la nature des données. Mais pour un problème précis (la vision par ordinateur), cette généricité coûte cher, et une architecture spécialisée s'impose : les **réseaux de neurones convolutifs**, ou **ConvNets** (*Convolutional Neural Networks*, CNN). C'est le sujet de cet article, toujours à partir du texte fondateur de LeCun, Bengio et Hinton[^lecun2015].

## Le problème que les ConvNets résolvent : l'explosion du nombre de poids

### Rappel : comment se comptent les poids dans une couche fully-connected

Dans une couche entièrement connectée, chaque unité de la couche cachée est reliée à **toutes** les unités de la couche précédente, avec un poids indépendant par connexion. Le nombre total de poids se calcule simplement :

$$
\text{nombre de poids} = (\text{nombre d'entrées}) \times (\text{nombre d'unités})
$$

Pour une petite image de 100 pixels et une couche cachée de 5 unités, ça donne $100 \times 5 = 500$ poids, encore gérable.

### Le problème à l'échelle réelle

Mais les images réelles ne font pas 100 pixels. Prenons une image de résolution modeste, 1000×1000 pixels (soit 1 000 000 de pixels), et une première couche cachée de seulement 1000 unités, ce qui reste modeste pour un réseau de vision :

$$
1\,000\,000 \text{ pixels} \times 1000 \text{ unités} = 1\,000\,000\,000 \text{ poids}
$$

**Un milliard de poids**, rien que pour la première couche. Ce chiffre pose trois problèmes concrets :

1. **Calcul et mémoire.** Stocker et mettre à jour un milliard de poids, encore et encore à chaque étape de la descente de gradient, demande une puissance de calcul et une mémoire considérables.
2. **Besoin de données.** Plus il y a de poids à régler, plus il faut d'exemples labellisés pour les ajuster correctement sans tomber dans le surapprentissage.
3. **Aucune structure spatiale exploitée.** Chaque poids est associé à un pixel à une position précise. Si le réseau apprend à détecter un motif en haut à gauche d'une image, ce motif n'est pas reconnu s'il apparaît ailleurs : les poids qui le détecteraient en bas à droite n'ont jamais été entraînés pour ça.

Ce dernier point rejoint directement le **dilemme sélectivité-invariance** évoqué dans le premier article : on veut un réseau sensible aux détails qui comptent (la forme d'un museau), mais insensible aux détails qui ne comptent pas (sa position exacte dans l'image). Une couche fully-connected n'offre structurellement aucun moyen d'obtenir cette invariance.

## Les quatre idées clés des ConvNets

Le texte identifie précisément les quatre principes qui répondent à ce problème :

> There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.[^lecun2015]

### 1. Connexions locales

Au lieu qu'une unité regarde l'image entière, elle ne regarde qu'un petit **patch local**, par exemple une fenêtre de 3×3 ou 5×5 pixels. Sur une image de 6×6 pixels avec un patch de 3×3, une unité n'a plus besoin que de 9 poids, contre 36 pour une unité fully-connected regardant toute l'image.

Cette fenêtre ne se déplace pas au hasard : elle balaie **systématiquement** toute l'image, position après position, de gauche à droite puis de haut en bas, de façon à ce qu'aucune zone ne soit ignorée.

### 2. Poids partagés : l'idée centrale

C'est le mécanisme le plus important, et celui qui distingue vraiment les ConvNets d'une simple couche à connexions locales :

> All units in a feature map share the same filter bank. […] the local statistics of images and other signals are invariant to location. In other words, if a motif can appear in one part of the image, it could appear anywhere, hence the idea of units at different locations sharing the same weights and detecting the same pattern in different parts of the array.[^lecun2015]

Le même petit jeu de poids (appelé **filter bank**, ou simplement filtre) est réutilisé à **toutes** les positions balayées par le patch. Ce n'est pas neuf unités différentes avec neuf jeux de poids différents ; c'est **une seule** unité, dont le filtre glisse sur l'image entière.

Le schéma ci-dessous illustre ce mécanisme : à la position 1 et à la position 2, c'est exactement le **même** filtre de 9 poids qui est appliqué : seule sa position sur l'image change.

![Mécanisme du partage de poids dans une couche convolutive](/assets/img/posts/convnet-shared-weights.svg)
_Le même filtre glisse sur toute l'image ; ses poids ne changent jamais, seule sa position change. Le résultat forme une feature map, ensuite réduite par le pooling._

**L'effet sur le nombre de poids est spectaculaire.** Reprenons l'exemple d'une image de 100×100 pixels, avec un filtre de 5×5 :

| Approche | Poids à apprendre |
|---|---|
| Fully-connected (1 unité regardant toute l'image) | 10 000 |
| Connexions locales, sans partage de poids | ≈ 230 400 |
| **Connexions locales + poids partagés** | **25** |

Peu importe le nombre de positions balayées par le filtre (dix ou dix mille) : un seul jeu de 25 poids suffit, puisqu'il est réutilisé partout. C'est mathématiquement une opération de **convolution**, d'où le nom de l'architecture :

> Mathematically, the filtering operation performed by a feature map is a discrete convolution, hence the name.[^lecun2015]

**Et ce partage résout aussi le problème d'invariance.** Puisque le même détecteur de motif (par exemple, un détecteur de contour vertical) est appliqué à toutes les positions de l'image, il n'a pas besoin d'avoir été entraîné séparément pour chaque endroit où ce motif pourrait apparaître.

Une couche a en général plusieurs filtres en parallèle, chacun indépendant, pour détecter différents types de motifs :

> Different feature maps in a layer use different filter banks.[^lecun2015]

Un filtre d'une couche profonde regarde d'ailleurs un patch à travers **toutes** les feature maps produites par la couche précédente, pas une seule. Un filtre de 3×3 recevant 10 feature maps en entrée compte donc $3 \times 3 \times 10 = 90$ poids.

### 3. Pooling : réduire sans apprendre

Le pooling répond à un problème plus fin que le partage de poids seul ne résout pas complètement : même avec un filtre partagé, la sortie d'un détecteur change encore légèrement si le motif se décale de quelques pixels seulement.

> A typical pooling unit computes the maximum of a local patch of units in one feature map […] thereby reducing the dimension of the representation and creating an invariance to small shifts and distortions.[^lecun2015]

Le mécanisme, le plus souvent sous forme de **max pooling**, est simple : on découpe la feature map en petits blocs (typiquement 2×2), et on ne garde que la valeur **maximale** de chaque bloc.

**Prenons un exemple concret.** Si une feature map contient ces quatre valeurs dans un bloc 2×2 :

```
[ 0.2  0.8 ]
[ 0.1  0.3 ]
```

Le pooling ne garde que **0.8**, le maximum. Peu importe que le motif détecté soit précisément en haut à droite du bloc ou légèrement décalé : tant qu'il est détecté quelque part dans cette petite zone, le signal fort est conservé.

Ce mécanisme a deux effets complémentaires : il **réduit la taille** de la représentation (un pooling 2×2 divise par quatre le nombre de valeurs), et il **ajoute de l'invariance** aux petits décalages de position.

**Un point important à ne pas confondre : le pooling n'a aucun poids appris.** Ce n'est pas une couche entraînée par rétropropagation comme la convolution : c'est une opération fixe, purement mécanique. Le partage de poids réduit le **nombre de paramètres** à apprendre dans la convolution ; le pooling réduit la **taille spatiale** de la sortie, sans aucun paramètre. Ce sont deux réductions différentes, à deux étapes différentes du traitement.

### 4. Empilement de plusieurs couches

L'architecture typique d'un ConvNet enchaîne ces briques plusieurs fois de suite :

> Two or three stages of convolution, non-linearity and pooling are stacked, followed by more convolutional and fully-connected layers.[^lecun2015]

L'ordre précis à l'intérieur de chaque étage n'est jamais arbitraire :

$$
\text{Convolution} \rightarrow \text{ReLU} \rightarrow \text{Pooling}
$$

La convolution calcule d'abord la somme pondérée locale ($z$) avec les poids appris. La non-linéarité ReLU ($f(z) = \max(0, z)$) s'applique ensuite à chaque valeur, sans changer la taille de la grille. Le pooling intervient **en dernier**, une fois la couche activée, pour réduire la taille spatiale avant de passer à l'étage suivant.

Ce cycle se répète plusieurs fois, pas une seule fois à la toute fin du réseau. Une image de 100×100 pixels, après trois étages de pooling 2×2 successifs, se retrouve autour de 12×12, tout en gagnant généralement en nombre de feature maps à chaque étage. Cette progression rejoint directement la hiérarchie de représentations décrite dans le premier article : chaque étage convolution-ReLU-pooling correspond, grossièrement, à un niveau de cette hiérarchie : contours, puis motifs, puis parties d'objets.

**Ce qui est fixé à l'avance, et ce qui est appris.** La taille des filtres, la taille de la fenêtre de pooling et le nombre de feature maps par couche sont des choix de conception, faits avant l'entraînement, et rien n'oblige à les garder identiques d'une couche à l'autre. Ce qui est appris par rétropropagation, ce sont uniquement les **valeurs numériques** à l'intérieur de chaque filtre.

## Le tournant de 2012 : ImageNet

Ces principes existaient déjà dans les années 1990, avec des succès pratiques réels : la lecture automatique de chèques bancaires, par exemple, traitait dans les années 1990 plus de 10 % des chèques aux États-Unis. Mais les ConvNets sont restés largement boudés par la communauté de vision par ordinateur, qui privilégiait des approches à base de caractéristiques conçues à la main, jusqu'à un moment précis :

> Despite these successes, ConvNets were largely forsaken by the mainstream computer-vision and machine-learning communities until the ImageNet competition in 2012. When deep convolutional networks were applied to a data set of about a million images from the web that contained 1,000 different classes, they achieved spectacular results, almost halving the error rates of the best competing approaches.[^lecun2015]

Diviser par deux le taux d'erreur des meilleures approches concurrentes, sur un million d'images et 1000 catégories : c'est ce résultat qui a déclenché l'adoption massive du deep learning en vision par ordinateur. Le texte identifie précisément les ingrédients de ce succès :

> This success came from the efficient use of GPUs, ReLUs, a new regularization technique called dropout, and techniques to generate more training examples by deforming the existing ones.[^lecun2015]

Le rôle des GPU et de ReLU a déjà été discuté dans le premier article. Deux nouveaux éléments méritent une explication.

**Le dropout** est une technique de régularisation qui consiste à désactiver aléatoirement une partie des unités du réseau pendant l'entraînement, forçant le réseau à ne pas trop dépendre d'une combinaison particulière d'unités, ce qui réduit le surapprentissage.

**L'augmentation de données**, elle, consiste à créer artificiellement de nouveaux exemples d'entraînement en déformant légèrement des images existantes : rotation, recadrage, effet miroir, variation de luminosité. Ce n'est pas équivalent à simplement collecter plus de vraies photos : cette technique **cible spécifiquement** les variations dont on sait, a priori, qu'elles ne devraient pas changer la classification. Elle enseigne directement au réseau l'invariance recherchée, en complément de celle déjà apportée par l'architecture (poids partagés et pooling).

## Ce qu'il faut retenir

- Une couche fully-connected appliquée à des images réelles produit un nombre de poids ingérable, sans exploiter la structure spatiale des données.
- Les ConvNets répondent à ce problème avec quatre idées : **connexions locales** (un petit patch, pas toute l'image), **poids partagés** (le même filtre réutilisé à toutes les positions, d'où le terme convolution), **pooling** (réduction de taille et invariance aux petits décalages, sans poids appris), et **empilement de couches** (répétition du cycle convolution-ReLU-pooling).
- Le partage de poids réduit le nombre de paramètres à apprendre ; le pooling réduit la taille spatiale de la sortie : deux mécanismes distincts, à ne pas confondre.
- Le tournant ImageNet de 2012 a montré la supériorité pratique de cette architecture, portée par les GPU, ReLU, le dropout et l'augmentation de données.

Prochain article : on quitte le terrain des images fixes pour celui des séquences (texte et parole) avec les réseaux de neurones récurrents et les représentations distribuées de mots.

---

*Le code accompagnant cet article, une implémentation d'un petit ConvNet from scratch, avec convolution, ReLU et max pooling codés à la main, est disponible sur [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/convnet-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
