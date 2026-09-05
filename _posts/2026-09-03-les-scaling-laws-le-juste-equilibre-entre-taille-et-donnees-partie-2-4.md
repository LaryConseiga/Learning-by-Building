---
title: "Les Scaling Laws : le juste équilibre entre taille et données (Partie 2/4)"
date: 2026-09-03 10:00:00 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, semaine-4]
math: true
---

*🇬🇧 [English version]({{ '/posts/scaling-laws-the-right-balance-between-size-and-data-part-2-4/' | relative_url }})*

Deuxième partie de cette série sur les Scaling Laws. Dans la [Partie 1]({{ '/posts/les-scaling-laws-pourquoi-la-taille-compte-plus-que-le-talent-partie-1-4/' | relative_url }}), on a vu les trois lois de base et pourquoi la forme du modèle compte moins que son échelle. Aujourd'hui, on regarde ce qui se passe quand deux de ces facteurs bougent en même temps.

## Le problème qu'on n'a pas encore résolu

Dans la Partie 1, on a vu trois lois séparées : L(N), L(D) et L(C). Chacune suppose que les deux autres facteurs sont "infinis", c'est-à-dire qu'ils ne posent pas de problème.

Mais dans la vraie vie, tu n'as jamais un budget de données infini pendant que tu fais grossir ton modèle. Alors la vraie question devient : **qu'est-ce qui se passe quand N et D bougent en même temps ?**

C'est le sujet de cette deuxième partie, et c'est là qu'on tombe sur l'un des résultats les plus utiles du papier au quotidien.

## Le piège classique : le surapprentissage

Tu connais probablement déjà l'idée du surapprentissage (overfitting). Un modèle trop gros pour la quantité de données qu'il voit finit par les mémoriser par cœur, plutôt que d'apprendre des règles générales. Résultat : il est excellent sur ses données d'entraînement, mais catastrophique sur tout ce qu'il n'a jamais vu.

C'est un peu comme un élève qui apprend par cœur les réponses d'un examen blanc précis, sans comprendre le raisonnement derrière. Le jour de l'examen réel, avec des questions légèrement différentes, il est perdu.

## L'équation qui unifie tout

Les auteurs proposent une formule qui capture le comportement de la loss quand N et D varient ensemble :

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

![Loss en fonction de N pour différentes valeurs de D, et ampleur du surapprentissage en fonction du ratio D/N](/assets/img/posts/scaling-laws-overfitting-regimes.png)
_Figure 9, page 11. On y voit très bien les deux régimes : à gauche, pour un D fixé, la performance stagne dès que N devient trop grand (signe de surapprentissage). À droite, l'ampleur du surapprentissage suit une courbe unique et prévisible._

Cette formule n'est pas sortie de nulle part. Elle respecte trois règles de bon sens.

Premièrement, si tu changes la façon dont tu découpes ton texte en tokens, la loss doit juste se décaler globalement, sans que la forme de l'équation change.

Deuxièmement, si tu as des données infinies (D tend vers l'infini), l'équation doit se réduire à L(N) tout seul. Et si tu as un modèle infiniment grand (N tend vers l'infini), elle doit se réduire à L(D) tout seul. Ça colle avec ce qu'on a vu dans la Partie 1.

Troisièmement, l'équation doit rester bien élevée mathématiquement même dans les cas extrêmes, ce qui est un principe plus technique mais qui garantit la cohérence du modèle.

## La règle pratique à retenir

De cette équation découle une règle magique pour savoir combien de données il te faut pour un modèle donné, si tu veux éviter le surapprentissage :

$$D \gtrsim (5 \times 10^3) \, N^{0.74}$$

Voici ce qui est contre-intuitif là-dedans. On pourrait penser que si tu multiplies la taille de ton modèle par 8, il te faut aussi 8 fois plus de données pour rester safe. Ce n'est pas le cas. Il te suffit d'environ **5 fois plus de données**, pas 8 fois plus.

Le besoin en données croît **plus lentement** que la taille du modèle.

## Pourquoi cette relation est sous-linéaire

Voici l'intuition clé. N et D n'apportent pas le même type de valeur au modèle.

Une partie de la capacité supplémentaire d'un grand modèle sert à mieux **généraliser** les régularités déjà présentes dans les données existantes. Cette partie-là n'a pas besoin de données neuves pour être utile.

Une autre partie de cette capacité supplémentaire sert effectivement à absorber de nouvelles informations, et c'est cette partie qui aurait besoin de plus de données pour ne pas finir mémorisée par cœur plutôt que généralisée.

Résultat : doubler la taille du modèle ne double pas le risque de mémorisation pure. Une partie de cette taille supplémentaire est absorbée par une meilleure compression des patterns déjà présents dans les données. C'est pour ça que la relation entre N et D est sous-linéaire.

Imagine deux étudiants qui révisent le même cours. Le premier a une mémoire moyenne et doit tout retenir mot pour mot. Le second a une excellente capacité d'analyse et arrive à extraire les grands principes du cours, ce qui lui permet de répondre correctement même à des questions qu'il n'a jamais vues sous cette forme précise. Le second étudiant "généralise" mieux avec la même quantité de matière étudiée. C'est un peu ce qui se passe quand tu augmentes N : tu donnes au modèle une meilleure capacité de généralisation, pas juste plus de mémoire brute.

## Ce que ça change concrètement

Cette relation offre un guide très concret. Si tu sais combien de paramètres aura ton modèle, tu peux estimer directement combien de tokens minimum il te faut pour ne pas gaspiller sa capacité dans du par cœur inutile.

C'est un résultat qui, quelques années plus tard, a directement inspiré des travaux comme Chinchilla (DeepMind, 2022), qui a affiné encore cette relation entre taille de modèle et taille de dataset. On y reviendra probablement dans une future session de lecture.

## Ce qu'il faut retenir de cette partie

Le surapprentissage dépend d'une combinaison précise de N et D, pas de l'un ou l'autre isolément.

La quantité de données nécessaire croît plus lentement que la taille du modèle, dans un rapport approximatif de $N^{0.74}$.

Cette sous-linéarité vient du fait qu'une partie de la capacité d'un grand modèle sert à mieux généraliser les données existantes, pas seulement à en absorber davantage.

Dans la [**Partie 3**]({{ '/posts/les-scaling-laws-comment-depenser-son-budget-de-calcul-partie-3-4/' | relative_url }}), on attaque le morceau le plus concret et le plus cité du papier : comment répartir un budget de calcul fixe entre taille du modèle, taille de batch et nombre d'étapes d'entraînement. C'est ce résultat-là qui a justifié, historiquement, la course aux modèles toujours plus gros.

## Section 5 : le temps d'entraînement entre aussi dans l'équation

Avant d'arriver à l'allocation du compute dans la Partie 3, il reste un ingrédient à poser : le temps d'entraînement, ou plus précisément le nombre d'étapes d'entraînement (S).

### Le problème de la taille de batch

Pour comparer proprement des modèles entraînés à des vitesses différentes, il faut d'abord régler un détail technique. La plupart des modèles étudiés dans ce papier n'ont pas été entraînés avec une taille de batch optimale. Il existe une taille de batch dite "critique", $B_{crit}$, en dessous de laquelle augmenter le batch permet d'aller plus vite presque sans perte d'efficacité, et au-dessus de laquelle les gains de vitesse plafonnent.

$$B_{crit}(L) \approx \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

Point intéressant : $B_{crit}$ ne dépend que de la loss actuelle, pas de la taille du modèle. Plus le modèle progresse (loss basse), plus on peut se permettre des batchs énormes.

![Taille de batch critique en fonction de la loss](/assets/img/posts/scaling-laws-critical-batch-size.png)
_Figure 10, page 12. La taille de batch critique suit une loi de puissance très nette en fonction de la loss._

### Une mesure de temps universelle

Pour comparer des runs entraînés à des batchs différents sur un pied d'égalité, les auteurs définissent $S_{min}$, une sorte de "nombre de pas équivalent" si on avait entraîné le modèle à la meilleure vitesse théorique possible.

L'équation combinée devient alors :

$$L(N, S_{min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{min}}\right)^{\alpha_S}$$

![Courbes d'apprentissage de tous les modèles superposées une fois ajustées avec S_min](/assets/img/posts/scaling-laws-loss-vs-smin.png)
_Figure 4 (droite), page 5. Les courbes d'apprentissage de tous les modèles, une fois ajustées avec $S_{min}$, se superposent selon cette même formule._

Remarque intéressante ici : contrairement à l'équation L(N, D) vue plus haut, c'est une simple **addition** des deux termes, pas une puissance globale qui les enveloppe tous les deux. Ça traduit le fait que taille du modèle et temps d'entraînement jouent des rôles plutôt indépendants. Avoir un petit modèle et avoir fait peu d'étapes sont deux défauts distincts qui s'additionnent, sans effet démultiplicateur entre eux, contrairement au couple N et D où un petit D peut complètement annuler l'intérêt d'un grand N.

Cette brique va être essentielle pour la Partie 3, où on va enfin répondre à la question qui a lancé toute la course aux grands modèles : si j'ai un budget de calcul fixe, comment je le répartis au mieux ?
