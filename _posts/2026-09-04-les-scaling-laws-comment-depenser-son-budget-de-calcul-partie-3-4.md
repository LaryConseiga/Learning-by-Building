---
title: "Les Scaling Laws : comment dépenser son budget de calcul (Partie 3/4)"
date: 2026-09-04 10:00:00 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, semaine-4]
math: true
---

*🇬🇧 [English version]({{ '/posts/scaling-laws-how-to-spend-your-compute-budget-part-3-4/' | relative_url }})*

Troisième partie de cette série sur les Scaling Laws. Dans la [Partie 2]({{ '/posts/les-scaling-laws-le-juste-equilibre-entre-taille-et-donnees-partie-2-4/' | relative_url }}), on a vu comment N et D interagissent, et posé la notion de $S_{min}$. Aujourd'hui, on répond à la question qui a lancé la course aux grands modèles.

## La question à un million de dollars (littéralement)

On y arrive enfin. C'est le passage du papier qui a le plus influencé la façon dont l'industrie entraîne ses modèles depuis 2020.

Imagine que tu as un budget de calcul fixe. Disons l'équivalent de X heures de GPU. Comment tu le répartis au mieux entre trois choses : la taille de ton modèle, la taille de tes batchs, et le nombre d'étapes d'entraînement ?

Avant ce papier, l'intuition dominante était plutôt simple : tu prends un modèle raisonnable, et tu l'entraînes jusqu'à convergence, c'est-à-dire jusqu'à ce que la loss arrête de baisser. Ce papier montre que cette intuition est en fait assez mauvaise.

## Le résultat central

En combinant tout ce qu'on a vu dans les parties précédentes, notamment l'équation L(N, $S_{min}$) de la fin de la Partie 2, les auteurs obtiennent la formule suivante pour le compute optimal :

$$L(C_{min}) = \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_C^{min}}, \quad \alpha_C^{min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.050$$

Cette formule pour $\alpha_C^{min}$ est plutôt élégante. C'est une sorte de moyenne qui combine la façon dont la taille du modèle, la taille de batch et le nombre d'étapes contribuent chacun à utiliser le compute disponible.

![Loss en fonction du compute optimal, ajustée à la vitesse d'entraînement idéale](/assets/img/posts/scaling-laws-optimal-compute-fit.png)
_Figure 13, page 15. Une fois ajustée pour un entraînement à la vitesse optimale, la courbe de performance en fonction du compute devient encore plus nette que dans la Figure 1 de la Partie 1._

## Le vrai scoop : où doit aller ton budget supplémentaire

Voici la partie qui a vraiment changé la donne dans l'industrie. Quand ton budget de compute augmente, comment répartir cette augmentation ?

$$N \propto C^{0.73}, \qquad B \propto C^{0.24}, \qquad S \propto C^{0.03}$$

![Croissance de la taille optimale du modèle et du nombre d'étapes en fonction du compute](/assets/img/posts/scaling-laws-optimal-allocation-growth.png)
_Figure 14, page 16. La taille optimale du modèle grandit très vite avec le compute disponible, alors que le nombre d'étapes reste presque constant._

Pour donner une idée concrète : si ton budget de compute est multiplié par un **milliard**, alors la taille de ton modèle doit grossir d'environ **un million de fois**. La taille de batch, elle, doit grossir d'environ 100 fois. Et le nombre d'étapes d'entraînement séquentielles ? À peine moins de 10 fois.

![Répartition de l'augmentation du compute entre taille du modèle, batch et étapes séquentielles](/assets/img/posts/scaling-laws-compute-allocation.png)
_Figure 3, page 4. Cette figure résume tout en une image : sur une augmentation de compute d'un milliard de fois, quasiment tout doit servir à agrandir le modèle, une petite partie à augmenter le batch, et presque rien à augmenter le nombre d'étapes séquentielles._

## Pourquoi c'est une excellente nouvelle en pratique

Il y a une distinction cruciale entre le compute disponible et le temps réellement nécessaire pour l'utiliser.

Le nombre d'étapes séquentielles, c'est le nombre d'itérations qui doivent se faire les unes après les autres, dans l'ordre. Tu ne peux pas calculer l'étape 501 avant d'avoir terminé l'étape 500, puisque chaque étape met à jour les poids qui seront utilisés à l'étape suivante.

En revanche, la taille de batch se parallélise facilement. Si tu as mille machines disponibles, tu peux leur donner chacune un morceau différent du même batch et tout calculer en même temps.

Reprenons l'image d'un déménagement. Si la charge à déplacer augmente, tu as deux options. Tu peux faire plus de voyages les uns après les autres, ce qui prend plus de temps. Ou tu peux prendre plus de camions en même temps, ce qui prend plus de ressources mais pas plus de temps.

Le résultat de ce papier dit que la meilleure stratégie, c'est presque exclusivement la deuxième option. Ça veut dire qu'on peut entraîner des modèles gigantesques sans que le temps d'entraînement n'explose de façon ingérable, à condition d'avoir assez de machines disponibles en parallèle.

## Convergence inefficace : la conclusion la plus contre-intuitive du papier

Voilà où tout se rejoint. Puisque N doit grossir extrêmement vite avec le compute, et S (le nombre d'étapes) doit grossir très lentement, la conséquence directe, c'est qu'un grand modèle avec un budget de compute donné va forcément faire beaucoup moins d'étapes qu'un petit modèle. Il va donc s'arrêter loin de la convergence complète.

![Comparaison de la performance des grands et petits modèles selon le nombre de tokens traités](/assets/img/posts/scaling-laws-sample-efficiency.png)
_Figure 2, page 4. On voit ici que les grands modèles atteignent un excellent niveau de performance en traitant beaucoup moins de tokens que les petits modèles._

Autrement dit, la meilleure stratégie n'est pas d'entraîner un petit modèle jusqu'à ce qu'il ait tout donné. C'est d'entraîner un très grand modèle, et de l'arrêter volontairement bien avant qu'il ait fini de converger.

Pourquoi ça marche ? Parce qu'un grand modèle est beaucoup plus efficace par échantillon vu. Chaque étape d'entraînement lui fait gagner davantage en performance qu'elle n'en ferait gagner à un petit modèle. Même avec seulement quelques milliers d'étapes, un grand modèle dépasse largement un petit modèle poussé jusqu'à sa limite absolue, car ce dernier plafonne à cause de sa capacité limitée, peu importe combien de temps on le laisse tourner.

C'est un peu comme comparer deux athlètes. Le premier a un potentiel physique limité et atteint très vite son plafond de performance, peu importe combien il s'entraîne. Le second a un potentiel énorme et continue de progresser rapidement à chaque séance, même s'il n'a pas encore accumulé autant d'heures d'entraînement que le premier. Miser sur le second, même avec moins de séances, donne de meilleurs résultats.

## Ce qu'il faut retenir de cette partie

Face à un budget de compute fixe, la meilleure stratégie consiste à faire grossir le modèle massivement, augmenter modérément la taille de batch, et augmenter à peine le nombre d'étapes séquentielles.

Cette répartition est une excellente nouvelle en pratique, car elle permet d'absorber l'essentiel de la croissance du compute via la parallélisation, plutôt que via un temps d'entraînement qui exploserait.

La stratégie optimale consiste à entraîner un très grand modèle et à s'arrêter volontairement bien avant sa convergence complète, plutôt que d'entraîner un petit modèle jusqu'au bout.

Dans la [**Partie 4**]({{ '/posts/les-scaling-laws-ce-que-ca-change-pour-la-suite-partie-4-4/' | relative_url }}), la dernière de cette série, on va prendre un peu de recul avec la discussion du papier. On y parlera notamment d'une idée qui deviendra centrale plus tard dans la recherche sur les LLM : le fait que des améliorations lisses et prévisibles de la performance peuvent cacher des sauts qualitatifs soudains de capacité. C'est le fameux concept des capacités émergentes.
