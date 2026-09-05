---
title: "Les Scaling Laws : ce que ça change pour la suite (Partie 4/4)"
date: 2026-09-05 10:00:00 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, semaine-4]
math: true
---

*🇬🇧 [English version]({{ '/posts/scaling-laws-what-this-means-going-forward-part-4-4/' | relative_url }})*

Quatrième et dernière partie de cette série sur les Scaling Laws. Dans la [Partie 3]({{ '/posts/les-scaling-laws-comment-depenser-son-budget-de-calcul-partie-3-4/' | relative_url }}), on a vu comment répartir un budget de calcul fixe. On termine aujourd'hui avec la discussion du papier.

## On referme le papier, mais pas le sujet

On a passé trois articles à décortiquer les lois de puissance de Kaplan et al. Dans cette dernière partie, on prend un peu de recul. Qu'est-ce que les auteurs eux-mêmes pensent de leurs résultats, où sont les limites, et surtout, qu'est-ce que ça a changé pour la suite de la recherche en IA ?

## Une loi sans théorie derrière

Voici un aveu assez honnête des auteurs. Ils comparent leurs lois de scaling à la loi des gaz parfaits en physique. C'est une loi macroscopique universelle, qui décrit très bien le comportement d'un gaz sans dépendre des détails microscopiques précis de chaque molécule.

Le problème, c'est qu'à l'époque où la loi des gaz parfaits a été découverte, on n'avait pas encore de théorie sous-jacente pour l'expliquer depuis les premiers principes. Cette théorie, la mécanique statistique, est arrivée plus tard.

C'est exactement la position dans laquelle se trouvent les auteurs de ce papier. Ils observent un phénomène solide, reproductible, mesuré sur sept ordres de grandeur. Mais ils n'ont pas d'explication fondamentale de pourquoi ça marche ainsi. On observe la thermodynamique du deep learning, sans encore avoir sa mécanique statistique.

## L'idée la plus visionnaire du papier : "more is different"

C'est probablement le passage le plus important de toute la discussion, et pourtant il tient en quelques phrases.

Les auteurs notent qu'une amélioration lisse et continue de la loss peut masquer des changements qualitatifs de capacité. Ils prennent l'exemple de la croissance économique globale. Vue de loin, elle semble lisse et continue. Mais cette courbe lisse ne révèle rien des ruptures technologiques spécifiques qui la sous-tendent en réalité, comme l'apparition d'internet ou de l'électricité.

Applique la même idée à un modèle de langage. Sa loss baisse de façon régulière et prévisible à mesure qu'il grossit. Mais cette baisse continue peut très bien cacher l'apparition soudaine de capacités totalement nouvelles, qui n'existaient tout simplement pas dans les modèles plus petits.

Cette intuition deviendra centrale quelques années plus tard sous le nom de capacités émergentes, un concept qui a été popularisé notamment par des travaux étudiant à partir de quelle taille certains modèles deviennent soudainement capables de résoudre des problèmes de raisonnement, de faire de l'arithmétique fiable, ou de suivre des instructions complexes, alors que des modèles légèrement plus petits en étaient incapables.

C'est un excellent candidat pour une future session de lecture, où on pourrait relier ce papier de 2020 aux travaux plus récents sur les capacités émergentes, ainsi qu'à Chinchilla, qui a affiné la relation optimale entre taille de modèle et taille de dataset qu'on a vue dans la Partie 2.

## Big models, big deal

La conclusion pratique la plus directe des auteurs tient en une phrase qu'on pourrait résumer ainsi : les gros modèles seront beaucoup plus efficaces par échantillon vu qu'on ne le pensait auparavant.

Avec le recul qu'on a aujourd'hui, ce résultat a été un signal extrêmement fort. Il a directement motivé la direction prise par la suite avec des modèles comme GPT-3, puis toute la génération de grands modèles de langage qu'on connaît maintenant. L'idée que "plus gros, c'est mieux, et de façon prévisible" a servi de justification empirique solide à des investissements massifs dans des modèles toujours plus grands.

## Une contradiction à très grande échelle

Un point plus technique mais intéressant : les auteurs remarquent eux-mêmes que leurs équations, poussées très loin au-delà des échelles qu'ils ont testées, finissent par se contredire entre elles.

![Intersection entre L(C_min) et L(D(C)) à très grande échelle](/assets/img/posts/scaling-laws-extrapolation-limit.png)
_Figure 15, page 17. L'intersection entre les deux courbes marque le point où les prédictions du papier commencent à se contredire._

En gros, la quantité de données nécessaire pour un entraînement optimal en compute croît trop lentement par rapport à ce qu'il faudrait pour éviter le surapprentissage à très grande échelle. Les deux tendances finissent par se croiser à des échelles absolument gigantesques, autour de $10^{12}$ paramètres et $10^{12}$ tokens.

Les auteurs interprètent prudemment ce point d'intersection comme une estimation possible de la limite ultime des performances d'un modèle de langage sur du texte naturel, peut-être liée à l'entropie intrinsèque du langage lui-même. Mais ils restent honnêtes sur le fait que cette extrapolation est très incertaine, et que leurs lois de scaling doivent probablement s'infléchir bien avant d'atteindre ce point.

## Les limites que les auteurs reconnaissent eux-mêmes

Le papier se termine sur une liste de mises en garde assez transparentes. Il n'existe pas de théorie solide sous-jacente pour expliquer pourquoi ces lois de puissance apparaissent. Le régime des très petits datasets n'a pas été bien exploré. Le comportement de la taille de batch critique reste incertain loin de la plage testée. Et surtout, toutes ces relations n'ont pas été vérifiées à des échelles beaucoup plus grandes que celles étudiées dans le papier, ce qui laisse la porte ouverte à des surprises.

Cette honnêteté scientifique fait partie de ce qui rend ce papier solide : les auteurs ne prétendent jamais avoir trouvé une loi universelle et définitive, seulement une tendance empirique très robuste dans une plage donnée.

## Ce qu'il faut retenir de toute cette série

On a vu que la performance d'un modèle de langage dépend principalement de trois facteurs d'échelle, taille du modèle, taille des données, et compute, et que cette dépendance suit des lois de puissance remarquablement régulières.

On a vu que la forme précise du modèle compte peu, mais que le type d'architecture, lui, compte beaucoup, notamment pour gérer les contextes longs.

On a vu que la quantité de données nécessaire croît plus lentement que la taille du modèle, ce qui a des implications concrètes pour éviter le surapprentissage.

On a vu que, face à un budget de compute fixe, la meilleure stratégie consiste à privilégier massivement la taille du modèle plutôt que le temps d'entraînement, ce qui justifie la stratégie de "gros modèle, arrêt prématuré" plutôt que "petit modèle, entraînement complet".

Et on a vu, pour finir, que des progrès lisses et prévisibles en apparence peuvent cacher des sauts qualitatifs de capacité, une idée qui a ouvert la voie à toute une ligne de recherche sur les capacités émergentes des grands modèles.
