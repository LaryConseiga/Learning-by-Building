---
title: "Le catastrophic forgetting : pourquoi les réseaux de neurones oublient tout d'un coup"
date: 2026-09-09 10:00:00 +0000
categories: [Deep Learning, Continual Learning]
tags: [catastrophic-forgetting, connexionnisme, neural-networks, memoire]
---

*🇬🇧 [English version]({{ '/posts/catastrophic-forgetting-why-neural-networks-forget-everything-all-at-once/' | relative_url }})*

## Introduction : un problème vieux de trente ans, toujours pas résolu

Avant de parler des derniers papiers sur le continual learning, il faut remonter à la source. En 1999, Robert French publie dans *Trends in Cognitive Sciences* une revue qui fait le point sur dix ans de recherche autour d'un problème qui semble tout bête au premier abord, mais qui s'avère être l'un des plus tenaces du domaine : le **catastrophic forgetting** (l'oubli catastrophique).

L'idée a été mise en lumière à la fin des années 1980 par McCloskey et Cohen, puis par Ratcliff. Le constat était simple et un peu inquiétant : quand un réseau de neurones à rétropropagation apprend un nouvel ensemble d'informations, il peut **effacer d'un coup, presque intégralement**, tout ce qu'il savait avant. Pas un oubli progressif comme celui qu'on connaît nous-mêmes (on oublie petit à petit le prénom d'un ancien camarade de classe), mais un vrai effondrement.

C'est ce mot "catastrophique" qui donne le ton. Et comprendre pourquoi ça arrive, c'est comprendre une tension fondamentale qui traverse encore aujourd'hui tout le champ du continual learning.

## Le vrai problème : la stabilité contre la plasticité

French relie ce phénomène à un problème plus général, connu depuis longtemps en sciences cognitives : le **problème stabilité-plasticité**. Un système de mémoire, pour être utile, doit réussir un grand écart :

- être **plastique**, donc capable d'apprendre de nouvelles choses,
- être **stable**, donc capable de ne pas perdre ce qu'il savait déjà.

Le problème, c'est que dans un réseau de neurones classique, ces deux exigences tirent dans des directions opposées, et voici pourquoi.

### L'histoire des poids partagés

Un réseau de neurones distribué (comme un simple réseau à rétropropagation) n'a pas un tiroir de mémoire par information apprise. Il a un jeu de poids, partagé par tous les patterns qu'il apprend. C'est ce qui le rend efficace : pas besoin de stocker chaque exemple individuellement, et surtout, c'est ce qui lui donne sa capacité à **généraliser**, c'est-à-dire à bien réagir face à une entrée qu'il n'a jamais vue exactement, du moment qu'elle ressemble à quelque chose de connu.

Le souci, c'est que ce même poids sert à plusieurs patterns différents. Imagine un même bouton de réglage utilisé pour ajuster deux instruments de musique différents. Si l'instrument A a besoin que le bouton soit sur 7, mais que l'instrument B a besoin qu'il soit sur 3, chaque fois que tu règles le bouton pour B, tu désaccordes A.

C'est exactement ce qui se passe pendant l'apprentissage. Quand le réseau apprend un nouveau pattern B, la rétropropagation ajuste les poids pour que B soit bien appris, sans se soucier de savoir si ces poids servaient aussi à A. Résultat : A se dégrade.

### Pourquoi "catastrophique" et pas juste "un peu gênant"

Là où ça devient intéressant, c'est que ce phénomène pourrait très bien être graduel et gérable, comme l'interférence qu'on observe chez l'humain (French cite une vieille expérience de Barnes et Underwood, où des sujets humains apprenant une nouvelle liste de mots oublient progressivement l'ancienne, sans jamais tout perdre d'un coup).

Mais chez les réseaux de neurones, ce n'est pas ce qui se passe. French rapporte une expérience de Kolen et Pollack (1990), menée sur un réseau minuscule (deux neurones d'entrée, deux cachés, un de sortie) apprenant la fonction logique XOR, l'un des exemples les plus simples qui existent en apprentissage automatique. Résultat : même sur ce jouet minuscule, ils montrent que l'espace des poids possibles contient des zones qu'on peut appeler des "falaises de poids". Un tout petit déplacement dans cet espace, presque imperceptible, peut faire radicalement diverger le comportement du réseau.

Imagine que tu marches sur un sentier de montagne dans le brouillard. La plupart du temps, un petit pas de travers ne change rien. Mais à certains endroits précis, ce même petit pas te fait tomber d'une falaise. C'est cette imprévisibilité du terrain qui transforme une interférence banale en effondrement brutal. Si le terrain était partout lisse et prévisible, on n'aurait qu'une interférence graduelle, comme chez l'humain. Ce sont ces falaises cachées qui rendent le problème catastrophique.

## Comment on mesure l'oubli (et pourquoi c'est plus subtil qu'il n'y paraît)

Pour étudier ce phénomène, encore fallait-il le mesurer. La méthode originale de McCloskey et Cohen et de Ratcliff s'appelle la **reconnaissance exacte** : on reprend un vieux pattern, on le fait passer dans le réseau, et on regarde si chaque neurone de sortie reste à moins de 0.5 de sa valeur d'origine. Si un seul neurone dépasse ce seuil, le pattern entier est déclaré "oublié".

Le problème avec ce genre de critère tout ou rien, c'est qu'il écrase toute nuance. Imagine un pattern dont un seul neurone de sortie sur dix dépasse légèrement le seuil, alors que les neuf autres sont parfaits : il sera classé "complètement oublié", ce qui surestime la casse réelle. À l'inverse, un pattern peut dériver franchement sur plusieurs neurones (par exemple passer de 0.9 à 0.6) sans jamais dépasser le seuil de 0.5, et il sera classé "pas oublié du tout", ce qui sous-estime la dégradation réelle.

Une mesure plus fine, complémentaire, consiste à regarder le **temps de réapprentissage** : combien de cycles d'entraînement faut-il au réseau pour retrouver son niveau de performance initial sur le vieux pattern ? Si l'information a été réellement détruite, le réapprentissage prendra à peu près aussi longtemps qu'un apprentissage depuis zéro. Si une trace subsiste dans les poids, le réapprentissage sera rapide, presque comme si le réseau se "souvenait" un peu. C'est une mesure continue, qui capture le degré d'oubli plutôt qu'un simple verdict binaire.

## Les premières solutions : réduire le chevauchement

Une fois le problème bien identifié, plusieurs chercheurs des années 90 ont proposé des solutions qui, en apparence, n'ont rien à voir entre elles, mais qui poursuivent en réalité toutes la même idée de fond.

- **Kortge (1990)** modifie la règle d'apprentissage avec des "novelty vectors" : au lieu de corriger toutes les unités actives quand le réseau se trompe, on ne corrige que celles qui sont vraiment responsables de l'erreur.
- **French lui-même (1991, 1992)** propose l'**activation sharpening** : un mécanisme qui accentue l'activité des neurones cachés déjà les plus actifs pour un pattern donné, et réduit celle des autres, pour créer des représentations plus "éparses".
- **McRae et Hetherington (1993)** montrent que si on pré-entraîne le réseau sur un échantillon aléatoire d'un domaine structuré (comme le langage), le catastrophic forgetting disparaît quasiment pour la suite de l'apprentissage.

Le fil conducteur entre ces trois approches, c'est de **réduire le chevauchement (overlap) des représentations internes** entre les patterns différents. Si deux patterns activent des neurones cachés différents, ils sollicitent des poids différents, et apprendre l'un n'écrase plus l'autre. C'est comme si, au lieu d'avoir tout le monde qui se bat pour les mêmes deux ou trois boutons de réglage, chaque groupe de patterns avait ses propres boutons dédiés, avec juste un peu de recoupement là où c'est utile pour généraliser.

French appelle ça des représentations **"semi-distribuées"** : ni complètement locales (un neurone unique par pattern, plus de généralisation possible), ni complètement distribuées (chevauchement total, risque catastrophique), mais un compromis entre les deux.

### Le cas extrême : tout localiser

Certains modèles, comme **CALM** ou **ALCOVE**, poussent cette logique jusqu'au bout. Dans ALCOVE par exemple, l'activation d'un neurone caché dépend de sa distance à l'entrée : chaque neurone "couvre" une zone limitée de l'espace des entrées possibles, un peu comme un projecteur qui n'éclaire qu'un petit coin de la scène. Si on règle ce projecteur pour qu'il soit très étroit, le réseau devient quasiment local, et le catastrophic forgetting disparaît presque totalement.

Mais rien n'est gratuit. Ce qui faisait la force des réseaux distribués depuis le début, c'est justement leur capacité à généraliser, à réagir intelligemment face à une entrée jamais vue mais proche de quelque chose de connu, ou face à une entrée bruitée ou incomplète. Si on va trop loin dans le "tout local", chaque pattern devient une île isolée : plus de chevauchement, donc plus d'oubli, mais aussi plus vraiment de généralisation. Le réseau ne sait plus quoi faire d'une entrée qu'il n'a pas vue à l'identique.

Et ça pose un problème plus profond que technique. Si l'objectif est de modéliser fidèlement comment un cerveau humain apprend et retient l'information, un système "tout local" n'est pas satisfaisant, parce que l'humain généralise en permanence : on reconnaît un visage à moitié caché, on comprend une phrase qu'on n'a jamais entendue mot pour mot exactement. En sacrifiant la généralisation pour éviter l'oubli, on s'éloigne du comportement qu'on cherche justement à reproduire.

## Le rehearsal : rejouer les vieux souvenirs

Une deuxième famille de solutions ne touche pas à la représentation interne, mais à la façon dont l'apprentissage se déroule. L'idée du **rehearsal** (répétition) : au lieu de n'entraîner le réseau que sur les nouveaux patterns, on mélange les nouveaux patterns avec des anciens patterns déjà appris, un peu comme si on révisait ses anciens cours en même temps qu'on apprend la nouvelle matière.

Le problème pratique est évident dès qu'on sort du labo : cette méthode suppose qu'on a gardé tous les anciens patterns quelque part. Dans un scénario réaliste d'apprentissage continu, ce n'est presque jamais le cas. On n'a pas forcément stocké chaque exemple qu'on a appris il y a des mois, et même si on le pouvait, ce serait coûteux de tout garder indéfiniment.

### L'idée astucieuse des pseudopatterns

C'est là qu'intervient une innovation proposée par Robins en 1995 : les **pseudopatterns**. L'idée est de contourner complètement le besoin de garder les vraies anciennes données. Voici comment ça marche :

1. On prend le réseau déjà entraîné, qui a appris une certaine fonction (une certaine façon d'associer des entrées à des sorties).
2. On génère une entrée **complètement aléatoire**.
3. On la fait passer dans le réseau, qui produit une sortie.
4. Cette paire (entrée aléatoire, sortie produite) forme un **pseudopattern**.

Pourquoi est-ce que ça marche ? Parce que la sortie produite pour une entrée aléatoire n'est pas aléatoire elle-même : elle reflète fidèlement la fonction que le réseau a intériorisée dans ses poids, c'est-à-dire sa façon de généraliser, façonnée par tout ce qu'il a appris jusque là. Le pseudopattern capture donc une sorte d'empreinte de la connaissance du réseau, sans jamais avoir besoin de stocker un vrai exemple d'origine.

On mélange ensuite ces pseudopatterns avec les nouveaux patterns pendant l'entraînement. Le réseau continue ainsi de "voir" un résumé condensé de ce qu'il savait avant, ce qui empêche ses poids de dériver trop loin de la solution d'origine.

### Quand ça tourne mal : le catastrophic remembering

French pointe une limite amusante et instructive à cette technique, qu'il appelle le **catastrophic remembering** (l'ironie du titre est assumée). Imagine un réseau auto-associatif, dont la tâche est de reproduire en sortie exactement ce qu'on lui donne en entrée. Le réseau "sait" qu'il a déjà vu un pattern si, en le lui redonnant, la sortie ressemble beaucoup à l'entrée.

Si on génère et repasse des pseudopatterns en boucle, sur de nombreux cycles, le réseau devient de plus en plus doué pour généraliser, au point de reproduire fidèlement n'importe quelle entrée, même du pur bruit qu'il n'a jamais vu. Le problème, c'est qu'à ce moment-là, le réseau perd sa capacité de **discrimination** : il ne sait plus distinguer un pattern qu'il a réellement appris d'une entrée aléatoire qui ressemble juste, par accident, à quelque chose de connu. Tout finit par lui sembler familier.

C'est un nouveau visage du même dilemme stabilité-plasticité, mais cette fois sur l'axe mémorisation contre discrimination plutôt que stabilité contre généralisation.

## La séparation en deux systèmes : s'inspirer du cerveau

La dernière grande piste que French développe, avec Ans et Rousset de leur côté, est architecturale : plutôt qu'un seul réseau qui doit tout faire, on utilise **deux réseaux séparés**. L'un pour le traitement rapide de l'information récente, l'autre pour le stockage à long terme des régularités déjà consolidées.

Cette idée est reprise et justifiée biologiquement par McClelland, McNaughton et O'Reilly (1995), qui l'ancrent dans une distinction bien connue en neurosciences : **l'hippocampe** (apprentissage rapide de nouvelles informations spécifiques) et le **néocortex** (découverte lente et progressive de structures générales, stockage à long terme). Leur argument est que ces deux fonctions sont fondamentalement incompatibles dans un seul et même système : apprendre vite du nouveau, spécifique, et découvrir lentement des régularités générales, ça tire dans des directions opposées, un peu comme essayer de sprinter et de marcher lentement en même temps avec les mêmes jambes.

Le lien avec les pseudopatterns devient alors évident. Si le réseau "long terme" continue de tourner en parallèle et génère ses propres pseudopatterns pour se consolider, pendant que le réseau "récent" apprend les nouveaux patterns, on évite complètement le mélange des genres qu'on avait vu dans le catastrophic remembering. Chaque réseau a un rôle clair : l'un encode le nouveau, l'autre digère et consolide l'ancien, sans que les deux flux d'information ne se télescopent dans le même espace de poids.

## Ce qui restait ouvert en 1999

French termine son article par une liste de questions non résolues à l'époque. Une des plus intéressantes concerne justement les pseudopatterns : est-ce que ce mécanisme, purement informatique au départ, a un véritable équivalent biologique ?

Autrement dit, est-ce que le cerveau génère vraiment quelque chose comme des "entrées aléatoires internes" qu'il repasserait dans ses propres circuits pour se consolider ? Et si oui, quand ? French avance une piste : peut-être pendant le **sommeil paradoxal (REM)**, ce moment où le cerveau reste actif mais coupé des stimulations extérieures, ce qui correspondrait assez bien à l'idée de générer des entrées qui ne viennent pas du monde réel. Il ajoute une nuance importante : si ce mécanisme existe vraiment, rien ne dit qu'il serait purement aléatoire comme dans les modèles informatiques. Le cerveau a peut-être évolué une façon plus intelligente de faire cette répétition interne, en rejouant préférentiellement ce qui compte le plus plutôt que du bruit pur.

Et toi, tu penses que tes rêves rejouent vraiment tes souvenirs au hasard, ou qu'ils trient déjà ce qui mérite d'être retenu ?
