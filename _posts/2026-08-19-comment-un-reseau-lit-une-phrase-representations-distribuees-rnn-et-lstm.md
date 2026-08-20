---
title: "Comment un réseau lit une phrase : représentations distribuées, RNN et LSTM"
date: 2026-08-19 09:30:00 +0000
categories: [Deep Learning, Traitement du langage]
tags: [deep-learning, rnn, lstm, nlp, word-embeddings, attention]
math: true
---

*🇬🇧 [English version]({{ '/posts/how-a-network-reads-a-sentence-distributed-representations-rnn-and-lstm/' | relative_url }})*

## Où on en était

Les [deux articles précédents]({{ '/posts/comment-un-reseau-de-neurones-apprend-vraiment/' | relative_url }}) de cette série ont couvert l'apprentissage supervisé, la rétropropagation, puis les réseaux convolutifs (ConvNets), spécialisés dans le traitement des images. Cet article clôt la lecture du texte fondateur de LeCun, Bengio et Hinton[^lecun2015] en changeant de terrain : le **langage**.

Une image est une donnée spatiale, de taille fixe, où toutes les positions comptent à peu près également. Un texte est fondamentalement différent : c'est une **séquence**, de longueur variable, où l'ordre des éléments porte le sens. Traiter ce type de données demande une approche différente : c'est l'objet de cet article.

## Représenter un mot : au-delà du symbole isolé

### Le problème des N-grams

Avant les réseaux de neurones, l'approche statistique standard pour modéliser le langage comptait les fréquences d'occurrence de courtes séquences de mots, les **N-grams** :

> The number of possible N-grams is on the order of $V^N$, where $V$ is the vocabulary size, so taking into account a context of more than a handful of words would require very large training corpora. N-grams treat each word as an atomic unit, so they cannot generalize across semantically related sequences of words.[^lecun2015]

Le problème est double. D'abord, un problème combinatoire : avec un vocabulaire de $V = 50\,000$ mots et un contexte de $N = 3$ mots, le nombre de combinaisons possibles est $50\,000^3$, un nombre gigantesque. Ensuite, et surtout, un problème de généralisation : chaque mot est un symbole **atomique**, sans lien avec les autres. Le modèle ne "sait" pas que "chien" et "loup" sont sémantiquement proches : ce sont juste deux jetons différents parmi des dizaines de milliers.

### La solution : des représentations distribuées

Les réseaux de neurones résolvent ce problème en abandonnant l'idée d'un symbole isolé, au profit d'une **représentation distribuée** :

> Deep-learning theory shows that deep nets have two different exponential advantages over classic learning algorithms that do not use distributed representations. […] learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example, $2^n$ combinations are possible with $n$ binary features).[^lecun2015]

L'idée : plutôt que d'assigner un neurone unique à chaque concept (une représentation "locale"), on encode un concept comme un **motif d'activation réparti** sur plusieurs unités, chacune représentant une caractéristique partagée entre plusieurs concepts. Avec seulement $n = 10$ caractéristiques binaires indépendantes (a-des-poils, a-quatre-pattes, aboie…), on peut représenter $2^{10} = 1024$ combinaisons différentes, sans avoir eu besoin de voir chacune de ces 1024 combinaisons pendant l'entraînement.

Concrètement, un système à représentation locale, confronté à une catégorie jamais vue (disons "renard"), n'a **aucun moyen structurel** de la représenter. Un système à représentation distribuée peut en revanche construire une représentation cohérente pour "renard" en combinant des caractéristiques déjà apprises séparément sur d'autres animaux (a-museau-pointu, a-une-queue-touffue), même sans avoir jamais vu cette combinaison exacte.

### Les word vectors

Appliqué au langage, ce principe donne les **word vectors** (ou *word embeddings*) :

> Each word in the context is presented to the network as a one-of-N vector, that is, one component has a value of 1 and the rest are 0. In the first layer, each word creates a different pattern of activations, or word vectors.[^lecun2015]

Chaque mot entre dans le réseau sous forme d'un vecteur *one-hot* : un 1 à la position correspondant à ce mot, des 0 partout ailleurs, sur toute la taille du vocabulaire. C'est encore une représentation purement locale à ce stade. La première couche du réseau transforme ensuite ce vecteur creux en un **vecteur dense**, de quelques centaines de dimensions seulement : c'est ce vecteur qui devient la véritable représentation du mot dans le réseau.

Pourquoi cette transformation capture-t-elle la proximité sémantique ? Parce que le réseau, entraîné à une tâche comme prédire le mot suivant dans une phrase, découvre que des mots apparaissant dans des **contextes similaires** sont plus efficacement représentés par des vecteurs proches :

> The network learns word vectors that contain many active components each of which can be interpreted as a separate feature of the word. […] When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday and Wednesday are very similar, as are the word vectors for Sweden and Norway.[^lecun2015]

Le point clé à ne pas confondre : cette proximité vient de la **substituabilité** dans des contextes similaires ("I'll see you on Tuesday" / "I'll see you on Wednesday" fonctionnent de façon interchangeable), pas simplement de la co-occurrence dans un même texte. "Tuesday" et "Sweden" peuvent très bien apparaître dans les mêmes articles de presse sans jamais être substituables l'un à l'autre grammaticalement, leurs vecteurs restent donc éloignés.

## Les réseaux de neurones récurrents (RNN)

### Traiter une séquence élément par élément

Pour exploiter ces représentations dans le traitement d'une phrase entière, il faut une architecture capable de lire une séquence, un élément à la fois, en gardant une trace de ce qui a déjà été vu :

> RNNs process an input sequence one element at a time, maintaining in their hidden units a 'state vector' that implicitly contains information about the history of all the past elements of the sequence.[^lecun2015]

Concrètement, pour une phrase comme "le chat dort" : à chaque pas de temps $t$, le réseau reçoit un mot et met à jour un **vecteur d'état** $s_t$, à partir du mot courant **et** de l'état précédent $s_{t-1}$. L'état $s_3$, après avoir lu "dort", contient donc une trace compressée de toute la phrase "le chat dort".

Point important : les mêmes matrices de poids sont **réutilisées à chaque pas de temps** : c'est le même principe de partage de poids que celui vu pour les ConvNets, mais appliqué ici dans la dimension temporelle plutôt que spatiale.

### Le "dépliage" et la rétropropagation dans le temps

Cette structure permet d'appliquer directement la rétropropagation vue dans le premier article :

> When we consider the outputs of the hidden units at different discrete time steps as if they were the outputs of different neurons in a deep multilayer network, it becomes clear how we can apply backpropagation to train RNNs.[^lecun2015]

Si on "déplie" le réseau dans le temps (une copie du réseau par pas de temps, mises bout à bout), la structure ressemble à un réseau profond classique, à la différence près que tous ces étages partagent les mêmes poids.

### Le problème des séquences longues

Cette analogie avec un réseau très profond a une conséquence directe : une phrase de 100 mots équivaut, une fois dépliée, à un réseau de 100 couches. Or, on a vu dans le premier article que la rétropropagation multiplie l'erreur par un facteur à chaque couche traversée, et que ce facteur, s'il est systématiquement inférieur à 1, écrase le gradient de façon exponentielle en remontant vers les premières couches (le *vanishing gradient*).

> RNNs are very powerful dynamic systems, but training them has proved to be problematic because the backpropagated gradients either grow or shrink at each time step, so over many time steps they typically explode or vanish.[^lecun2015]

Concrètement, sur une longue séquence, un RNN classique a beaucoup de mal à apprendre des **dépendances à long terme** : le signal d'apprentissage devenant quasiment nul pour les mots situés loin en amont, le réseau n'apprend presque rien sur leur importance pour la prédiction finale.

## LSTM : une mémoire conçue pour durer

### Le principe de la cellule mémoire

Face à ce problème, l'idée du LSTM (*Long Short-Term Memory*) est d'augmenter le réseau d'une mémoire explicite, dont le comportement par défaut est de **préserver** l'information plutôt que de la laisser se dégrader :

> A special unit called the memory cell acts like an accumulator or a gated leaky neuron: it has a connection to itself at the next time step that has a weight of one, so it copies its own real-valued state and accumulates the external signal, but this self-connection is multiplicatively gated by another unit that learns to decide when to clear the content of the memory.[^lecun2015]

Le schéma ci-dessous représente ce mécanisme : la cellule mémoire dispose d'une connexion à elle-même, dont le **poids vaut exactement 1**, et une porte apprise décide quand laisser passer un nouveau signal ou effacer le contenu existant.

![Mécanisme de la cellule mémoire LSTM](/assets/img/posts/lstm-memory-cell.svg)
_La connexion à soi-même de poids 1 est le seul point d'équilibre stable : en dessous, le signal s'évanouit exponentiellement ; au-dessus, il explose._

**Pourquoi précisément un poids de 1, ni plus ni moins ?** Sur une séquence de 100 pas de temps, un poids de connexion de 0,5 donnerait $0{,}5^{100} \approx 8 \times 10^{-31}$ : un signal négligeable, c'est le retour du vanishing gradient. Un poids de 1,5 donnerait $1{,}5^{100} \approx 4 \times 10^{17}$ : une valeur qui explose numériquement, rendant l'entraînement instable. Un poids exactement égal à 1 est le seul point d'équilibre : $1^{100} = 1$, quel que soit le nombre de pas de temps traversés. C'est ce choix précis qui permet à l'information de circuler sur de longues séquences sans se dégrader ni exploser.

### L'impact pratique

> LSTM networks have subsequently proved to be more effective than conventional RNNs, especially when they have several layers for each time step, enabling an entire speech recognition system that goes all the way from acoustics to the sequence of characters in the transcription.[^lecun2015]

Le LSTM est devenu, pendant longtemps, le composant standard pour tout système traitant des séquences longues : traduction automatique, reconnaissance vocale.

## Traduction automatique : l'architecture encodeur-décodeur

### Le principe

Une application concrète illustre bien tous ces mécanismes combinés : traduire une phrase d'une langue à une autre, avec deux réseaux entraînés conjointement.

> After reading an English sentence one word at a time, an English 'encoder' network can be trained so that the final state vector of its hidden units is a good representation of the thought expressed by the sentence. This thought vector can then be used as the initial hidden state of […] a jointly trained French 'decoder' network, which outputs a probability distribution for the first word of the French translation.[^lecun2015]

L'**encodeur** lit la phrase source mot par mot, et ne garde que son état final, un vecteur compressé, appelé "vecteur de pensée" (*thought vector*), censé résumer le sens de toute la phrase. Le **décodeur** part de ce vecteur pour générer la traduction, un mot à la fois : chaque mot généré est réinjecté comme entrée pour prédire le suivant, jusqu'à un symbole de fin de phrase.

Un résultat de cette approche a une portée presque philosophique :

> This rather naive way of performing machine translation has quickly become competitive with the state-of-the-art, and this raises serious doubts about whether understanding a sentence requires anything like the internal symbolic expressions that are manipulated by using inference rules.[^lecun2015]

Ce système ne manipule aucune règle grammaticale explicite, aucune structure logique symbolique, juste un vecteur de nombres réels, obtenu par composition de transformations apprises. Et pourtant, il rivalise avec les meilleures approches de traduction de l'époque.

### La limite du vecteur unique, et le mécanisme d'attention

Compresser tout le sens d'une phrase dans un seul vecteur de **taille fixe** devient un goulot d'étranglement à mesure que la phrase source s'allonge : faire tenir un paragraphe de 200 mots dans le même espace qu'une phrase de 5 mots implique nécessairement une perte d'information. Le texte anticipe la solution à ce problème :

> We expect systems that use RNNs to understand sentences or whole documents will become much better when they learn strategies for selectively attending to one part at a time.[^lecun2015]

Le principe, connu aujourd'hui sous le nom de **mécanisme d'attention** : au lieu de ne garder que l'état final de l'encodeur, on conserve l'état caché à **chaque** mot de la phrase source. À chaque étape de la génération, le décodeur compare son état actuel à chacun de ces états sources, calcule un **score de pertinence** pour chacun, puis combine ces états sources selon leur pertinence : une moyenne pondérée, recalculée à chaque mot généré.

![Mécanisme d'attention dans un décodeur](/assets/img/posts/attention-mechanism.svg)
_Pour générer "noir", le décodeur assigne un poids d'attention élevé au mot source "black" et un poids faible aux autres, cette pondération est apprise, pas fixée à l'avance._

Ces poids de pertinence sont eux-mêmes appris par rétropropagation, exactement comme le reste du système : le réseau découvre, à partir des données, quelle partie de la phrase source est pertinente pour générer chaque mot de la traduction. Le texte illustre cette idée appliquée à la génération de légendes d'images :

> When the RNN is given the ability to focus its attention on a different location in the input image […] as it generates each word, we found that it exploits this to achieve better 'translation' of images into captions.[^lecun2015]

C'est ce principe (remplacer un résumé unique et figé par une pondération dynamique, recalculée à chaque étape) qui deviendra, deux ans après la publication de cet article, le cœur de l'architecture **Transformer**.

## Au-delà de la mémoire simple : les mémoires externes

Le texte va plus loin que le LSTM, vers des architectures dotées d'une mémoire externe encore plus structurée, capable d'un raisonnement plus complexe :

> Proposals include the Neural Turing Machine in which the network is augmented by a 'tape-like' memory that the RNN can choose to read from or write to, and memory networks, in which a regular network is augmented by a kind of associative memory.[^lecun2015]

Un exemple frappant illustre les capacités de ces Memory Networks :

> In one test example, the network is shown a 15-sentence version of the *The Lord of the Rings* and correctly answers questions such as "where is Frodo now?"[^lecun2015]

Pour répondre correctement, le réseau doit avoir suivi, à travers 15 phrases, les déplacements successifs du personnage, et récupérer l'information la plus récente et pertinente : un principe qui repose sur la même logique que l'attention : comparer une requête (la question posée) à un ensemble d'éléments mémorisés, puis pondérer leur pertinence. Les Memory Networks peuvent être vues comme une généralisation de ce principe, appliqué à une mémoire plus durable et plus grande qu'une simple phrase.

## Ce que l'article anticipait pour la suite

Dans sa conclusion, le texte prend du recul sur l'avenir du domaine, avec des observations qui se sont révélées largement fondées :

> Unsupervised learning had a catalytic effect in reviving interest in deep learning, but has since been overshadowed by the successes of purely supervised learning. […] We expect unsupervised learning to become far more important in the longer term. Human and animal learning is largely unsupervised: we discover the structure of the world by observing it, not by being told the name of every object.[^lecun2015]

Cette prédiction s'est concrétisée, mais pas immédiatement : c'est surtout avec l'essor des grands modèles de langage, entraînés par apprentissage auto-supervisé sur d'immenses corpus de texte non labellisé, que cette anticipation s'est pleinement réalisée. Un sujet pour la suite de ce programme.

Le texte conclut sur une note qui reste, aujourd'hui encore, un axe de recherche actif :

> Ultimately, major progress in artificial intelligence will come about through systems that combine representation learning with complex reasoning.[^lecun2015]

## Ce qu'il faut retenir

- Les **word vectors** remplacent le symbole atomique par une représentation distribuée, où la proximité de deux mots reflète leur substituabilité dans des contextes similaires, pas leur simple co-occurrence.
- Les **RNN** traitent une séquence élément par élément, en partageant les mêmes poids à chaque pas de temps, ce qui permet d'appliquer la rétropropagation en les "dépliant" dans le temps.
- Sur de longues séquences, les RNN classiques souffrent du même problème de vanishing/exploding gradient qu'un réseau très profond.
- Le **LSTM** résout ce problème avec une cellule mémoire dont la connexion à elle-même a un poids exactement égal à 1 : le seul point d'équilibre qui évite à la fois l'évanouissement et l'explosion du signal.
- Le mécanisme d'**attention**, né du besoin de dépasser le goulot d'étranglement d'un vecteur de taille fixe, pondère dynamiquement l'importance de chaque élément source, une idée qui deviendra le cœur des Transformers.

Ceci clôt la lecture de l'article fondateur de LeCun, Bengio et Hinton. La semaine prochaine, le programme aborde justement l'architecture Transformer, directement héritière du mécanisme d'attention qu'on vient de voir émerger.

---

*Le code accompagnant cet article, une implémentation d'un RNN et d'une cellule LSTM from scratch, appliqués à une tâche simple de prédiction de séquence, est disponible sur [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/rnn-lstm-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
