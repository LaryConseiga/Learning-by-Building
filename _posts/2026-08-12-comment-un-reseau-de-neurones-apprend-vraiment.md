---
title: "Comment un réseau de neurones apprend vraiment : des pixels bruts à la rétropropagation"
date: 2026-08-12 09:00:00 +0000
categories: [Deep Learning, Fondamentaux]
tags: [deep-learning, backpropagation, supervised-learning, neural-networks, mlp]
math: true
---

*🇬🇧 [English version]({{ '/posts/how-a-neural-network-really-learns/' | relative_url }})*

## Pourquoi ce premier article

Je démarre cette semaine un programme d'auto-formation structuré en deep learning, sur quatre semaines : perceptrons et rétropropagation, puis CNN/RNN, puis Transformers, puis LLMs et lois d'échelle. Chaque semaine s'accompagne d'un projet de code et d'un article : celui-ci est le premier.

Le fil rouge de cette semaine 1 est un texte fondateur : *Deep Learning*, publié par Yann LeCun, Yoshua Bengio et Geoffrey Hinton dans *Nature* en 2015[^lecun2015]. Ce n'est pas un article de recherche technique pointu, mais une synthèse écrite par trois des chercheurs qui ont le plus contribué à faire du deep learning ce qu'il est aujourd'hui, destinée à un public scientifique large. C'est donc un excellent point d'entrée.

Cet article couvre trois choses : **pourquoi** le deep learning a changé la donne par rapport au machine learning classique, **comment** un réseau de neurones apprend concrètement (l'apprentissage supervisé), et **par quel mécanisme mathématique** cet apprentissage se propage à travers toutes les couches d'un réseau (la rétropropagation, ou *backpropagation*). Le code associé (implémentation d'un perceptron multicouche from scratch) est disponible dans [ce dépôt GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/mlp-from-scratch){:target="_blank"}.

## Le problème que le deep learning est venu résoudre

Avant de comprendre ce que le deep learning apporte, il faut comprendre ce qui posait problème avant lui.

### Le machine learning classique avait besoin d'un traducteur humain

Pendant des décennies, construire un système de reconnaissance de motifs (pattern recognition) demandait un travail d'ingénierie manuel considérable. Un expert devait concevoir un **extracteur de caractéristiques** (feature extractor) : une fonction, écrite à la main, qui transformait les données brutes (par exemple les valeurs de pixels d'une image) en une représentation plus exploitable, à partir de laquelle un algorithme de classification simple pouvait ensuite faire son travail.

Comme le disent les auteurs :

> Conventional machine-learning techniques were limited in their ability to process natural data in their raw form. For decades, constructing a pattern-recognition or machine-learning system required careful engineering and considerable domain expertise to design a feature extractor that transformed the raw data […] into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input.[^lecun2015]

Concrètement, pour un système de reconnaissance de visages en 2005, il fallait qu'un ingénieur passe du temps à définir : quels contours chercher, quelles textures, quels ratios de proportions du visage étaient pertinents. Ce travail était long, coûteux, spécifique à chaque tâche, et ne se généralisait pas bien à un problème voisin.

### L'idée centrale : apprendre les représentations, pas les concevoir

Le deep learning renverse ce paradigme. Au lieu qu'un humain conçoive les caractéristiques pertinentes, le réseau les **découvre lui-même**, à partir des données, via une procédure d'apprentissage générale.

> The key aspect of deep learning is that these layers of features are not designed by human engineers: they are learned from data using a general-purpose learning procedure.[^lecun2015]

Ce qui rend ça possible, c'est l'idée de **hiérarchie de représentations**. Un réseau profond est composé de plusieurs couches empilées, et chaque couche transforme la représentation qu'elle reçoit de la couche précédente en une représentation légèrement plus abstraite. Pour une image, cette hiérarchie ressemble typiquement à ceci :

1. **Couche 1** : détecte la présence ou l'absence de contours (edges), à des orientations et positions particulières
2. **Couche 2** : détecte des motifs, en repérant des arrangements particuliers de contours de la couche 1
3. **Couche 3** : assemble ces motifs en combinaisons plus larges correspondant à des parties d'objets familiers
4. **Couches suivantes** : détectent des objets entiers, comme combinaisons de ces parties

Un point essentiel à bien saisir : **chaque couche ne « voit » jamais les données brutes**, sauf la toute première. La couche 3 ne regarde jamais les pixels de l'image directement ; elle ne reçoit que la sortie, déjà transformée, de la couche 2. C'est un processus strictement séquentiel, où l'abstraction s'accumule étage après étage, un peu comme une chaîne de montage où chaque poste de travail ne voit que le résultat du poste précédent, jamais la matière première d'origine.

Et surtout : **rien de tout cela n'est programmé à l'avance**. Personne ne dit explicitement à la couche 2 « cherche des boucles fermées » ou « cherche des intersections en croix ». C'est un phénomène **émergent** : c'est le processus d'optimisation global (qu'on va détailler plus bas) qui, en cherchant à minimiser une erreur de classification, pousse chaque couche à construire les représentations les plus utiles pour la tâche, et ces représentations se trouvent correspondre, après coup, à des concepts qu'un humain reconnaîtrait (contours, motifs, parties d'objets).

Une conséquence intéressante de cette hiérarchie : les toutes premières couches d'un réseau (les détecteurs de contours) sont presque **identiques** d'une tâche à l'autre, que le réseau soit entraîné à reconnaître des chats, des visages ou des panneaux de signalisation, car ce sont des structures statistiques universelles des images naturelles. Les couches plus profondes, en revanche, divergent fortement d'une tâche à l'autre, car elles encodent ce qui est spécifiquement discriminant pour ce problème précis. C'est d'ailleurs le principe sur lequel repose le *transfer learning*, qu'on croisera plus tard dans ce programme.

### Pourquoi cette approche a autant progressé

Une dernière remarque des auteurs, qui explique une bonne partie de la dynamique du domaine depuis 2012 :

> We think that deep learning will have many more successes in the near future because it requires very little engineering by hand, so it can easily take advantage of increases in the amount of available computation and data.[^lecun2015]

Contrairement à une approche à base de caractéristiques conçues à la main (qui plafonne assez vite, même si on lui donne davantage de données ou de calcul, parce qu'un humain ne peut pas concevoir 100 fois plus de caractéristiques pertinentes juste parce qu'on a 100 fois plus de données), le deep learning **absorbe directement** ces deux ressources. Plus de données et plus de calcul se traduisent presque mécaniquement par de meilleures représentations apprises. C'est cette propriété de scalabilité qui explique la course actuelle aux modèles toujours plus grands, entraînés sur toujours plus de données.

## L'apprentissage supervisé : comment un réseau apprend, concrètement

Passons maintenant du « pourquoi » au « comment ». La forme la plus courante d'apprentissage, profond ou non, est l'**apprentissage supervisé**.

### Le mécanisme en cinq étapes

Imaginons qu'on veuille construire un système capable de classifier des images en quatre catégories : maison, voiture, personne, animal domestique. On collecte d'abord un grand jeu de données d'images, chacune labellisée avec sa catégorie correcte.

**1. Le forward pass (passe avant).** On montre une image au réseau. Il produit en sortie un vecteur de scores, un par catégorie. On aimerait que la bonne catégorie ait le score le plus élevé, mais avant tout entraînement, avec des poids initialisés aléatoirement, ce n'est presque jamais le cas.

**2. La fonction objectif (loss).** On calcule une fonction qui mesure l'écart (l'erreur, ou la distance) entre les scores produits par le réseau et le motif de scores désiré (typiquement, un score de 1 pour la bonne catégorie, 0 pour les autres).

**3. Les poids.** Le réseau modifie ensuite ses paramètres internes ajustables pour réduire cette erreur. Ces paramètres, appelés poids (*weights*), sont des nombres réels : de véritables « boutons » qui définissent la fonction entrée-sortie du réseau. Dans un système de deep learning typique, il peut y en avoir des centaines de millions.

**4. Le gradient.** Pour ajuster correctement le vecteur de poids, l'algorithme d'apprentissage calcule un **gradient** : pour chaque poids, il indique de quelle quantité l'erreur augmenterait ou diminuerait si ce poids était augmenté d'une quantité infinitésimale.

**5. La mise à jour.** Le vecteur de poids est alors ajusté **dans la direction opposée** au gradient :

$$
w \leftarrow w - \eta \cdot \frac{\partial E}{\partial w}
$$

où $\frac{\partial E}{\partial w}$ est le gradient de l'erreur $E$ par rapport au poids $w$, et $\eta$ (eta) est le **taux d'apprentissage** (*learning rate*), un petit nombre positif qui contrôle la taille du pas de mise à jour. Si le gradient est positif (augmenter $w$ ferait augmenter l'erreur), on diminue $w$. S'il est négatif, on l'augmente. Le signe moins devant le taux d'apprentissage garantit qu'on va toujours dans le sens qui réduit l'erreur.

### L'image du paysage vallonné

Les auteurs proposent une analogie utile pour visualiser ce processus :

> [The objective function, averaged over all the training examples,] can be seen as a kind of hilly landscape in the high-dimensional space of weight values. The negative gradient vector indicates the direction of steepest descent in this landscape, taking it closer to a minimum, where the output error is low on average.[^lecun2015]

Imaginez un paysage vallonné, mais en plusieurs millions de dimensions (une dimension par poids, impossible à se représenter mentalement au-delà de trois, mais l'intuition en 2D ou 3D aide). On cherche le point le plus bas de ce paysage, c'est-à-dire l'erreur minimale. Le gradient donne la pente locale à l'endroit où l'on se trouve, et on progresse en descendant cette pente, un petit pas à la fois, d'où le nom de **descente de gradient**.

### En pratique : la descente de gradient stochastique

Calculer le gradient exact nécessiterait, en théorie, de faire la moyenne sur l'intégralité du jeu de données à chaque étape de mise à jour. Avec des centaines de millions d'exemples, ce serait beaucoup trop coûteux en temps et en calcul.

En pratique, on utilise donc la **descente de gradient stochastique** (*Stochastic Gradient Descent*, SGD) :

> This consists of showing the input vector for a few examples, computing the outputs and the errors, computing the average gradient for those examples, and adjusting the weights accordingly. The process is repeated for many small sets of examples from the training set until the average of the objective function stops decreasing.[^lecun2015]

On prend un petit sous-ensemble d'exemples (un *mini-batch*, par exemple 32 ou 128 images), on calcule le gradient uniquement sur ce sous-ensemble, on met à jour les poids, puis on recommence avec un nouveau sous-ensemble. C'est « stochastique » parce que chaque petit lot ne donne qu'une estimation bruitée du gradient exact, calculé sur tout le dataset, pas la valeur parfaite.

Un point qui a surpris beaucoup de praticiens à l'époque : cette méthode simple, malgré son caractère approximatif, trouve généralement un bon jeu de poids étonnamment vite, comparée à des techniques d'optimisation bien plus sophistiquées.

Une précision utile : l'aléatoire du SGD porte sur **l'ordre de présentation** des exemples, pas sur leur omission. En pratique, on mélange le jeu de données puis on le découpe en mini-batchs successifs qui couvrent l'intégralité des exemples : un passage complet s'appelle une *epoch*. Chaque exemple est donc vu exactement une fois par epoch, sans biais systématique.

### Pourquoi les classifieurs linéaires ne suffisent pas

Une bonne partie des applications pratiques de machine learning utilise des classifieurs linéaires appliqués à des caractéristiques conçues à la main. Un classifieur linéaire à deux classes calcule une somme pondérée des composantes du vecteur de caractéristiques ; si cette somme dépasse un seuil, l'entrée est classée dans une catégorie donnée.

Le problème, connu depuis longtemps :

> Since the 1960s we have known that linear classifiers can only carve their input space into very simple regions, namely half-spaces separated by a hyperplane.[^lecun2015]

Un classifieur linéaire ne peut tracer qu'une frontière droite (ou plate, en haute dimension) entre deux classes. Si les données ne sont pas séparables par une ligne droite, il échoue, quel que soit le réglage de ses poids.

L'exemple donné par les auteurs illustre bien pourquoi c'est un problème sérieux en vision par ordinateur : deux photos d'un même chien Samoyède, prises dans des poses différentes et des environnements différents, peuvent être très différentes pixel par pixel, alors qu'une photo de Samoyède et une photo de loup blanc, prises dans une pose et un environnement similaires, peuvent être très proches pixel par pixel. Un classifieur linéaire opérant directement sur les pixels bruts n'a aucun moyen fiable de faire la différence.

Ce problème porte un nom précis dans l'article : le **dilemme sélectivité-invariance**.

> This is why shallow classifiers require a good feature extractor that solves the selectivity–invariance dilemma — one that produces representations that are selective to the aspects of the image that are important for discrimination, but that are invariant to irrelevant aspects such as the pose of the animal.[^lecun2015]

- **Sélectivité** : rester sensible aux détails qui comptent vraiment pour la distinction (la forme du museau, la structure du crâne)
- **Invariance** : rester insensible aux détails qui ne comptent pas (la pose, l'éclairage, l'arrière-plan)

C'est exactement ce que la hiérarchie de représentations, décrite plus haut, permet de résoudre : chaque couche devient progressivement plus invariante aux détails non pertinents, tout en restant sélective sur ce qui compte réellement pour la tâche.

## La rétropropagation : comment l'erreur remonte jusqu'aux premières couches

On sait maintenant comment un réseau ajuste ses poids en théorie. Reste une question cruciale : dans un réseau à plusieurs couches, un poids situé tout au fond (près de l'entrée) influence la sortie de façon **indirecte** : il influence la couche suivante, qui influence la couche d'après, et ainsi de suite jusqu'à la sortie, où l'erreur est calculée. Comment calculer l'effet d'un poids aussi éloigné sur une erreur mesurée si loin en aval ?

C'est exactement ce que résout l'algorithme de **rétropropagation** (*backpropagation*).

### Le forward pass, formellement

Avant de voir le mécanisme de retour, formalisons le trajet aller. Pour une unité $j$ d'une couche cachée, recevant les sorties $x_i$ de la couche précédente :

$$
z_j = \sum_{i} w_{ij}\, x_i \qquad\qquad y_j = f(z_j)
$$

$z_j$ est la somme pondérée des entrées, et $y_j$ est la sortie de l'unité, obtenue en appliquant une fonction non-linéaire $f$ à $z_j$. Sans cette non-linéarité, empiler des couches ne servirait à rien : une somme pondérée de sommes pondérées reste une somme pondérée, incapable de représenter des fonctions complexes. C'est elle qui donne au réseau sa capacité à apprendre des frontières de décision non linéaires, exactement ce qui manquait au classifieur linéaire vu plus haut.

Parmi les fonctions non-linéaires utilisées, la plus employée aujourd'hui est le **ReLU** (*Rectified Linear Unit*) : $f(z) = \max(0, z)$. On reviendra sur pourquoi elle a supplanté les sigmoïdes plus classiques.

### Le principe clé de la rétropropagation

L'intuition fondamentale, énoncée ainsi par les auteurs :

> The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module.[^lecun2015]

Autrement dit : pour connaître l'effet d'une couche sur l'erreur finale, il suffit de connaître l'effet de la couche **suivante**, pas besoin de tout recalculer depuis zéro à chaque fois. Ce principe s'appuie sur la **règle de la chaîne** des dérivées : si $x$ influence $y$, et $y$ influence $z$, alors l'effet de $x$ sur $z$ est le produit des deux effets intermédiaires :

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

### Le mécanisme, couche par couche

En partant de la sortie et en remontant vers l'entrée :

**Étape 1 : l'erreur à la sortie**, directement calculable puisqu'on connaît à la fois la prédiction $y_l$ et la cible $t_l$ :

$$
\frac{\partial E}{\partial y_l} = y_l - t_l
$$

**Étape 2 : conversion via la non-linéarité**, en appliquant la règle de la chaîne :

$$
\frac{\partial E}{\partial z_l} = \frac{\partial E}{\partial y_l} \cdot \frac{\partial y_l}{\partial z_l}
$$

**Étape 3 : propagation vers la couche du dessous.** C'est l'étape la plus importante à bien comprendre :

$$
\frac{\partial E}{\partial y_j} = \sum_{l} w_{jl}\, \frac{\partial E}{\partial z_l}
$$

Concrètement, cette formule dit : l'erreur « héritée » par une unité $j$ est une **somme pondérée** des erreurs de toutes les unités de la couche suivante auxquelles $j$ est connectée, pondérée par les **mêmes poids** $w_{jl}$ que ceux utilisés dans le forward pass, mais employés maintenant en sens inverse.

Pourquoi précisément ce poids-là ? Parce que $w_{jl}$ mesure, dans le forward pass, l'ampleur de l'influence de $j$ sur $l$. Si $w_{jl}$ est grand, un petit changement de $y_j$ a un gros effet sur $z_l$, donc sur l'erreur finale : $j$ hérite alors d'une part importante de responsabilité. Si $w_{jl}$ est proche de zéro, $j$ n'a presque aucune influence sur la sortie via cette connexion, et hérite de presque aucune erreur par ce chemin. Si $j$ est connectée à plusieurs unités en aval, elle accumule sa part de responsabilité venant de **chacune** d'elles, proportionnellement à l'importance de chaque connexion.

On répète ensuite les étapes 2 et 3 couche après couche, en descendant : l'erreur se propage à rebours de la sortie vers l'entrée. C'est de là que vient le nom « rétro-propagation ».

**Étape 4 : le gradient de chaque poids**, une fois qu'on dispose de $\partial E/\partial z_k$ pour une couche :

$$
\frac{\partial E}{\partial w_{jk}} = y_j \cdot \frac{\partial E}{\partial z_k}
$$

Ce gradient est ensuite injecté directement dans la formule de mise à jour vue plus haut.

### Pourquoi ce détour économise énormément de calcul

Sans ce mécanisme, calculer le gradient de chaque poids indépendamment demanderait, pour chaque poids d'une couche profonde, de retraverser tout le chemin d'influence jusqu'à la sortie, un coût qui exploserait avec le nombre de couches et de poids. Grâce à la rétropropagation, on calcule l'erreur d'une couche **une seule fois**, en la déduisant algébriquement de la couche juste au-dessus, et tous les gradients de poids de cette couche s'en déduisent ensuite à faible coût.

Le résultat : le coût total de la rétropropagation reste du même ordre de grandeur qu'un simple forward pass. C'est précisément ce qui rend possible, en pratique, l'entraînement de réseaux comptant des centaines de millions de poids.

### Le mythe des minima locaux

Dans les années 1990, la communauté du machine learning pensait qu'une descente de gradient simple resterait piégée dans de mauvais **minima locaux**, des configurations de poids où aucun petit changement ne réduirait l'erreur, mais qui seraient loin d'être optimales.

Reprenez l'image du paysage vallonné : un minimum local est une petite vallée où le terrain remonte dans toutes les directions autour de vous, alors qu'il existe peut-être une vallée bien plus profonde ailleurs ; mais pour l'atteindre, il faudrait d'abord remonter, ce qu'une descente de gradient ne fait jamais.

L'expérience a montré que cette crainte était largement infondée :

> In practice, poor local minima are rarely a problem with large networks. Regardless of the initial conditions, the system nearly always reaches solutions of very similar quality.[^lecun2015]

Le véritable phénomène en jeu est différent : le paysage d'erreur, en très haute dimension, contient un nombre combinatoirement grand de **points-selles**, des points où le gradient est nul, mais où le terrain monte dans certaines directions et descend dans d'autres, comme le col d'une selle de cheval. Ces points ne bloquent pas durablement l'algorithme, car il existe presque toujours au moins une direction de descente disponible, et empiriquement, la quasi-totalité de ces points-selles présentent des valeurs d'erreur assez similaires, peu importe donc lequel on croise en chemin.

### ReLU contre le problème du gradient qui s'évanouit

Un dernier point technique important, qui explique pourquoi ReLU s'est imposé :

> The ReLU typically learns much faster in networks with many layers, allowing training of a deep supervised network without unsupervised pre-training.[^lecun2015]

Les fonctions sigmoïdes classiques (comme la tangente hyperbolique) ont une dérivée quasiment nulle loin de zéro. Or, dans la formule de rétropropagation vue plus haut, on **multiplie** l'erreur par cette dérivée à chaque couche traversée. Si cette dérivée est proche de zéro à chaque étage, et qu'on multiplie ce facteur dix ou vingt fois de suite (une fois par couche), le gradient devient exponentiellement petit en remontant vers les premières couches : c'est le problème du **gradient qui s'évanouit** (*vanishing gradient*). ReLU, dont la dérivée vaut soit 0 soit exactement 1, ne s'écrase jamais progressivement de cette manière, ce qui permet d'entraîner des réseaux profonds directement, sans étape de pré-entraînement préalable.

## Ce qu'il faut retenir

- Le deep learning remplace l'ingénierie manuelle de caractéristiques par une **hiérarchie de représentations apprises automatiquement**, chaque couche construisant ses propres features à partir de la sortie de la couche précédente.
- L'**apprentissage supervisé** ajuste les poids d'un réseau en minimisant une fonction d'erreur, via une descente de gradient : en pratique, une version stochastique (SGD) utilisant de petits mini-batchs plutôt que le dataset complet.
- Les classifieurs linéaires classiques échouent sur des problèmes comme la vision, car ils ne peuvent pas résoudre le **dilemme sélectivité-invariance**.
- La **rétropropagation** permet de calculer efficacement le gradient de chaque poids, à travers toutes les couches, en réutilisant les poids du forward pass pour propager l'erreur en sens inverse : un calcul explicite, couche par couche, jamais un phénomène automatique.
- Les craintes historiques autour des minima locaux se sont révélées largement infondées ; le vrai défi (les points-selles) ne bloque pas durablement l'apprentissage en haute dimension.

La semaine prochaine, on passera des perceptrons multicouches génériques à une architecture spécialisée pour le traitement des images : les **réseaux de neurones convolutifs** (CNN), qui exploitent directement la structure spatiale des données pour rendre l'entraînement praticable sur de grandes images.

---

*Le code accompagnant cet article, une implémentation d'un perceptron multicouche et de la rétropropagation, entièrement from scratch (sans framework de deep learning), est disponible sur [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/mlp-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
