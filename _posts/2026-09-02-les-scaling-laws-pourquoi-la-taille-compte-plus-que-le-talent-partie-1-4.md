---
title: "Les Scaling Laws : pourquoi la taille compte plus que le talent (Partie 1/4)"
date: 2026-09-02 10:00:00 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, semaine-4]
math: true
---

*🇬🇧 [English version]({{ '/posts/scaling-laws-why-size-matters-more-than-skill-part-1-4/' | relative_url }})*

## Le papier qui a (discrètement) tout changé

Si tu devais pointer du doigt UN papier qui explique pourquoi on est passé de "petits modèles bien réglés" à "des monstres à des milliards de paramètres", ce serait probablement celui-là : **Scaling Laws for Neural Language Models**, publié par une équipe d'OpenAI (Kaplan et al.) en janvier 2020.

Pas de nouvelle architecture révolutionnaire dedans. Pas de trick d'entraînement magique. Juste... des courbes. Beaucoup de courbes. Et une conclusion qui a fini par justifier la course au gigantisme qu'on connaît aujourd'hui avec les LLM.

Dans cette série en 4 parties, on va décortiquer ce papier ensemble, sans jargon inutile. Aujourd'hui : les trois lois de base, et pourquoi elles sont plus surprenantes qu'il n'y paraît.

![Trois lois de puissance : loss en fonction du compute, de la taille du dataset et du nombre de paramètres](/assets/img/posts/scaling-laws-three-power-laws.png)
_Figure 1, page 3. Le graphique à retenir de tout le papier. Trois courbes, trois lois de puissance, un seul message._

## Le setup : trois ingrédients, une recette

Pour entraîner un modèle de langage, tu as trois leviers principaux :

- **N** : la taille du modèle (le nombre de paramètres)
- **D** : la quantité de données sur laquelle tu l'entraînes
- **C** : la quantité de calcul (compute) que tu mets dans l'entraînement

Question naturelle : si j'augmente l'un de ces trois trucs, qu'est-ce qui se passe sur la performance du modèle ?

Réponse du papier, en une phrase : **ça suit une loi de puissance, de façon incroyablement régulière**, et ce sur presque **sept ordres de grandeur** (on parle de modèles allant de 768 paramètres à 1,5 milliard).

Concrètement :

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C_{min}) = \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_C^{min}}, \quad \alpha_C^{min} \approx 0.050$$

Ne panique pas devant les formules. Le message derrière est simple : **plus tu montes N, D ou C, plus la loss (l'erreur du modèle) baisse, et elle baisse de façon parfaitement prévisible**, comme une courbe bien élevée qui ne fait jamais de caprice.

## L'analogie qui aide : la recette de gâteau qui ne rate jamais

Imagine une recette de gâteau où, peu importe la quantité de pâte que tu prépares, le résultat est **toujours proportionnellement aussi bon**. Tu peux prédire à l'avance, avec une règle mathématique simple, exactement à quel point ton gâteau sera réussi selon la quantité de farine que tu utilises.

C'est un peu ce que dit ce papier sur les réseaux de neurones : la performance n'est pas un mystère artisanal où il faut "avoir la main". C'est prévisible. **On peut tracer la courbe avant même d'avoir fini de cuire.**

## Le vrai scoop : la forme du modèle n'a presque aucune importance

Voici le résultat qui casse le plus d'intuitions reçues : à nombre de paramètres égal, **la façon dont tu organises ton modèle** (profond et fin ? large et court ?) **change à peine la performance**.

![Variation de la loss selon le ratio profondeur/largeur, le ratio feed-forward et la dimension des têtes d'attention](/assets/img/posts/scaling-laws-depth-width-ratio.png)
_Figure 5, page 8. On voit ici que le ratio profondeur/largeur peut varier par un facteur de 40 en ne changeant la loss que de quelques pourcents à peine._

Autrement dit : un architecte qui passe des semaines à peaufiner le nombre de couches, la largeur exacte, le nombre de têtes d'attention... perd probablement son temps, comparé à quelqu'un qui se contente d'augmenter la taille totale du modèle. **Ce qui compte, c'est l'échelle. Pas le design fin.**

C'est un peu comme découvrir que peu importe la forme de ton réservoir d'eau (carré, rond, tout en hauteur, tout en largeur), ce qui détermine combien d'eau il contient, c'est son volume total. La forme est presque un détail cosmétique.

## Transformer vs LSTM : où l'architecture *compte* vraiment

Attention, nuance importante : dire que "la forme *interne*" d'un Transformer importe peu ne veut pas dire que le *type* d'architecture n'a aucune importance.

![Comparaison de la loss entre LSTM et Transformer selon la taille du modèle et la position dans le contexte](/assets/img/posts/scaling-laws-lstm-vs-transformer.png)
_Figure 7, page 9. Comparaison directe entre LSTM et Transformer._

Le papier montre que les LSTM (l'ancêtre des Transformers pour le texte) suivent très bien les Transformers... sur les premiers tokens d'un texte. Mais dès que le contexte s'allonge, les LSTM décrochent, alors que les Transformers continuent de s'améliorer.

**Pourquoi ?** Un LSTM doit faire voyager l'information à travers une longue chaîne d'étapes, un peu comme un jeu de téléphone arabe où le message se dilue au fil des relais. Un Transformer, lui, peut "regarder" directement n'importe quel mot du texte, peu importe sa distance, sans relais et sans dilution.

C'est ce qui explique, entre autres, pourquoi l'architecture Transformer est devenue le standard incontesté pour le NLP moderne.

## Bonus : la généralisation vient "gratuitement"

Dernier point sympa de cette section : les auteurs ont testé leurs modèles (entraînés uniquement sur un dataset appelé WebText2) sur d'autres types de textes, comme Wikipedia, des livres ou du Common Crawl.

![Généralisation de la loss sur d'autres distributions de texte (Wikipedia, livres, Common Crawl)](/assets/img/posts/scaling-laws-generalization.png)
_Figure 8, page 10. La performance sur d'autres distributions de texte suit la performance sur les données d'entraînement, avec un écart constant._

Résultat : le modèle s'améliore sur ces autres textes **au même rythme** qu'il s'améliore sur ses propres données d'entraînement, avec juste un petit décalage fixe. Autrement dit, améliorer un modèle sur ses données d'entraînement, c'est aussi, sans effort supplémentaire, l'améliorer partout ailleurs.

## Ce qu'il faut retenir de cette partie

- Trois leviers (N, D, C), trois lois de puissance, prévisibles sur des ordres de grandeur énormes
- La forme précise du modèle compte à peine, c'est l'échelle qui domine
- Le *type* d'architecture (attention vs récurrence) compte, lui, beaucoup, surtout sur les contextes longs
- Un modèle qui s'améliore sur ses données d'entraînement généralise "gratuitement" ailleurs

Dans la [**Partie 2**]({{ '/posts/les-scaling-laws-le-juste-equilibre-entre-taille-et-donnees-partie-2-4/' | relative_url }}), on va voir ce qui se passe quand on fait varier N et D *en même temps*, et pourquoi la relation entre les deux n'est pas celle à laquelle on s'attendrait naturellement (spoiler : doubler ton modèle ne demande pas de doubler tes données).
