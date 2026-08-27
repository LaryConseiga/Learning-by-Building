---
title: "Attention Is All You Need : quand l'attention remplace la récurrence"
date: 2026-08-26 10:00:00 +0000
categories: [Deep Learning, NLP]
tags: [transformer, attention, nlp, deep-learning, semaine-3]
render_with_liquid: false
---

*🇬🇧 [English version](https://laryconseiga.github.io/Learning-by-Building/posts/attention-is-all-you-need-when-attention-replaces-recurrence/)*

Semaine 3 de mon plan de lecture. Après les MLP, les CNN et les LSTM, on attaque le papier qui a redéfini le NLP moderne : **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017). C'est le papier qui introduit le **Transformer**, l'architecture derrière à peu près tous les grands modèles de langage actuels.

Le pitch du papier est presque provocateur : on peut jeter les réseaux récurrents (RNN, LSTM) à la poubelle et ne garder « que » de l'attention. Résultat : un nouveau state-of-the-art en traduction, obtenu en une fraction du temps d'entraînement des modèles précédents (3.5 jours sur 8 GPU contre plusieurs semaines pour la concurrence).

Dans cet article, je reconstruis le raisonnement du papier étape par étape : pourquoi les RNN posent problème, comment fonctionne l'attention concrètement, et comment tout ça s'assemble dans l'architecture complète.

## Le problème que règle le Transformer

Un RNN traite une séquence mot par mot. Pour calculer son état caché à la position *t*, il a **besoin** de l'état à la position *t-1*, qui a besoin de *t-2*, etc. Cette chaîne de dépendances est stricte : impossible de calculer l'état 50 sans être passé par les 49 précédents.

Le problème, c'est que ça **interdit toute parallélisation à l'intérieur d'une même séquence**. Même avec 8 GPU, on ne peut pas paralleliser le calcul des états d'une seule phrase, seulement paralleliser entre plusieurs phrases différentes. Sur des séquences longues, c'est un vrai goulot d'étranglement.

Les CNN règlent déjà une partie du problème : à l'intérieur d'une couche, chaque position se calcule indépendamment des autres, à partir d'une fenêtre locale déjà disponible. Mais pour relier deux positions éloignées dans la séquence, il faut empiler plusieurs couches : le « champ réceptif » grandit progressivement, couche après couche.

Le Transformer pousse cette idée à son terme logique : et si chaque position pouvait directement consulter **toutes les autres positions**, en une seule opération, sans fenêtre limitée ni empilement nécessaire pour élargir la portée ? C'est exactement ce que fait l'attention.

## La logique de l'attention, en une image

Avant les formules, l'intuition. Le mécanisme d'attention repose sur trois objets : la **query** (ce que je cherche), la **key** (une étiquette qui dit « voilà ce que je contiens ») et la **value** (le contenu réel qu'on récupère si le match est bon).

C'est un peu le fonctionnement d'un moteur de recherche : votre requête (query) est comparée à des étiquettes de documents (keys), et vous récupérez le contenu (values) des documents dont l'étiquette matche le mieux, sauf qu'ici, au lieu de ne récupérer que le meilleur match, on récupère une **moyenne pondérée de tout**, proportionnellement à la qualité du match.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="523.035pt" height="294.638281pt" viewBox="0 0 523.035 294.638281" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-26T22:54:23.568178</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.10.8, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 294.638281 
L 523.035 294.638281 
L 523.035 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 22.45905 93.500148 
L 124.18605 93.500148 
Q 128.25513 93.500148 128.25513 90.067615 
L 128.25513 47.160948 
Q 128.25513 43.728415 124.18605 43.728415 
L 22.45905 43.728415 
Q 18.38997 43.728415 18.38997 47.160948 
L 18.38997 90.067615 
Q 18.38997 93.500148 22.45905 93.500148 
z
" clip-path="url(#p3bfed92241)" style="fill: #f6ddb8; stroke: #e08a2c; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_3">
    <path d="M 22.45905 161.292681 
L 103.84065 161.292681 
Q 106.89246 161.292681 106.89246 158.718281 
L 106.89246 124.392948 
Q 106.89246 121.818548 103.84065 121.818548 
L 22.45905 121.818548 
Q 19.40724 121.818548 19.40724 124.392948 
L 19.40724 158.718281 
Q 19.40724 161.292681 22.45905 161.292681 
z
" clip-path="url(#p3bfed92241)" style="fill: #cfe0f0; stroke: #3b6ea5; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_4">
    <path d="M 22.45905 212.780681 
L 103.84065 212.780681 
Q 106.89246 212.780681 106.89246 210.206281 
L 106.89246 175.880948 
Q 106.89246 173.306548 103.84065 173.306548 
L 22.45905 173.306548 
Q 19.40724 173.306548 19.40724 175.880948 
L 19.40724 210.206281 
Q 19.40724 212.780681 22.45905 212.780681 
z
" clip-path="url(#p3bfed92241)" style="fill: #cfe0f0; stroke: #3b6ea5; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_5">
    <path d="M 22.45905 264.268681 
L 103.84065 264.268681 
Q 106.89246 264.268681 106.89246 261.694281 
L 106.89246 227.368948 
Q 106.89246 224.794548 103.84065 224.794548 
L 22.45905 224.794548 
Q 19.40724 224.794548 19.40724 227.368948 
L 19.40724 261.694281 
Q 19.40724 264.268681 22.45905 264.268681 
z
" clip-path="url(#p3bfed92241)" style="fill: #cfe0f0; stroke: #3b6ea5; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_6">
    <path d="M 180.1359 123.534815 
L 281.8629 123.534815 
Q 285.93198 123.534815 285.93198 120.102281 
L 285.93198 60.032948 
Q 285.93198 56.600415 281.8629 56.600415 
L 180.1359 56.600415 
Q 176.06682 56.600415 176.06682 60.032948 
L 176.06682 120.102281 
Q 176.06682 123.534815 180.1359 123.534815 
z
" clip-path="url(#p3bfed92241)" style="fill: #fff3d6; stroke: #c99a2e; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_7">
    <path d="M 317.46735 123.534815 
L 403.9353 123.534815 
Q 408.00438 123.534815 408.00438 120.102281 
L 408.00438 60.032948 
Q 408.00438 56.600415 403.9353 56.600415 
L 317.46735 56.600415 
Q 313.39827 56.600415 313.39827 60.032948 
L 313.39827 120.102281 
Q 313.39827 123.534815 317.46735 123.534815 
z
" clip-path="url(#p3bfed92241)" style="fill: #e4f0e8; stroke: #4c9a6f; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_8">
    <path d="M 312.381 161.292681 
L 393.7626 161.292681 
Q 396.81441 161.292681 396.81441 158.718281 
L 396.81441 124.392948 
Q 396.81441 121.818548 393.7626 121.818548 
L 312.381 121.818548 
Q 309.32919 121.818548 309.32919 124.392948 
L 309.32919 158.718281 
Q 309.32919 161.292681 312.381 161.292681 
z
" clip-path="url(#p3bfed92241)" style="fill: #e9dff2; stroke: #7b5ea3; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_9">
    <path d="M 312.381 212.780681 
L 393.7626 212.780681 
Q 396.81441 212.780681 396.81441 210.206281 
L 396.81441 175.880948 
Q 396.81441 173.306548 393.7626 173.306548 
L 312.381 173.306548 
Q 309.32919 173.306548 309.32919 175.880948 
L 309.32919 210.206281 
Q 309.32919 212.780681 312.381 212.780681 
z
" clip-path="url(#p3bfed92241)" style="fill: #e9dff2; stroke: #7b5ea3; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_10">
    <path d="M 312.381 264.268681 
L 393.7626 264.268681 
Q 396.81441 264.268681 396.81441 261.694281 
L 396.81441 227.368948 
Q 396.81441 224.794548 393.7626 224.794548 
L 312.381 224.794548 
Q 309.32919 224.794548 309.32919 227.368948 
L 309.32919 261.694281 
Q 309.32919 264.268681 312.381 264.268681 
z
" clip-path="url(#p3bfed92241)" style="fill: #e9dff2; stroke: #7b5ea3; stroke-width: 1.3; stroke-linejoin: miter"/>
   </g>
   <g id="patch_11">
    <path d="M 434.4534 187.894815 
L 505.6623 187.894815 
Q 509.73138 187.894815 509.73138 184.462281 
L 509.73138 124.392948 
Q 509.73138 120.960415 505.6623 120.960415 
L 434.4534 120.960415 
Q 430.38432 120.960415 430.38432 124.392948 
L 430.38432 184.462281 
Q 430.38432 187.894815 434.4534 187.894815 
z
" clip-path="url(#p3bfed92241)" style="fill: #f9d8d6; stroke: #c0504d; stroke-width: 1.6; stroke-linejoin: miter"/>
   </g>
   <g id="text_1">
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'" transform="translate(56.395988 65.657563)">Query</text>
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'" transform="translate(9.3038 76.9335)">(« ce que je cherche »)</text>
   </g>
   <g id="text_2">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="63.14985" y="144.31499" transform="rotate(-0 63.14985 144.31499)">Key₁</text>
   </g>
   <g id="text_3">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="63.14985" y="195.80299" transform="rotate(-0 63.14985 195.80299)">Key₂</text>
   </g>
   <g id="text_4">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="63.14985" y="247.29099" transform="rotate(-0 63.14985 247.29099)">Key₃</text>
   </g>
   <g id="patch_12">
    <path d="M 126.188428 70.759615 
Q 149.618445 70.759615 171.706822 70.759615 
" style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
    <path d="M 167.706822 68.759615 
L 171.706822 70.759615 
L 167.706822 72.759615 
" style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
   </g>
   <g id="patch_13">
    <path d="M 105.281431 140.166595 
Q 139.444772 107.230598 172.803217 75.070579 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 168.535422 76.406965 
L 172.803217 75.070579 
L 171.311649 79.286647 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="patch_14">
    <path d="M 104.859845 191.324101 
Q 139.443826 132.976431 173.457738 75.590542 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 169.697712 78.011742 
L 173.457738 75.590542 
L 173.138682 80.051282 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="patch_15">
    <path d="M 104.606676 242.68535 
Q 139.444921 158.718713 173.854702 75.784752 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 170.474479 78.712907 
L 173.854702 75.784752 
L 174.169093 80.245824 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="text_5">
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="230.9994" y="80.02999" transform="rotate(-0 230.9994 80.02999)">QKᵀ/√dₖ</text>
   </g>
   <g id="text_6">
    <text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="230.9994" y="98.849083" transform="rotate(-0 230.9994 98.849083)">scores bruts</text>
   </g>
   <g id="patch_16">
    <path d="M 283.863415 90.067615 
Q 297.121377 90.067615 309.037698 90.067615 
" style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
    <path d="M 305.037698 88.067615 
L 309.037698 90.067615 
L 305.037698 92.067615 
" style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
   </g>
   <g id="text_7">
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="360.701325" y="79.95499" transform="rotate(-0 360.701325 79.95499)">softmax</text>
   </g>
   <g id="text_8">
    <text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(349.309997 94.090013)">poids</text>
    <text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(348.434098 103.608154)">(Σ=1)</text>
   </g>
   <g id="text_9">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="353.0718" y="144.31499" transform="rotate(-0 353.0718 144.31499)">Value₁</text>
   </g>
   <g id="text_10">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="353.0718" y="195.80299" transform="rotate(-0 353.0718 195.80299)">Value₂</text>
   </g>
   <g id="text_11">
    <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="353.0718" y="247.29099" transform="rotate(-0 353.0718 247.29099)">Value₃</text>
   </g>
   <g id="text_12">
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="470.05785" y="157.18699" transform="rotate(-0 470.05785 157.18699)">Sortie</text>
   </g>
   <g id="text_13">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'" transform="translate(449.793475 169.126656)">(moyenne</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'" transform="translate(449.40785 178.468906)">pondérée)</text>
   </g>
   <g id="patch_17">
    <path d="M 395.667498 142.158204 
Q 414.106425 147.991117 431.479383 153.486823 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 428.268864 150.373534 
L 431.479383 153.486823 
L 427.062441 154.187265 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="patch_18">
    <path d="M 395.215488 191.664808 
Q 414.109989 173.733727 432.193516 156.572269 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 427.915341 157.875042 
L 432.193516 156.572269 
L 430.668829 160.776473 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="patch_19">
    <path d="M 394.585903 242.708526 
Q 414.107812 199.48003 433.169566 157.270483 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
    <path d="M 429.700513 160.092834 
L 433.169566 157.270483 
L 433.346015 161.739136 
" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
   </g>
   <g id="text_14">
    <text style="font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(367.484575 220.556031)">poids appliqués</text>
    <text style="font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(377.21395 229.514281)">aux values</text>
   </g>
   <g id="text_15">
    <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="261.5175" y="15.998281" transform="rotate(-0 261.5175 15.998281)">La logique générale de l'attention : Query + Keys → poids → moyenne pondérée des Values</text>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p3bfed92241">
   <rect x="7.2" y="29.998281" width="508.635" height="257.44"/>
  </clipPath>
 </defs>
</svg>
</div>

## Scaled Dot-Product Attention : la formule

Concrètement, voici la formule centrale du papier :

**Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V**

Décomposée en étapes :

1. **QKᵀ** : produit scalaire entre chaque query et chaque key. Pour une séquence de *n* tokens, ça donne une matrice *n×n* : un score de compatibilité pour chaque paire de positions.
2. **/ √dₖ** : mise à l'échelle. Sans ça, quand la dimension dₖ est grande, les produits scalaires explosent en magnitude et poussent le softmax dans une zone saturée (gradient quasi nul). Diviser par √dₖ compense exactement cette croissance.
3. **softmax** : transforme les scores bruts en poids qui somment à 1 sur chaque ligne.
4. **× V** : combine les values selon ces poids : c'est la sortie finale, une moyenne pondérée.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="554.4pt" height="251.838203pt" viewBox="0 0 554.4 251.838203" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-26T22:54:24.248561</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.10.8, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 251.838203 
L 554.4 251.838203 
L 554.4 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 20.057143 174.392203 
L 95.057143 174.392203 
Q 98.057143 174.392203 98.057143 169.618203 
L 98.057143 74.138203 
Q 98.057143 69.364203 95.057143 69.364203 
L 20.057143 69.364203 
Q 17.057143 69.364203 17.057143 74.138203 
L 17.057143 169.618203 
Q 17.057143 174.392203 20.057143 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #f6ddb8; stroke: #e08a2c; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_3">
    <path d="M 105.771429 174.392203 
L 180.771429 174.392203 
Q 183.771429 174.392203 183.771429 169.618203 
L 183.771429 74.138203 
Q 183.771429 69.364203 180.771429 69.364203 
L 105.771429 69.364203 
Q 102.771429 69.364203 102.771429 74.138203 
L 102.771429 169.618203 
Q 102.771429 174.392203 105.771429 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #fff3d6; stroke: #c99a2e; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_4">
    <path d="M 191.485714 174.392203 
L 266.485714 174.392203 
Q 269.485714 174.392203 269.485714 169.618203 
L 269.485714 74.138203 
Q 269.485714 69.364203 266.485714 69.364203 
L 191.485714 69.364203 
Q 188.485714 69.364203 188.485714 74.138203 
L 188.485714 169.618203 
Q 188.485714 174.392203 191.485714 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #fde3e1; stroke: #c0504d; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_5">
    <path d="M 277.2 174.392203 
L 352.2 174.392203 
Q 355.2 174.392203 355.2 169.618203 
L 355.2 74.138203 
Q 355.2 69.364203 352.2 69.364203 
L 277.2 69.364203 
Q 274.2 69.364203 274.2 74.138203 
L 274.2 169.618203 
Q 274.2 174.392203 277.2 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #eeeeee; stroke: #888888; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_6">
    <path d="M 362.914286 174.392203 
L 437.914286 174.392203 
Q 440.914286 174.392203 440.914286 169.618203 
L 440.914286 74.138203 
Q 440.914286 69.364203 437.914286 69.364203 
L 362.914286 69.364203 
Q 359.914286 69.364203 359.914286 74.138203 
L 359.914286 169.618203 
Q 359.914286 174.392203 362.914286 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #e4f0e8; stroke: #4c9a6f; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="patch_7">
    <path d="M 448.628571 174.392203 
L 523.628571 174.392203 
Q 526.628571 174.392203 526.628571 169.618203 
L 526.628571 74.138203 
Q 526.628571 69.364203 523.628571 69.364203 
L 448.628571 69.364203 
Q 445.628571 69.364203 445.628571 74.138203 
L 445.628571 169.618203 
Q 445.628571 174.392203 448.628571 174.392203 
z
" clip-path="url(#p996308cd41)" style="fill: #e9dff2; stroke: #7b5ea3; stroke-width: 1.5; stroke-linejoin: miter"/>
   </g>
   <g id="text_1">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="57.557143" y="114.545547" transform="rotate(-0 57.557143 114.545547)">Q, K, V</text>
   </g>
   <g id="text_2">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(40.141518 146.886578)">matrices</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(40.787143 155.844828)">d'entrée</text>
   </g>
   <g id="patch_8">
    <path d="M 97.913797 121.878203 
Q 100.414216 121.878203 101.349387 121.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
    <path d="M 97.349387 119.878203 
L 101.349387 121.878203 
L 97.349387 123.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
   </g>
   <g id="text_3">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="143.271429" y="114.545547" transform="rotate(-0 143.271429 114.545547)">QKᵀ</text>
   </g>
   <g id="text_4">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(128.886429 146.886578)">produit</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(127.757679 155.844828)">scalaire</text>
   </g>
   <g id="patch_9">
    <path d="M 183.628083 121.878203 
Q 186.128502 121.878203 187.063673 121.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
    <path d="M 183.063673 119.878203 
L 187.063673 121.878203 
L 183.063673 123.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
   </g>
   <g id="text_5">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="228.985714" y="114.545547" transform="rotate(-0 228.985714 114.545547)">/√dₖ</text>
   </g>
   <g id="text_6">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(215.710714 146.886578)">mise à</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(212.434464 155.844828)">l'échelle</text>
   </g>
   <g id="patch_10">
    <path d="M 269.342369 121.878203 
Q 271.842787 121.878203 272.777958 121.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
    <path d="M 268.777958 119.878203 
L 272.777958 121.878203 
L 268.777958 123.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
   </g>
   <g id="text_7">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'" transform="translate(299.31668 108.666695)">Mask</text>
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'" transform="translate(298.032891 120.424398)">(opt.)</text>
   </g>
   <g id="text_8">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(295.8725 146.886578)">décodeur</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(293.586875 155.844828)">seulement</text>
   </g>
   <g id="patch_11">
    <path d="M 355.056655 121.878203 
Q 357.557073 121.878203 358.492244 121.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
    <path d="M 354.492244 119.878203 
L 358.492244 121.878203 
L 354.492244 123.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
   </g>
   <g id="text_9">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="400.414286" y="114.545547" transform="rotate(-0 400.414286 114.545547)">softmax</text>
   </g>
   <g id="text_10">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(389.693036 146.886578)">poids</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(391.989911 155.844828)">Σ=1</text>
   </g>
   <g id="patch_12">
    <path d="M 440.77094 121.878203 
Q 443.271359 121.878203 444.20653 121.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
    <path d="M 440.20653 119.878203 
L 444.20653 121.878203 
L 440.20653 123.878203 
" style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
   </g>
   <g id="text_11">
    <text style="font-weight: 700; font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="486.128571" y="114.545547" transform="rotate(-0 486.128571 114.545547)">×V</text>
   </g>
   <g id="text_12">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(467.424821 146.694578)">moyenne</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(467.039196 156.036828)">pondérée</text>
   </g>
   <g id="text_13">
    <text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #444444" x="271.842857" y="207.128203" transform="rotate(-0 271.842857 207.128203)">Exemple : dₖ=64  →  diviser par √64=8 pour éviter que le softmax ne sature</text>
   </g>
   <g id="text_14">
    <text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="277.2" y="16.398203" transform="rotate(-0 277.2 16.398203)">Scaled Dot-Product Attention : la chaîne complète des opérations</text>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p996308cd41">
   <rect x="7.2" y="26.398203" width="540" height="218.24"/>
  </clipPath>
 </defs>
</svg>
</div>

## À quoi ça ressemble, concrètement

Voici un exemple illustratif (valeurs inventées à but pédagogique, pas issues d'un modèle entraîné) sur la phrase « Le chat noir dort profondément ». Chaque ligne représente la distribution d'attention d'un mot vers tous les mots de la phrase (y compris lui-même), et somme toujours à 1 :

Ce qu'on voit : « noir » prête fortement attention à « chat » (l'adjectif regarde le nom qu'il qualifie), et « profondément » prête fortement attention à « dort » (l'adverbe regarde le verbe qu'il modifie). C'est exactement le genre de structure syntaxique que les auteurs rapportent avoir observée dans les têtes d'attention d'un vrai modèle entraîné.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="446.4pt" height="374.4pt" viewBox="0 0 446.4 374.4" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:32:44.773796</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.11.1, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 374.4 
L 446.4 374.4 
L 446.4 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 55.8 333.022687 
L 336.213374 333.022687 
L 336.213374 52.609313 
L 55.8 52.609313 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#p0cf25dde80)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAYUAAAGGCAYAAACUt53mAAAFjUlEQVR4nO3XvYkUYBSGUUd2EAYWR9gCVtAqtgMTEwM7MNIeDIxNTTUxsQQ7MNgGDA1ERPxBwR8YsycVTa4fe04Fb3J5uJsPX38dLvFX3n3+Pj1hSfvddnrCkk5vPZyesKQXT+5PT1jS5ekBAPw/RAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoAZPP204/D9IjV3Lz3fHrCkl49vjM9YUmv33+ZnrCkGyfH0xOW5FMAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFAHI0PYCL49n5m+kJS3pwdn16AheITwGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAyNHHbz+nNyxnf7KfnrCk02tXpicsyY3+m/1uOz1hST4FACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCb47tPD9MjVvPy0e3pCUu6uttOTwD+wKcAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAOQ3igAjTtKTlgMAAAAASUVORK5CYII=" id="image11ca4d3d34" transform="scale(1 -1) translate(0 -280.8)" x="56.16" y="-52.56" width="280.08" height="280.8"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="m5f070783d3" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m5f070783d3" x="83.841337" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- Le -->
      <g transform="translate(71.517196 353.612927) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-2f" d="M 628 4666 
L 1259 4666 
L 1259 531 
L 3531 531 
L 3531 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-48" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#m5f070783d3" x="139.924012" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- chat -->
      <g transform="translate(117.755056 359.297577) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-46" d="M 3122 3366 
L 3122 2828 
Q 2878 2963 2633 3030 
Q 2388 3097 2138 3097 
Q 1578 3097 1268 2742 
Q 959 2388 959 1747 
Q 959 1106 1268 751 
Q 1578 397 2138 397 
Q 2388 397 2633 464 
Q 2878 531 3122 666 
L 3122 134 
Q 2881 22 2623 -34 
Q 2366 -91 2075 -91 
Q 1284 -91 818 406 
Q 353 903 353 1747 
Q 353 2603 823 3093 
Q 1294 3584 2113 3584 
Q 2378 3584 2631 3529 
Q 2884 3475 3122 3366 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-4b" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-44" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-57" d="M 1172 4494 
L 1172 3500 
L 2356 3500 
L 2356 3053 
L 1172 3053 
L 1172 1153 
Q 1172 725 1289 603 
Q 1406 481 1766 481 
L 2356 481 
L 2356 0 
L 1766 0 
Q 1100 0 847 248 
Q 594 497 594 1153 
L 594 3053 
L 172 3053 
L 172 3500 
L 594 3500 
L 594 4494 
L 1172 4494 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#m5f070783d3" x="196.006687" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- noir -->
      <g transform="translate(176.256513 357.901093) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-51" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-52" d="M 1959 3097 
Q 1497 3097 1228 2736 
Q 959 2375 959 1747 
Q 959 1119 1226 758 
Q 1494 397 1959 397 
Q 2419 397 2687 759 
Q 2956 1122 2956 1747 
Q 2956 2369 2687 2733 
Q 2419 3097 1959 3097 
z
M 1959 3584 
Q 2709 3584 3137 3096 
Q 3566 2609 3566 1747 
Q 3566 888 3137 398 
Q 2709 -91 1959 -91 
Q 1206 -91 779 398 
Q 353 888 353 1747 
Q 353 2609 779 3096 
Q 1206 3584 1959 3584 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-4c" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-55" d="M 2631 2963 
Q 2534 3019 2420 3045 
Q 2306 3072 2169 3072 
Q 1681 3072 1420 2755 
Q 1159 2438 1159 1844 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1341 3275 1631 3429 
Q 1922 3584 2338 3584 
Q 2397 3584 2469 3576 
Q 2541 3569 2628 3553 
L 2631 2963 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-51"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.375 0)"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(124.5625 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(152.34375 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#m5f070783d3" x="252.089362" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- dort -->
      <g transform="translate(231.240688 358.535311) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-47" d="M 2906 2969 
L 2906 4863 
L 3481 4863 
L 3481 0 
L 2906 0 
L 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
z
M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_5">
      <g>
       <use xlink:href="#m5f070783d3" x="308.172036" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- profondément -->
      <g transform="translate(239.226071 386.685347) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-53" d="M 1159 525 
L 1159 -1331 
L 581 -1331 
L 581 3500 
L 1159 3500 
L 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
z
M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-49" d="M 2375 4863 
L 2375 4384 
L 1825 4384 
Q 1516 4384 1395 4259 
Q 1275 4134 1275 3809 
L 1275 3500 
L 2222 3500 
L 2222 3053 
L 1275 3053 
L 1275 0 
L 697 0 
L 697 3053 
L 147 3053 
L 147 3500 
L 697 3500 
L 697 3744 
Q 697 4328 969 4595 
Q 1241 4863 1831 4863 
L 2375 4863 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-ab" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
M 2468 5119 
L 3090 5119 
L 2072 3944 
L 1593 3944 
L 2468 5119 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-50" d="M 3328 2828 
Q 3544 3216 3844 3400 
Q 4144 3584 4550 3584 
Q 5097 3584 5394 3201 
Q 5691 2819 5691 2113 
L 5691 0 
L 5113 0 
L 5113 2094 
Q 5113 2597 4934 2840 
Q 4756 3084 4391 3084 
Q 3944 3084 3684 2787 
Q 3425 2491 3425 1978 
L 3425 0 
L 2847 0 
L 2847 2094 
Q 2847 2600 2669 2842 
Q 2491 3084 2119 3084 
Q 1678 3084 1418 2786 
Q 1159 2488 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1356 3278 1631 3431 
Q 1906 3584 2284 3584 
Q 2666 3584 2933 3390 
Q 3200 3197 3328 2828 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-53"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(102.390625 0)"/>
       <use xlink:href="#DejaVuSans-49" transform="translate(163.578125 0)"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(198.78125 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(259.96875 0)"/>
       <use xlink:href="#DejaVuSans-47" transform="translate(323.34375 0)"/>
       <use xlink:href="#DejaVuSans-ab" transform="translate(386.828125 0)"/>
       <use xlink:href="#DejaVuSans-50" transform="translate(448.359375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(545.765625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(607.296875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(670.671875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_6">
     <!-- Keyⱼ (à qui on prête attention) -->
     <g transform="translate(121.155906 400.972324) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-2e" d="M 628 4666 
L 1259 4666 
L 1259 2694 
L 3353 4666 
L 4166 4666 
L 1850 2491 
L 4331 0 
L 3500 0 
L 1259 2247 
L 1259 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-5c" d="M 2059 -325 
Q 1816 -950 1584 -1140 
Q 1353 -1331 966 -1331 
L 506 -1331 
L 506 -850 
L 844 -850 
Q 1081 -850 1212 -737 
Q 1344 -625 1503 -206 
L 1606 56 
L 191 3500 
L 800 3500 
L 1894 763 
L 2988 3500 
L 3597 3500 
L 2059 -325 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-11f6" d="M 381 1959 
L 744 1959 
L 744 -35 
Q 744 -410 581 -579 
Q 422 -747 66 -747 
L -72 -747 
L -72 -472 
L 25 -472 
Q 231 -472 306 -388 
Q 381 -304 381 -35 
L 381 1959 
z
M 381 2721 
L 744 2721 
L 744 2315 
L 381 2315 
L 381 2721 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-3" transform="scale(0.015625)"/>
       <path id="DejaVuSans-b" d="M 1984 4856 
Q 1566 4138 1362 3434 
Q 1159 2731 1159 2009 
Q 1159 1288 1364 580 
Q 1569 -128 1984 -844 
L 1484 -844 
Q 1016 -109 783 600 
Q 550 1309 550 2009 
Q 550 2706 781 3412 
Q 1013 4119 1484 4856 
L 1984 4856 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-a2" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
M 1403 5119 
L 2284 3950 
L 1806 3950 
L 787 5119 
L 1403 5119 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-54" d="M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
M 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 -1331 
L 2906 -1331 
L 2906 525 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-58" d="M 544 1381 
L 544 3500 
L 1119 3500 
L 1119 1403 
Q 1119 906 1312 657 
Q 1506 409 1894 409 
Q 2359 409 2629 706 
Q 2900 1003 2900 1516 
L 2900 3500 
L 3475 3500 
L 3475 0 
L 2900 0 
L 2900 538 
Q 2691 219 2414 64 
Q 2138 -91 1772 -91 
Q 1169 -91 856 284 
Q 544 659 544 1381 
z
M 1991 3584 
L 1991 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-ac" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
M 1803 5119 
L 2265 5119 
L 3031 3944 
L 2597 3944 
L 2034 4709 
L 1472 3944 
L 1037 3944 
L 1803 5119 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-c" d="M 513 4856 
L 1013 4856 
Q 1481 4119 1714 3412 
Q 1947 2706 1947 2009 
Q 1947 1309 1714 600 
Q 1481 -109 1013 -844 
L 513 -844 
Q 928 -128 1133 580 
Q 1338 1288 1338 2009 
Q 1338 2731 1133 3434 
Q 928 4138 513 4856 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-2e"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(60.59375 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(122.125 0)"/>
      <use xlink:href="#DejaVuSans-11f6" transform="translate(181.3125 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(198.796875 0)"/>
      <use xlink:href="#DejaVuSans-b" transform="translate(230.578125 0)"/>
      <use xlink:href="#DejaVuSans-a2" transform="translate(269.59375 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(330.875 0)"/>
      <use xlink:href="#DejaVuSans-54" transform="translate(362.65625 0)"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(426.140625 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(489.515625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(517.296875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(549.078125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(610.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(673.640625 0)"/>
      <use xlink:href="#DejaVuSans-53" transform="translate(705.421875 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(768.90625 0)"/>
      <use xlink:href="#DejaVuSans-ac" transform="translate(807.8125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(869.34375 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(908.546875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(970.078125 0)"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(1001.859375 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1063.140625 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1102.34375 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1141.546875 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(1203.078125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1266.453125 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(1305.65625 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(1333.4375 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(1394.625 0)"/>
      <use xlink:href="#DejaVuSans-c" transform="translate(1458 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_6">
      <defs>
       <path id="m33410e2a02" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m33410e2a02" x="55.8" y="80.65065" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- Le -->
      <g transform="translate(36.095 84.829361) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_7">
      <g>
       <use xlink:href="#m33410e2a02" x="55.8" y="136.733325" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- chat -->
      <g transform="translate(24.727188 140.912466) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_8">
      <g>
       <use xlink:href="#m33410e2a02" x="55.8" y="192.816" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- noir -->
      <g transform="translate(27.520156 196.995141) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-51"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.375 0)"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(124.5625 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(152.34375 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_9">
      <g>
       <use xlink:href="#m33410e2a02" x="55.8" y="248.898675" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- dort -->
      <g transform="translate(26.251719 253.077815) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_10">
      <g>
       <use xlink:href="#m33410e2a02" x="55.8" y="304.98135" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- profondément -->
      <g transform="translate(-29.28625 309.38049) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-53"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(102.390625 0)"/>
       <use xlink:href="#DejaVuSans-49" transform="translate(163.578125 0)"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(198.78125 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(259.96875 0)"/>
       <use xlink:href="#DejaVuSans-47" transform="translate(323.34375 0)"/>
       <use xlink:href="#DejaVuSans-ab" transform="translate(386.828125 0)"/>
       <use xlink:href="#DejaVuSans-50" transform="translate(448.359375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(545.765625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(607.296875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(670.671875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_12">
     <!-- Queryᵢ (position qui regarde) -->
     <g transform="translate(-35.688594 264.590219) rotate(-90) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-34" d="M 2522 4238 
Q 1834 4238 1429 3725 
Q 1025 3213 1025 2328 
Q 1025 1447 1429 934 
Q 1834 422 2522 422 
Q 3209 422 3611 934 
Q 4013 1447 4013 2328 
Q 4013 3213 3611 3725 
Q 3209 4238 2522 4238 
z
M 3406 84 
L 4238 -825 
L 3475 -825 
L 2784 -78 
Q 2681 -84 2626 -87 
Q 2572 -91 2522 -91 
Q 1538 -91 948 567 
Q 359 1225 359 2328 
Q 359 3434 948 4092 
Q 1538 4750 2522 4750 
Q 3503 4750 4090 4092 
Q 4678 3434 4678 2328 
Q 4678 1516 4351 937 
Q 4025 359 3406 84 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-8c4" d="M 381 1959 
L 744 1959 
L 744 0 
L 381 0 
L 381 1959 
z
M 381 2721 
L 744 2721 
L 744 2315 
L 381 2315 
L 381 2721 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-56" d="M 2834 3397 
L 2834 2853 
Q 2591 2978 2328 3040 
Q 2066 3103 1784 3103 
Q 1356 3103 1142 2972 
Q 928 2841 928 2578 
Q 928 2378 1081 2264 
Q 1234 2150 1697 2047 
L 1894 2003 
Q 2506 1872 2764 1633 
Q 3022 1394 3022 966 
Q 3022 478 2636 193 
Q 2250 -91 1575 -91 
Q 1294 -91 989 -36 
Q 684 19 347 128 
L 347 722 
Q 666 556 975 473 
Q 1284 391 1588 391 
Q 1994 391 2212 530 
Q 2431 669 2431 922 
Q 2431 1156 2273 1281 
Q 2116 1406 1581 1522 
L 1381 1569 
Q 847 1681 609 1914 
Q 372 2147 372 2553 
Q 372 3047 722 3315 
Q 1072 3584 1716 3584 
Q 2034 3584 2315 3537 
Q 2597 3491 2834 3397 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-4a" d="M 2906 1791 
Q 2906 2416 2648 2759 
Q 2391 3103 1925 3103 
Q 1463 3103 1205 2759 
Q 947 2416 947 1791 
Q 947 1169 1205 825 
Q 1463 481 1925 481 
Q 2391 481 2648 825 
Q 2906 1169 2906 1791 
z
M 3481 434 
Q 3481 -459 3084 -895 
Q 2688 -1331 1869 -1331 
Q 1566 -1331 1297 -1286 
Q 1028 -1241 775 -1147 
L 775 -588 
Q 1028 -725 1275 -790 
Q 1522 -856 1778 -856 
Q 2344 -856 2625 -561 
Q 2906 -266 2906 331 
L 2906 616 
Q 2728 306 2450 153 
Q 2172 0 1784 0 
Q 1141 0 747 490 
Q 353 981 353 1791 
Q 353 2603 747 3093 
Q 1141 3584 1784 3584 
Q 2172 3584 2450 3431 
Q 2728 3278 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 434 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-34"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(78.71875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(142.09375 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(203.625 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(244.734375 0)"/>
      <use xlink:href="#DejaVuSans-8c4" transform="translate(303.921875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(321.796875 0)"/>
      <use xlink:href="#DejaVuSans-b" transform="translate(353.578125 0)"/>
      <use xlink:href="#DejaVuSans-53" transform="translate(392.59375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(456.078125 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(517.265625 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(569.359375 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(597.140625 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(636.34375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(664.125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(725.3125 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(788.6875 0)"/>
      <use xlink:href="#DejaVuSans-54" transform="translate(820.46875 0)"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(883.953125 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(947.328125 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(975.109375 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(1006.890625 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1045.796875 0)"/>
      <use xlink:href="#DejaVuSans-4a" transform="translate(1107.328125 0)"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(1170.8125 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(1232.09375 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(1271.453125 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1334.9375 0)"/>
      <use xlink:href="#DejaVuSans-c" transform="translate(1396.46875 0)"/>
     </g>
    </g>
   </g>
   <g id="patch_3">
    <path d="M 55.8 333.022687 
L 55.8 52.609313 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 336.213374 333.022687 
L 336.213374 52.609313 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 55.8 333.022687 
L 336.213374 333.022687 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 55.8 52.609313 
L 336.213374 52.609313 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_13">
    <!-- 0.55 -->
    <g style="fill: #ffffff" transform="translate(73.265166 83.118424) scale(0.095 -0.095)">
     <defs>
      <path id="DejaVuSans-13" d="M 2034 4250 
Q 1547 4250 1301 3770 
Q 1056 3291 1056 2328 
Q 1056 1369 1301 889 
Q 1547 409 2034 409 
Q 2525 409 2770 889 
Q 3016 1369 3016 2328 
Q 3016 3291 2770 3770 
Q 2525 4250 2034 4250 
z
M 2034 4750 
Q 2819 4750 3233 4129 
Q 3647 3509 3647 2328 
Q 3647 1150 3233 529 
Q 2819 -91 2034 -91 
Q 1250 -91 836 529 
Q 422 1150 422 2328 
Q 422 3509 836 4129 
Q 1250 4750 2034 4750 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-11" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-18" d="M 691 4666 
L 3169 4666 
L 3169 4134 
L 1269 4134 
L 1269 2991 
Q 1406 3038 1543 3061 
Q 1681 3084 1819 3084 
Q 2600 3084 3056 2656 
Q 3513 2228 3513 1497 
Q 3513 744 3044 326 
Q 2575 -91 1722 -91 
Q 1428 -91 1123 -41 
Q 819 9 494 109 
L 494 744 
Q 775 591 1075 516 
Q 1375 441 1709 441 
Q 2250 441 2565 725 
Q 2881 1009 2881 1497 
Q 2881 1984 2565 2268 
Q 2250 2553 1709 2553 
Q 1456 2553 1204 2497 
Q 953 2441 691 2322 
L 691 4666 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_14">
    <!-- 0.15 -->
    <g style="fill: #222222" transform="translate(129.34784 83.118424) scale(0.095 -0.095)">
     <defs>
      <path id="DejaVuSans-14" d="M 794 531 
L 1825 531 
L 1825 4091 
L 703 3866 
L 703 4441 
L 1819 4666 
L 2450 4666 
L 2450 531 
L 3481 531 
L 3481 0 
L 794 0 
L 794 531 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_15">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(185.430515 83.118424) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_16">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(241.51319 83.118424) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_17">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(297.595865 83.118424) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_18">
    <!-- Σ=1.00 -->
    <g style="fill: #555555" transform="translate(347.429909 82.988541) scale(0.09 -0.09)">
     <defs>
      <path id="DejaVuSans-337" d="M 1353 531 
L 3634 531 
L 3634 0 
L 628 0 
L 628 531 
L 2125 2481 
L 628 4134 
L 628 4666 
L 3578 4666 
L 3578 4134 
L 1353 4134 
L 2850 2494 
L 1353 531 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-20" d="M 678 2906 
L 4684 2906 
L 4684 2381 
L 678 2381 
L 678 2906 
z
M 678 1631 
L 4684 1631 
L 4684 1100 
L 678 1100 
L 678 1631 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-337"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(146.984375 0)"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(210.609375 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(242.390625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(306.015625 0)"/>
    </g>
   </g>
   <g id="text_19">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(73.265166 139.201099) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_20">
    <!-- 0.50 -->
    <g style="fill: #ffffff" transform="translate(129.34784 139.201099) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_21">
    <!-- 0.25 -->
    <g style="fill: #222222" transform="translate(185.430515 139.201099) scale(0.095 -0.095)">
     <defs>
      <path id="DejaVuSans-15" d="M 1228 531 
L 3431 531 
L 3431 0 
L 469 0 
L 469 531 
Q 828 903 1448 1529 
Q 2069 2156 2228 2338 
Q 2531 2678 2651 2914 
Q 2772 3150 2772 3378 
Q 2772 3750 2511 3984 
Q 2250 4219 1831 4219 
Q 1534 4219 1204 4116 
Q 875 4013 500 3803 
L 500 4441 
Q 881 4594 1212 4672 
Q 1544 4750 1819 4750 
Q 2544 4750 2975 4387 
Q 3406 4025 3406 3419 
Q 3406 3131 3298 2873 
Q 3191 2616 2906 2266 
Q 2828 2175 2409 1742 
Q 1991 1309 1228 531 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_22">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(241.51319 139.201099) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_23">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(297.595865 139.201099) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_24">
    <!-- Σ=1.00 -->
    <g style="fill: #555555" transform="translate(347.429909 139.071216) scale(0.09 -0.09)">
     <use xlink:href="#DejaVuSans-337"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(146.984375 0)"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(210.609375 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(242.390625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(306.015625 0)"/>
    </g>
   </g>
   <g id="text_25">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(73.265166 195.283773) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_26">
    <!-- 0.55 -->
    <g style="fill: #ffffff" transform="translate(129.34784 195.283773) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_27">
    <!-- 0.30 -->
    <g style="fill: #222222" transform="translate(185.430515 195.283773) scale(0.095 -0.095)">
     <defs>
      <path id="DejaVuSans-16" d="M 2597 2516 
Q 3050 2419 3304 2112 
Q 3559 1806 3559 1356 
Q 3559 666 3084 287 
Q 2609 -91 1734 -91 
Q 1441 -91 1130 -33 
Q 819 25 488 141 
L 488 750 
Q 750 597 1062 519 
Q 1375 441 1716 441 
Q 2309 441 2620 675 
Q 2931 909 2931 1356 
Q 2931 1769 2642 2001 
Q 2353 2234 1838 2234 
L 1294 2234 
L 1294 2753 
L 1863 2753 
Q 2328 2753 2575 2939 
Q 2822 3125 2822 3475 
Q 2822 3834 2567 4026 
Q 2313 4219 1838 4219 
Q 1578 4219 1281 4162 
Q 984 4106 628 3988 
L 628 4550 
Q 988 4650 1302 4700 
Q 1616 4750 1894 4750 
Q 2613 4750 3031 4423 
Q 3450 4097 3450 3541 
Q 3450 3153 3228 2886 
Q 3006 2619 2597 2516 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_28">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(241.51319 195.283773) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_29">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(297.595865 195.283773) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_30">
    <!-- Σ=1.00 -->
    <g style="fill: #555555" transform="translate(347.429909 195.153891) scale(0.09 -0.09)">
     <use xlink:href="#DejaVuSans-337"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(146.984375 0)"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(210.609375 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(242.390625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(306.015625 0)"/>
    </g>
   </g>
   <g id="text_31">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(73.265166 251.366448) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_32">
    <!-- 0.55 -->
    <g style="fill: #ffffff" transform="translate(129.34784 251.366448) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_33">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(185.430515 251.366448) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_34">
    <!-- 0.20 -->
    <g style="fill: #222222" transform="translate(241.51319 251.366448) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_35">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(297.595865 251.366448) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_36">
    <!-- Σ=1.00 -->
    <g style="fill: #555555" transform="translate(347.429909 251.236565) scale(0.09 -0.09)">
     <use xlink:href="#DejaVuSans-337"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(146.984375 0)"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(210.609375 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(242.390625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(306.015625 0)"/>
    </g>
   </g>
   <g id="text_37">
    <!-- 0.03 -->
    <g style="fill: #222222" transform="translate(73.265166 307.449123) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_38">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(129.34784 307.449123) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_39">
    <!-- 0.05 -->
    <g style="fill: #222222" transform="translate(185.430515 307.449123) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_40">
    <!-- 0.62 -->
    <g style="fill: #ffffff" transform="translate(241.51319 307.449123) scale(0.095 -0.095)">
     <defs>
      <path id="DejaVuSans-19" d="M 2113 2584 
Q 1688 2584 1439 2293 
Q 1191 2003 1191 1497 
Q 1191 994 1439 701 
Q 1688 409 2113 409 
Q 2538 409 2786 701 
Q 3034 994 3034 1497 
Q 3034 2003 2786 2293 
Q 2538 2584 2113 2584 
z
M 3366 4563 
L 3366 3988 
Q 3128 4100 2886 4159 
Q 2644 4219 2406 4219 
Q 1781 4219 1451 3797 
Q 1122 3375 1075 2522 
Q 1259 2794 1537 2939 
Q 1816 3084 2150 3084 
Q 2853 3084 3261 2657 
Q 3669 2231 3669 1497 
Q 3669 778 3244 343 
Q 2819 -91 2113 -91 
Q 1303 -91 875 529 
Q 447 1150 447 2328 
Q 447 3434 972 4092 
Q 1497 4750 2381 4750 
Q 2619 4750 2861 4703 
Q 3103 4656 3366 4563 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-19" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_41">
    <!-- 0.20 -->
    <g style="fill: #222222" transform="translate(297.595865 307.449123) scale(0.095 -0.095)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_42">
    <!-- Σ=1.00 -->
    <g style="fill: #555555" transform="translate(347.429909 307.31924) scale(0.09 -0.09)">
     <use xlink:href="#DejaVuSans-337"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(146.984375 0)"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(210.609375 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(242.390625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(306.015625 0)"/>
    </g>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_7">
    <path d="M 387.739331 333.216 
L 401.76 333.216 
L 401.76 52.416 
L 387.739331 52.416 
z
" style="fill: #ffffff"/>
   </g>
   <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAABMAAAGGCAYAAACZjLqIAAABwElEQVR4nO2cW4oEMQwDvUPf/7r7MZDMGWQEXSDpACKFnMTOPP7+v/eOSR+X0czM41vXzHPH5+bFNC5sHqNX01yoaW7MsGk6Ma0rC8G0lgYa02jmPIKapq6muTEDYzrNsJjTNHUzMKYxTfIR5DQDY1Ib5I47C7Omqapp6mqaG7Omqapp6mqaG7OINNsg6+q4szEDYzZN1ewY4wRjhhTtycBsmrLAmOA0sSs7IQFkYMak6TTDNi7HacbFbJq6GRfT/OLikzlNIycY01ln1h0ALlrwJIy90a3bCTzuWOsM+xxt3k4+M3DRkvem0YzbuFgxuUWb8rCUcaOb96bPrCetLnCDnBJAr7pXzTq86vJikvdmBGbT1M1S3s+waYYUbciNzsVM+cZSSJoZM3pImt4jKONtOyVNLGaPoI1ZAia3c8zBTKizYuoqpq5irsyKKaqYutoe6ALX2TEOT2DMkDSLKYt7apB/jZsyO0Vgev9fA4vZNHUzLmbGysCYrbN3zZqmrpQ024a+a1ZMXT2CFmaDxZzr+xZyCKZ7ZdY642K2znSzYooqpq5ibsywH8iY0+ReKCH3ZjFFobdTMUUVU5cV8wcYxRvIsD0NiAAAAABJRU5ErkJggg==" id="image66ae04ed9a" transform="scale(1 -1) translate(0 -280.8)" x="388.08" y="-52.56" width="13.68" height="280.8"/>
   <g id="matplotlib.axis_3"/>
   <g id="matplotlib.axis_4">
    <g id="ytick_6">
     <g id="line2d_11">
      <defs>
       <path id="m7fbb178ffe" d="M 0 0 
L 3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="333.216" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_43">
      <!-- 0.0 -->
      <g transform="translate(408.76 336.634945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_12">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="290.016" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_44">
      <!-- 0.1 -->
      <g transform="translate(408.76 293.434945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_13">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="246.816" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_45">
      <!-- 0.2 -->
      <g transform="translate(408.76 250.234945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_9">
     <g id="line2d_14">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="203.616" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_46">
      <!-- 0.3 -->
      <g transform="translate(408.76 207.034945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_10">
     <g id="line2d_15">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="160.416" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_47">
      <!-- 0.4 -->
      <g transform="translate(408.76 163.834945) scale(0.09 -0.09)">
       <defs>
        <path id="DejaVuSans-17" d="M 2419 4116 
L 825 1625 
L 2419 1625 
L 2419 4116 
z
M 2253 4666 
L 3047 4666 
L 3047 1625 
L 3713 1625 
L 3713 1100 
L 3047 1100 
L 3047 0 
L 2419 0 
L 2419 1100 
L 313 1100 
L 313 1709 
L 2253 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-17" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_11">
     <g id="line2d_16">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="117.216" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_48">
      <!-- 0.5 -->
      <g transform="translate(408.76 120.634945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_12">
     <g id="line2d_17">
      <g>
       <use xlink:href="#m7fbb178ffe" x="401.76" y="74.016" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_49">
      <!-- 0.6 -->
      <g transform="translate(408.76 77.434945) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-19" transform="translate(95.40625 0)"/>
      </g>
     </g>
    </g>
    <g id="text_50">
     <!-- poids d'attention -->
     <g transform="translate(433.911406 230.927484) rotate(-90) scale(0.09 -0.09)">
      <defs>
       <path id="DejaVuSans-a" d="M 1147 4666 
L 1147 2931 
L 616 2931 
L 616 4666 
L 1147 4666 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-53"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(124.671875 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(152.453125 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(215.9375 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(268.03125 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(299.8125 0)"/>
      <use xlink:href="#DejaVuSans-a" transform="translate(363.296875 0)"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(390.78125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(452.0625 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(491.265625 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(530.46875 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(592 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(655.375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(694.578125 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(722.359375 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(783.546875 0)"/>
     </g>
    </g>
   </g>
   <g id="LineCollection_1"/>
   <g id="patch_8">
    <path d="M 387.739331 333.216 
L 394.749666 333.216 
L 401.76 333.216 
L 401.76 52.416 
L 394.749666 52.416 
L 387.739331 52.416 
L 387.739331 333.216 
z
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
  </g>
  <g id="text_51">
   <!-- Exemple illustratif : poids d'attention pour « Le chat noir dort profondément » -->
   <g transform="translate(18.55418 8.398359) scale(0.105 -0.105)">
    <defs>
     <path id="DejaVuSans-28" d="M 628 4666 
L 3578 4666 
L 3578 4134 
L 1259 4134 
L 1259 2753 
L 3481 2753 
L 3481 2222 
L 1259 2222 
L 1259 531 
L 3634 531 
L 3634 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-5b" d="M 3513 3500 
L 2247 1797 
L 3578 0 
L 2900 0 
L 1881 1375 
L 863 0 
L 184 0 
L 1544 1831 
L 300 3500 
L 978 3500 
L 1906 2253 
L 2834 3500 
L 3513 3500 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-1d" d="M 750 794 
L 1409 794 
L 1409 0 
L 750 0 
L 750 794 
z
M 750 3309 
L 1409 3309 
L 1409 2516 
L 750 2516 
L 750 3309 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-6d" d="M 3316 3309 
L 3316 2713 
L 2375 1875 
L 3316 1038 
L 3316 441 
L 1850 1747 
L 1850 2003 
L 3316 3309 
z
M 1959 3309 
L 1959 2713 
L 1019 1875 
L 1959 1038 
L 1959 441 
L 494 1747 
L 494 2003 
L 1959 3309 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-7d" d="M 603 3309 
L 2069 2003 
L 2069 1747 
L 603 441 
L 603 1038 
L 1544 1875 
L 603 2713 
L 603 3309 
z
M 1959 3309 
L 3425 2003 
L 3425 1747 
L 1959 441 
L 1959 1038 
L 2900 1875 
L 1959 2713 
L 1959 3309 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-28"/>
    <use xlink:href="#DejaVuSans-5b" transform="translate(63.1875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(119.296875 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(180.828125 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(278.234375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(341.71875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(369.5 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(431.03125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(462.8125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(490.59375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(518.375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(546.15625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(609.53125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(661.625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(700.828125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(741.9375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(803.21875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(842.421875 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(870.203125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(905.40625 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(937.1875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(970.875 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1002.65625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1066.140625 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1127.328125 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1155.109375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1218.59375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1270.6875 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1302.46875 0)"/>
    <use xlink:href="#DejaVuSans-a" transform="translate(1365.953125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1393.4375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1454.71875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1493.921875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1533.125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1594.65625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1658.03125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1697.234375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1725.015625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1786.203125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1849.578125 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1881.359375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1944.84375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(2006.03125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2069.40625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2110.515625 0)"/>
    <use xlink:href="#DejaVuSans-6d" transform="translate(2142.296875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2203.484375 0)"/>
    <use xlink:href="#DejaVuSans-2f" transform="translate(2235.265625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2289.234375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2350.765625 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(2382.546875 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(2437.53125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2500.90625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2562.1875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2601.390625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2633.171875 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2696.546875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2757.734375 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2785.515625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2826.625 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(2858.40625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2921.890625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2983.078125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(3024.1875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(3063.390625 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(3095.171875 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(3158.65625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(3197.5625 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(3258.75 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(3293.953125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3355.140625 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(3418.515625 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(3482 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(3543.53125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3640.9375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3702.46875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(3765.84375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(3805.046875 0)"/>
    <use xlink:href="#DejaVuSans-7d" transform="translate(3836.828125 0)"/>
   </g>
  </g>
  <g id="text_52">
   <!-- (valeurs inventées à but pédagogique, pas issues d'un modèle entraîné) -->
   <g transform="translate(59.898516 16.848) scale(0.09 -0.09)">
    <defs>
     <path id="DejaVuSans-59" d="M 191 3500 
L 800 3500 
L 1894 563 
L 2988 3500 
L 3597 3500 
L 2284 0 
L 1503 0 
L 191 3500 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-45" d="M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
M 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2969 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-f" d="M 750 794 
L 1409 794 
L 1409 256 
L 897 -744 
L 494 -744 
L 750 256 
L 750 794 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-aa" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
M 1581 5119 
L 2462 3950 
L 1984 3950 
L 965 5119 
L 1581 5119 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-b0" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 891 3584 
L 891 3584 
z
M 660 5119 
L 1122 5119 
L 1888 3944 
L 1454 3944 
L 891 4709 
L 329 3944 
L -106 3944 
L 660 5119 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-b"/>
    <use xlink:href="#DejaVuSans-59" transform="translate(39.015625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(98.203125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(159.484375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(187.265625 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(248.796875 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(312.171875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(353.28125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(405.375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(437.15625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(464.9375 0)"/>
    <use xlink:href="#DejaVuSans-59" transform="translate(528.3125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(587.5 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(649.03125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(712.40625 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(751.609375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(813.140625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(874.671875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(926.765625 0)"/>
    <use xlink:href="#DejaVuSans-a2" transform="translate(958.546875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1019.828125 0)"/>
    <use xlink:href="#DejaVuSans-45" transform="translate(1051.609375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1115.09375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1178.46875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1217.671875 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1249.453125 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(1312.9375 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1374.46875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1437.953125 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(1499.234375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1562.71875 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(1623.90625 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1687.390625 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(1715.171875 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1778.65625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1842.03125 0)"/>
    <use xlink:href="#DejaVuSans-f" transform="translate(1903.5625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1935.34375 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1967.125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2030.609375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2091.890625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2143.984375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2175.765625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2203.546875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2255.640625 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(2307.734375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2371.109375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2432.640625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2484.734375 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(2516.515625 0)"/>
    <use xlink:href="#DejaVuSans-a" transform="translate(2580 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(2607.484375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2670.859375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2734.234375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(2766.015625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2863.421875 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(2924.609375 0)"/>
    <use xlink:href="#DejaVuSans-aa" transform="translate(2988.09375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(3049.625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3077.40625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(3138.9375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3170.71875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3232.25 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(3295.625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(3334.828125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(3375.9375 0)"/>
    <use xlink:href="#DejaVuSans-b0" transform="translate(3437.21875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3465 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(3528.375 0)"/>
    <use xlink:href="#DejaVuSans-c" transform="translate(3589.90625 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p0cf25dde80">
   <rect x="55.8" y="52.609313" width="280.413374" height="280.413374"/>
  </clipPath>
 </defs>
</svg>
</div>

## Multi-Head Attention : plusieurs regards en parallèle

Une seule fonction d'attention, c'est une seule « moyenne pondérée » : un seul point de vue sur la phrase. Le problème : cette moyenne peut diluer de l'information qui aurait dû rester nette (un pronom qui devrait pointer très précisément vers un seul nom, par exemple).

La solution du papier : projeter Q, K, V plusieurs fois avec des matrices de projection **différentes et apprises**, calculer l'attention en parallèle sur chaque projection, puis concaténer et projeter une dernière fois.

**MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ**, avec **headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)**

Dans le papier : **h = 8 têtes**, chacune de dimension dₖ = dᵥ = 512/8 = 64. Astuce importante : grâce à cette dimension réduite par tête, le coût de calcul total reste comparable à celui d'une seule tête pleine dimension : le même budget de calcul, simplement réparti sur 8 sous-espaces au lieu d'un seul.

Chaque tête peut alors se spécialiser : une capture des relations syntaxiques de proximité, une autre des références à longue distance, etc. C'est exactement ce que les auteurs observent en inspectant les têtes d'un modèle entraîné (voir l'annexe du papier).

> **Le compromis caché :** l'étude d'ablation du papier (Table 3) montre qu'une seule tête donne 24.9 BLEU, 8 têtes donnent 25.8, et 32 têtes redescendent à 25.4. Trop peu de têtes = pas assez de perspectives différentes. Trop de têtes = chaque tête devient trop étroite (dₖ=16) pour représenter quoi que ce soit d'utile. Il y a un vrai optimum à trouver.
{: .prompt-tip }

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="554.4pt" height="352.638203pt" viewBox="0 0 554.4 352.638203" xmlns="http://www.w3.org/2000/svg" version="1.1">
<metadata>
<rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<cc:Work>
<dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
<dc:date>2026-08-26T22:54:24.519556</dc:date>
<dc:format>image/svg+xml</dc:format>
<dc:creator>
<cc:Agent>
<dc:title>Matplotlib v3.10.8, https://matplotlib.org/</dc:title>
</cc:Agent>
</dc:creator>
</cc:Work>
</rdf:RDF>
</metadata>
<defs>
<style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
</defs>
<g id="figure_1">
<g id="patch_1">
<path d="M 0 352.638203 L 554.4 352.638203 L 554.4 0 L 0 0 z " style="fill: #ffffff"/>
</g>
<g id="axes_1">
<g id="patch_2">
<path d="M 20.057143 125.485968 L 105.771429 125.485968 Q 108.771429 125.485968 108.771429 122.160556 L 108.771429 74.654674 Q 108.771429 71.329262 105.771429 71.329262 L 20.057143 71.329262 Q 17.057143 71.329262 17.057143 74.654674 L 17.057143 122.160556 Q 17.057143 125.485968 20.057143 125.485968 z " clip-path="url(#p55249ce0e6)" style="fill: #f6ddb8; stroke: #e08a2c; stroke-width: 1.5; stroke-linejoin: miter"/>
</g>
<g id="patch_3">
<path d="M 140.057143 329.286203 L 251.485714 329.286203 Q 254.057143 329.286203 254.057143 326.43585 L 254.057143 278.929968 Q 254.057143 276.079615 251.485714 276.079615 L 140.057143 276.079615 Q 137.485714 276.079615 137.485714 278.929968 L 137.485714 326.43585 Q 137.485714 329.286203 140.057143 329.286203 z " clip-path="url(#p55249ce0e6)" style="fill: #ffffff; stroke: #3b6ea5; stroke-width: 1.8; stroke-linejoin: miter"/>
</g>
<g id="patch_4">
<path d="M 140.057143 265.945027 L 251.485714 265.945027 Q 254.057143 265.945027 254.057143 263.094674 L 254.057143 215.588791 Q 254.057143 212.738438 251.485714 212.738438 L 140.057143 212.738438 Q 137.485714 212.738438 137.485714 215.588791 L 137.485714 263.094674 Q 137.485714 265.945027 140.057143 265.945027 z " clip-path="url(#p55249ce0e6)" style="fill: #ffffff; stroke: #4c9a6f; stroke-width: 1.8; stroke-linejoin: miter"/>
</g>
<g id="patch_5">
<path d="M 140.057143 202.60385 L 251.485714 202.60385 Q 254.057143 202.60385 254.057143 199.753497 L 254.057143 152.247615 Q 254.057143 149.397262 251.485714 149.397262 L 140.057143 149.397262 Q 137.485714 149.397262 137.485714 152.247615 L 137.485714 199.753497 Q 137.485714 202.60385 140.057143 202.60385 z " clip-path="url(#p55249ce0e6)" style="fill: #ffffff; stroke: #7b5ea3; stroke-width: 1.8; stroke-linejoin: miter"/>
</g>
<g id="patch_6">
<path d="M 140.057143 139.262674 L 251.485714 139.262674 Q 254.057143 139.262674 254.057143 136.412321 L 254.057143 88.906438 Q 254.057143 86.056085 251.485714 86.056085 L 140.057143 86.056085 Q 137.485714 86.056085 137.485714 88.906438 L 137.485714 136.412321 Q 137.485714 139.262674 140.057143 139.262674 z " clip-path="url(#p55249ce0e6)" style="fill: #ffffff; stroke: #c0504d; stroke-width: 1.8; stroke-linejoin: miter"/>
</g>
<g id="patch_7">
<path d="M 298.628571 239.500085 L 380.057143 239.500085 Q 383.057143 239.500085 383.057143 236.174674 L 383.057143 174.417027 Q 383.057143 171.091615 380.057143 171.091615 L 298.628571 171.091615 Q 295.628571 171.091615 295.628571 174.417027 L 295.628571 236.174674 Q 295.628571 239.500085 298.628571 239.500085 z " clip-path="url(#p55249ce0e6)" style="fill: #fff3d6; stroke: #c99a2e; stroke-width: 1.6; stroke-linejoin: miter"/>
</g>
<g id="patch_8">
<path d="M 410.057143 239.500085 L 512.914286 239.500085 Q 515.914286 239.500085 515.914286 236.174674 L 515.914286 174.417027 Q 515.914286 171.091615 512.914286 171.091615 L 410.057143 171.091615 Q 407.057143 171.091615 407.057143 174.417027 L 407.057143 236.174674 Q 407.057143 239.500085 410.057143 239.500085 z " clip-path="url(#p55249ce0e6)" style="fill: #f9d8d6; stroke: #c0504d; stroke-width: 1.6; stroke-linejoin: miter"/>
</g>
<g id="text_1">
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'" transform="translate(44.603778 95.577951)">Q, K, V</text>
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'" transform="translate(26.780887 106.215873)">d_model=512</text>
</g>
<g id="patch_9">
<path d="M 106.109912 105.127998 Q 122.914403 202.921236 139.529549 299.61259 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
<path d="M 140.82324 295.33166 L 139.529549 299.61259 L 136.881019 296.009079 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
</g>
<g id="text_2">
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #3b6ea5" x="195.771429" y="296.753256" transform="rotate(-0 195.771429 296.753256)">head₁</text>
</g>
<g id="text_3">
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(136.618147 311.146531)">Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)</text>
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(178.058147 318.985)">dₖ=dᵥ=64</text>
</g>
<g id="patch_10">
<path d="M 252.308381 300.859105 Q 272.914485 255.176585 293.014908 210.615131 " style="fill: none; stroke: #3b6ea5; stroke-width: 1.1; stroke-linecap: round"/>
<path d="M 289.547088 213.438998 L 293.014908 210.615131 L 293.19331 215.083707 " style="fill: none; stroke: #3b6ea5; stroke-width: 1.1; stroke-linecap: round"/>
</g>
<g id="patch_11">
<path d="M 106.260059 105.099051 Q 122.914673 171.251508 139.296328 236.319763 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
<path d="M 140.259237 231.95252 L 139.296328 236.319763 L 136.380279 232.929091 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
</g>
<g id="text_4">
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #4c9a6f" x="195.771429" y="233.41208" transform="rotate(-0 195.771429 233.41208)">head₂</text>
</g>
<g id="text_5">
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(136.618147 247.805355)">Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)</text>
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(178.058147 255.643824)">dₖ=dᵥ=64</text>
</g>
<g id="patch_12">
<path d="M 253.094427 238.152926 Q 272.914267 223.506452 291.745031 209.590887 " style="fill: none; stroke: #4c9a6f; stroke-width: 1.1; stroke-linecap: round"/>
<path d="M 287.33947 210.359671 L 291.745031 209.590887 L 289.716722 213.576607 " style="fill: none; stroke: #4c9a6f; stroke-width: 1.1; stroke-linecap: round"/>
</g>
<g id="patch_13">
<path d="M 106.623131 104.967702 Q 122.914661 139.580177 138.730057 173.181071 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
<path d="M 138.836157 168.710194 L 138.730057 173.181071 L 135.217015 170.413665 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
</g>
<g id="text_6">
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #7b5ea3" x="195.771429" y="170.070903" transform="rotate(-0 195.771429 170.070903)">head₃</text>
</g>
<g id="text_7">
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(136.618147 184.464178)">Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)</text>
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(178.058147 192.302647)">dₖ=dᵥ=64</text>
</g>
<g id="patch_14">
<path d="M 253.094427 177.189363 Q 272.914267 191.835837 291.745031 205.751402 " style="fill: none; stroke: #7b5ea3; stroke-width: 1.1; stroke-linecap: round"/>
<path d="M 289.716722 201.765682 L 291.745031 205.751402 L 287.33947 204.982617 " style="fill: none; stroke: #7b5ea3; stroke-width: 1.1; stroke-linecap: round"/>
</g>
<g id="patch_15">
<path d="M 107.700837 103.692876 Q 122.917227 107.909606 137.056188 111.827762 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
<path d="M 133.735568 108.832186 L 137.056188 111.827762 L 132.667355 112.686913 " style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: #888888; stroke-linecap: round"/>
</g>
<g id="text_8">
<text style="font-weight: 700; font-size: 9.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #c0504d" x="195.771429" y="106.729727" transform="rotate(-0 195.771429 106.729727)">...</text>
</g>
<g id="text_9">
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(136.618147 121.123002)">Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)</text>
<text style="font-size: 7px; font-family: 'DejaVu Sans'" transform="translate(178.058147 128.961471)">dₖ=dᵥ=64</text>
</g>
<g id="patch_16">
<path d="M 252.308381 114.483183 Q 272.914485 160.165703 293.014908 204.727158 " style="fill: none; stroke: #c0504d; stroke-width: 1.1; stroke-linecap: round"/>
<path d="M 293.19331 200.258582 L 293.014908 204.727158 L 289.547088 201.90329 " style="fill: none; stroke: #c0504d; stroke-width: 1.1; stroke-linecap: round"/>
</g>
<g id="text_10">
<text style="font-weight: 700; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(322.859498 202.882249)">Concat</text>
<text style="font-weight: 700; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(300.027037 212.400389)">(head₁,...,head₈)</text>
</g>
<g id="patch_17">
<path d="M 382.056655 205.29585 Q 392.913864 205.29585 402.205826 205.29585 " style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
<path d="M 398.205826 203.29585 L 402.205826 205.29585 L 398.205826 207.29585 " style="fill: none; stroke: #555555; stroke-width: 1.4; stroke-linecap: round"/>
</g>
<g id="text_11">
<text style="font-weight: 700; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(446.062199 202.764045)">×W^O</text>
<text style="font-weight: 700; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(413.760206 212.282186)">sortie d_model=512</text>
</g>
<g id="text_12">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #444444" x="277.2" y="50.901733" transform="rotate(-0 277.2 50.901733)">En pratique, ce papier utilise h=8 têtes (4 représentées ici pour la lisibilité)</text>
</g>
<g id="text_13">
<text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="277.2" y="16.398203" transform="rotate(-0 277.2 16.398203)">Multi-Head Attention : h attentions en parallèle, puis concaténation et projection</text>
</g>
</g>
</g>
<defs>
<clipPath id="p55249ce0e6">
<rect x="7.2" y="22.398203" width="540" height="323.04"/>
</clipPath>
</defs>
</svg>
</div>

## Pourquoi le décodeur doit être masqué

Le décodeur génère sa sortie mot par mot, dans l'ordre. Au moment de prédire le mot 3, il n'a **logiquement pas le droit** de connaître le mot 4 : ce mot n'existe pas encore au moment réel de la génération.

Le problème : pendant l'entraînement, pour paralleliser le calcul, on donne au modèle la phrase cible **entière** d'un coup. Sans précaution, la self-attention du décodeur pourrait « tricher » en regardant directement les mots futurs, qu'il n'aura jamais en situation réelle d'inférence.

La solution : un **masque causal**. On force à −∞ tous les scores de compatibilité vers une position future, avant le softmax, ce qui leur donne un poids de 0 après exponentiation. Chaque position ne peut alors « voir » qu'elle-même et les positions qui la précèdent.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="652.954483pt" height="377.86723pt" viewBox="0 0 652.954483 377.86723" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:44:29.937767</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.11.1, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 377.86723 
L 652.954483 377.86723 
L 652.954483 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 51.178594 339.16641 
L 314.242723 339.16641 
L 314.242723 76.102281 
L 51.178594 76.102281 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#p945314bdd1)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAW0AAAFtCAYAAADMATsiAAAFlElEQVR4nO3WQSoEYBjHYSMldqSUNSdwByd2BhtrG2WFBUmpaRoTxhWsfP36nucE/97Fr3dx97zc7vAvVp9foydMZblx7//yslqPnjCN3dEDAPg70QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBgjZu3l8G71hGlfnp6MnTGX7MXrBPE4O9kdPmIZPGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUIWD6+r7egRs7h9eh89YSqb75/RE6ZxeXY8esI0fNoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIaINECLaACGiDRAi2gAhog0QItoAIYvH9/V29IhZLNdfoydM5fr+ZfSEaVwcHY6eMA2fNkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4SINkCIaAOEiDZAiGgDhIg2QIhoA4T8ArQqJ3GxNjU5AAAAAElFTkSuQmCC" id="imagebe0959c31e" transform="scale(1 -1) translate(0 -262.8)" x="51.12" y="-76.18723" width="262.8" height="262.8"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="md6bb42c8e9" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#md6bb42c8e9" x="84.06161" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- Le -->
      <g transform="translate(77.99786 354.143949) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-2f" d="M 628 4666 
L 1259 4666 
L 1259 531 
L 3531 531 
L 3531 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-48" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#md6bb42c8e9" x="149.827642" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- chat -->
      <g transform="translate(138.338345 354.144769) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-46" d="M 3122 3366 
L 3122 2828 
Q 2878 2963 2633 3030 
Q 2388 3097 2138 3097 
Q 1578 3097 1268 2742 
Q 959 2388 959 1747 
Q 959 1106 1268 751 
Q 1578 397 2138 397 
Q 2388 397 2633 464 
Q 2878 531 3122 666 
L 3122 134 
Q 2881 22 2623 -34 
Q 2366 -91 2075 -91 
Q 1284 -91 818 406 
Q 353 903 353 1747 
Q 353 2603 823 3093 
Q 1294 3584 2113 3584 
Q 2378 3584 2631 3529 
Q 2884 3475 3122 3366 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-4b" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-44" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-57" d="M 1172 4494 
L 1172 3500 
L 2356 3500 
L 2356 3053 
L 1172 3053 
L 1172 1153 
Q 1172 725 1289 603 
Q 1406 481 1766 481 
L 2356 481 
L 2356 0 
L 1766 0 
Q 1100 0 847 248 
Q 594 497 594 1153 
L 594 3053 
L 172 3053 
L 172 3500 
L 594 3500 
L 594 4494 
L 1172 4494 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#md6bb42c8e9" x="215.593674" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- dort -->
      <g transform="translate(204.831995 354.144769) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-47" d="M 2906 2969 
L 2906 4863 
L 3481 4863 
L 3481 0 
L 2906 0 
L 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
z
M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-52" d="M 1959 3097 
Q 1497 3097 1228 2736 
Q 959 2375 959 1747 
Q 959 1119 1226 758 
Q 1494 397 1959 397 
Q 2419 397 2687 759 
Q 2956 1122 2956 1747 
Q 2956 2369 2687 2733 
Q 2419 3097 1959 3097 
z
M 1959 3584 
Q 2709 3584 3137 3096 
Q 3566 2609 3566 1747 
Q 3566 888 3137 398 
Q 2709 -91 1959 -91 
Q 1206 -91 779 398 
Q 353 888 353 1747 
Q 353 2609 779 3096 
Q 1206 3584 1959 3584 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-55" d="M 2631 2963 
Q 2534 3019 2420 3045 
Q 2306 3072 2169 3072 
Q 1681 3072 1420 2755 
Q 1159 2438 1159 1844 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1341 3275 1631 3429 
Q 1922 3584 2338 3584 
Q 2397 3584 2469 3576 
Q 2541 3569 2628 3553 
L 2631 2963 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#md6bb42c8e9" x="281.359707" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- bien -->
      <g transform="translate(270.010683 354.144769) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-45" d="M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
M 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2969 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-4c" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-51" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(152.796875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_5">
     <!-- Keyⱼ -->
     <g transform="translate(172.770815 368.264886) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-2e" d="M 628 4666 
L 1259 4666 
L 1259 2694 
L 3353 4666 
L 4166 4666 
L 1850 2491 
L 4331 0 
L 3500 0 
L 1259 2247 
L 1259 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-5c" d="M 2059 -325 
Q 1816 -950 1584 -1140 
Q 1353 -1331 966 -1331 
L 506 -1331 
L 506 -850 
L 844 -850 
Q 1081 -850 1212 -737 
Q 1344 -625 1503 -206 
L 1606 56 
L 191 3500 
L 800 3500 
L 1894 763 
L 2988 3500 
L 3597 3500 
L 2059 -325 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-11f6" d="M 381 1959 
L 744 1959 
L 744 -35 
Q 744 -410 581 -579 
Q 422 -747 66 -747 
L -72 -747 
L -72 -472 
L 25 -472 
Q 231 -472 306 -388 
Q 381 -304 381 -35 
L 381 1959 
z
M 381 2721 
L 744 2721 
L 744 2315 
L 381 2315 
L 381 2721 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-2e"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(60.59375 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(122.125 0)"/>
      <use xlink:href="#DejaVuSans-11f6" transform="translate(181.3125 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_5">
      <defs>
       <path id="md65c6d23bc" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#md65c6d23bc" x="51.178594" y="108.985297" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- Le -->
      <g transform="translate(32.051094 112.974066) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_6">
      <g>
       <use xlink:href="#md65c6d23bc" x="51.178594" y="174.751329" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- chat -->
      <g transform="translate(21.2 178.740509) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_7">
      <g>
       <use xlink:href="#md65c6d23bc" x="51.178594" y="240.517361" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- dort -->
      <g transform="translate(22.655234 244.506541) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_8">
      <g>
       <use xlink:href="#md65c6d23bc" x="51.178594" y="306.283393" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- bien -->
      <g transform="translate(21.480547 310.272573) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(152.796875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_10">
     <!-- Queryᵢ -->
     <g transform="translate(14.797656 223.724189) rotate(-90) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-34" d="M 2522 4238 
Q 1834 4238 1429 3725 
Q 1025 3213 1025 2328 
Q 1025 1447 1429 934 
Q 1834 422 2522 422 
Q 3209 422 3611 934 
Q 4013 1447 4013 2328 
Q 4013 3213 3611 3725 
Q 3209 4238 2522 4238 
z
M 3406 84 
L 4238 -825 
L 3475 -825 
L 2784 -78 
Q 2681 -84 2626 -87 
Q 2572 -91 2522 -91 
Q 1538 -91 948 567 
Q 359 1225 359 2328 
Q 359 3434 948 4092 
Q 1538 4750 2522 4750 
Q 3503 4750 4090 4092 
Q 4678 3434 4678 2328 
Q 4678 1516 4351 937 
Q 4025 359 3406 84 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-58" d="M 544 1381 
L 544 3500 
L 1119 3500 
L 1119 1403 
Q 1119 906 1312 657 
Q 1506 409 1894 409 
Q 2359 409 2629 706 
Q 2900 1003 2900 1516 
L 2900 3500 
L 3475 3500 
L 3475 0 
L 2900 0 
L 2900 538 
Q 2691 219 2414 64 
Q 2138 -91 1772 -91 
Q 1169 -91 856 284 
Q 544 659 544 1381 
z
M 1991 3584 
L 1991 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-8c4" d="M 381 1959 
L 744 1959 
L 744 0 
L 381 0 
L 381 1959 
z
M 381 2721 
L 744 2721 
L 744 2315 
L 381 2315 
L 381 2721 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-34"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(78.71875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(142.09375 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(203.625 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(244.734375 0)"/>
      <use xlink:href="#DejaVuSans-8c4" transform="translate(303.921875 0)"/>
     </g>
    </g>
   </g>
   <g id="patch_3">
    <path d="M 51.178594 339.16641 
L 51.178594 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 314.242723 339.16641 
L 314.242723 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 51.178594 339.16641 
L 314.242723 339.16641 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 51.178594 76.102281 
L 314.242723 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_11">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(72.928797 111.582953) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-13" d="M 2034 4250 
Q 1547 4250 1301 3770 
Q 1056 3291 1056 2328 
Q 1056 1369 1301 889 
Q 1547 409 2034 409 
Q 2525 409 2770 889 
Q 3016 1369 3016 2328 
Q 3016 3291 2770 3770 
Q 2525 4250 2034 4250 
z
M 2034 4750 
Q 2819 4750 3233 4129 
Q 3647 3509 3647 2328 
Q 3647 1150 3233 529 
Q 2819 -91 2034 -91 
Q 1250 -91 836 529 
Q 422 1150 422 2328 
Q 422 3509 836 4129 
Q 1250 4750 2034 4750 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-11" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-14" d="M 794 531 
L 1825 531 
L 1825 4091 
L 703 3866 
L 703 4441 
L 1819 4666 
L 2450 4666 
L 2450 531 
L 3481 531 
L 3481 0 
L 794 0 
L 794 531 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_12">
    <!-- 0.16 -->
    <g style="fill: #222222" transform="translate(138.69483 111.582953) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-19" d="M 2113 2584 
Q 1688 2584 1439 2293 
Q 1191 2003 1191 1497 
Q 1191 994 1439 701 
Q 1688 409 2113 409 
Q 2538 409 2786 701 
Q 3034 994 3034 1497 
Q 3034 2003 2786 2293 
Q 2538 2584 2113 2584 
z
M 3366 4563 
L 3366 3988 
Q 3128 4100 2886 4159 
Q 2644 4219 2406 4219 
Q 1781 4219 1451 3797 
Q 1122 3375 1075 2522 
Q 1259 2794 1537 2939 
Q 1816 3084 2150 3084 
Q 2853 3084 3261 2657 
Q 3669 2231 3669 1497 
Q 3669 778 3244 343 
Q 2819 -91 2113 -91 
Q 1303 -91 875 529 
Q 447 1150 447 2328 
Q 447 3434 972 4092 
Q 1497 4750 2381 4750 
Q 2619 4750 2861 4703 
Q 3103 4656 3366 4563 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-19" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_13">
    <!-- 0.42 -->
    <g style="fill: #222222" transform="translate(204.460862 111.582953) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-17" d="M 2419 4116 
L 825 1625 
L 2419 1625 
L 2419 4116 
z
M 2253 4666 
L 3047 4666 
L 3047 1625 
L 3713 1625 
L 3713 1100 
L 3047 1100 
L 3047 0 
L 2419 0 
L 2419 1100 
L 313 1100 
L 313 1709 
L 2253 4666 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-15" d="M 1228 531 
L 3431 531 
L 3431 0 
L 469 0 
L 469 531 
Q 828 903 1448 1529 
Q 2069 2156 2228 2338 
Q 2531 2678 2651 2914 
Q 2772 3150 2772 3378 
Q 2772 3750 2511 3984 
Q 2250 4219 1831 4219 
Q 1534 4219 1204 4116 
Q 875 4013 500 3803 
L 500 4441 
Q 881 4594 1212 4672 
Q 1544 4750 1819 4750 
Q 2544 4750 2975 4387 
Q 3406 4025 3406 3419 
Q 3406 3131 3298 2873 
Q 3191 2616 2906 2266 
Q 2828 2175 2409 1742 
Q 1991 1309 1228 531 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_14">
    <!-- 0.32 -->
    <g style="fill: #222222" transform="translate(270.226894 111.582953) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-16" d="M 2597 2516 
Q 3050 2419 3304 2112 
Q 3559 1806 3559 1356 
Q 3559 666 3084 287 
Q 2609 -91 1734 -91 
Q 1441 -91 1130 -33 
Q 819 25 488 141 
L 488 750 
Q 750 597 1062 519 
Q 1375 441 1716 441 
Q 2309 441 2620 675 
Q 2931 909 2931 1356 
Q 2931 1769 2642 2001 
Q 2353 2234 1838 2234 
L 1294 2234 
L 1294 2753 
L 1863 2753 
Q 2328 2753 2575 2939 
Q 2822 3125 2822 3475 
Q 2822 3834 2567 4026 
Q 2313 4219 1838 4219 
Q 1578 4219 1281 4162 
Q 984 4106 628 3988 
L 628 4550 
Q 988 4650 1302 4700 
Q 1616 4750 1894 4750 
Q 2613 4750 3031 4423 
Q 3450 4097 3450 3541 
Q 3450 3153 3228 2886 
Q 3006 2619 2597 2516 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_15">
    <!-- 0.13 -->
    <g style="fill: #222222" transform="translate(72.928797 177.348985) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_16">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(138.69483 177.348985) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_17">
    <!-- 0.36 -->
    <g style="fill: #222222" transform="translate(204.460862 177.348985) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-19" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_18">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(270.226894 177.348985) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-1a" d="M 525 4666 
L 3525 4666 
L 3525 4397 
L 1831 0 
L 1172 0 
L 2766 4134 
L 525 4134 
L 525 4666 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_19">
    <!-- 0.38 -->
    <g style="fill: #222222" transform="translate(72.928797 243.115017) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-1b" d="M 2034 2216 
Q 1584 2216 1326 1975 
Q 1069 1734 1069 1313 
Q 1069 891 1326 650 
Q 1584 409 2034 409 
Q 2484 409 2743 651 
Q 3003 894 3003 1313 
Q 3003 1734 2745 1975 
Q 2488 2216 2034 2216 
z
M 1403 2484 
Q 997 2584 770 2862 
Q 544 3141 544 3541 
Q 544 4100 942 4425 
Q 1341 4750 2034 4750 
Q 2731 4750 3128 4425 
Q 3525 4100 3525 3541 
Q 3525 3141 3298 2862 
Q 3072 2584 2669 2484 
Q 3125 2378 3379 2068 
Q 3634 1759 3634 1313 
Q 3634 634 3220 271 
Q 2806 -91 2034 -91 
Q 1263 -91 848 271 
Q 434 634 434 1313 
Q 434 1759 690 2068 
Q 947 2378 1403 2484 
z
M 1172 3481 
Q 1172 3119 1398 2916 
Q 1625 2713 2034 2713 
Q 2441 2713 2670 2916 
Q 2900 3119 2900 3481 
Q 2900 3844 2670 4047 
Q 2441 4250 2034 4250 
Q 1625 4250 1398 4047 
Q 1172 3844 1172 3481 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1b" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_20">
    <!-- 0.11 -->
    <g style="fill: #222222" transform="translate(138.69483 243.115017) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_21">
    <!-- 0.23 -->
    <g style="fill: #222222" transform="translate(204.460862 243.115017) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_22">
    <!-- 0.28 -->
    <g style="fill: #222222" transform="translate(270.226894 243.115017) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1b" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_23">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(72.928797 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_24">
    <!-- 0.22 -->
    <g style="fill: #222222" transform="translate(138.69483 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_25">
    <!-- 0.27 -->
    <g style="fill: #222222" transform="translate(204.460862 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_26">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(270.226894 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_27">
    <!-- triche : voit le futur -->
    <g style="fill: #555555" transform="translate(139.162611 49.795868) scale(0.09 -0.09)">
     <defs>
      <path id="DejaVuSans-3" transform="scale(0.015625)"/>
      <path id="DejaVuSans-1d" d="M 750 794 
L 1409 794 
L 1409 0 
L 750 0 
L 750 794 
z
M 750 3309 
L 1409 3309 
L 1409 2516 
L 750 2516 
L 750 3309 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-59" d="M 191 3500 
L 800 3500 
L 1894 563 
L 2988 3500 
L 3597 3500 
L 2284 0 
L 1503 0 
L 191 3500 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-49" d="M 2375 4863 
L 2375 4384 
L 1825 4384 
Q 1516 4384 1395 4259 
Q 1275 4134 1275 3809 
L 1275 3500 
L 2222 3500 
L 2222 3053 
L 1275 3053 
L 1275 0 
L 697 0 
L 697 3053 
L 147 3053 
L 147 3500 
L 697 3500 
L 697 3744 
Q 697 4328 969 4595 
Q 1241 4863 1831 4863 
L 2375 4863 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-57"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(39.203125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(80.3125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(108.09375 0)"/>
     <use xlink:href="#DejaVuSans-4b" transform="translate(163.078125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(226.453125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(287.984375 0)"/>
     <use xlink:href="#DejaVuSans-1d" transform="translate(319.765625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(353.453125 0)"/>
     <use xlink:href="#DejaVuSans-59" transform="translate(385.234375 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(444.421875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(505.609375 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(533.390625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(572.59375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(604.375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(632.15625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(693.6875 0)"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(725.46875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(760.671875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(824.046875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(863.25 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(926.625 0)"/>
    </g>
   </g>
   <g id="text_28">
    <!-- Sans masque (interdit) -->
    <g transform="translate(119.861127 70.102281) scale(0.11 -0.11)">
     <defs>
      <path id="DejaVuSans-36" d="M 3425 4513 
L 3425 3897 
Q 3066 4069 2747 4153 
Q 2428 4238 2131 4238 
Q 1616 4238 1336 4038 
Q 1056 3838 1056 3469 
Q 1056 3159 1242 3001 
Q 1428 2844 1947 2747 
L 2328 2669 
Q 3034 2534 3370 2195 
Q 3706 1856 3706 1288 
Q 3706 609 3251 259 
Q 2797 -91 1919 -91 
Q 1588 -91 1214 -16 
Q 841 59 441 206 
L 441 856 
Q 825 641 1194 531 
Q 1563 422 1919 422 
Q 2459 422 2753 634 
Q 3047 847 3047 1241 
Q 3047 1584 2836 1778 
Q 2625 1972 2144 2069 
L 1759 2144 
Q 1053 2284 737 2584 
Q 422 2884 422 3419 
Q 422 4038 858 4394 
Q 1294 4750 2059 4750 
Q 2388 4750 2728 4690 
Q 3069 4631 3425 4513 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-56" d="M 2834 3397 
L 2834 2853 
Q 2591 2978 2328 3040 
Q 2066 3103 1784 3103 
Q 1356 3103 1142 2972 
Q 928 2841 928 2578 
Q 928 2378 1081 2264 
Q 1234 2150 1697 2047 
L 1894 2003 
Q 2506 1872 2764 1633 
Q 3022 1394 3022 966 
Q 3022 478 2636 193 
Q 2250 -91 1575 -91 
Q 1294 -91 989 -36 
Q 684 19 347 128 
L 347 722 
Q 666 556 975 473 
Q 1284 391 1588 391 
Q 1994 391 2212 530 
Q 2431 669 2431 922 
Q 2431 1156 2273 1281 
Q 2116 1406 1581 1522 
L 1381 1569 
Q 847 1681 609 1914 
Q 372 2147 372 2553 
Q 372 3047 722 3315 
Q 1072 3584 1716 3584 
Q 2034 3584 2315 3537 
Q 2597 3491 2834 3397 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-50" d="M 3328 2828 
Q 3544 3216 3844 3400 
Q 4144 3584 4550 3584 
Q 5097 3584 5394 3201 
Q 5691 2819 5691 2113 
L 5691 0 
L 5113 0 
L 5113 2094 
Q 5113 2597 4934 2840 
Q 4756 3084 4391 3084 
Q 3944 3084 3684 2787 
Q 3425 2491 3425 1978 
L 3425 0 
L 2847 0 
L 2847 2094 
Q 2847 2600 2669 2842 
Q 2491 3084 2119 3084 
Q 1678 3084 1418 2786 
Q 1159 2488 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1356 3278 1631 3431 
Q 1906 3584 2284 3584 
Q 2666 3584 2933 3390 
Q 3200 3197 3328 2828 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-54" d="M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
M 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 -1331 
L 2906 -1331 
L 2906 525 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-b" d="M 1984 4856 
Q 1566 4138 1362 3434 
Q 1159 2731 1159 2009 
Q 1159 1288 1364 580 
Q 1569 -128 1984 -844 
L 1484 -844 
Q 1016 -109 783 600 
Q 550 1309 550 2009 
Q 550 2706 781 3412 
Q 1013 4119 1484 4856 
L 1984 4856 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-c" d="M 513 4856 
L 1013 4856 
Q 1481 4119 1714 3412 
Q 1947 2706 1947 2009 
Q 1947 1309 1714 600 
Q 1481 -109 1013 -844 
L 513 -844 
Q 928 -128 1133 580 
Q 1338 1288 1338 2009 
Q 1338 2731 1133 3434 
Q 928 4138 513 4856 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-36"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(63.484375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(124.765625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(188.140625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(240.234375 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(272.015625 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(369.421875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(430.703125 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(482.796875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(546.28125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(609.65625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(671.1875 0)"/>
     <use xlink:href="#DejaVuSans-b" transform="translate(702.96875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(741.984375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(769.765625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(833.140625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(872.34375 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(933.875 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(973.234375 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1036.71875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(1064.5 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1103.703125 0)"/>
    </g>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_7">
    <path d="M 382.690354 339.16641 
L 645.754483 339.16641 
L 645.754483 76.102281 
L 382.690354 76.102281 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#pb894fa2e3a)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAW0AAAFtCAYAAADMATsiAAAFiklEQVR4nO3WPS4FUBRGUU8U5qBUSwxBpzQQBqDXKk3APFRaEZXodK8QhYL4i+SZgsa72blrjeDLKXbO4n75ttpgLd6/fkZPmMbbt1uv09P75+gJ09gcPQCAvxNtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AkK3z68fRG6ZxerA7esI0lq8foydM5WhvZ/SEafi0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIGRx9fC8Gj1iFieXt6MnTOPu7HD0BPgXPm2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AENEGCBFtgBDRBggRbYAQ0QYIEW2AkMX2/vFq9IhZvNxcjJ4AxPm0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIES0AUJEGyBEtAFCRBsgRLQBQkQbIOQXr3cggFuCDE0AAAAASUVORK5CYII=" id="imaged02effe100" transform="scale(1 -1) translate(0 -262.8)" x="383.04" y="-76.18723" width="262.8" height="262.8"/>
   </g>
   <g id="patch_8">
    <path d="M 448.456386 76.102281 
L 514.222418 76.102281 
L 514.222418 141.868313 
L 448.456386 141.868313 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="patch_9">
    <path d="M 514.222418 76.102281 
L 579.98845 76.102281 
L 579.98845 141.868313 
L 514.222418 141.868313 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="patch_10">
    <path d="M 579.98845 76.102281 
L 645.754483 76.102281 
L 645.754483 141.868313 
L 579.98845 141.868313 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="patch_11">
    <path d="M 514.222418 141.868313 
L 579.98845 141.868313 
L 579.98845 207.634345 
L 514.222418 207.634345 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="patch_12">
    <path d="M 579.98845 141.868313 
L 645.754483 141.868313 
L 645.754483 207.634345 
L 579.98845 207.634345 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="patch_13">
    <path d="M 579.98845 207.634345 
L 645.754483 207.634345 
L 645.754483 273.400377 
L 579.98845 273.400377 
z
" clip-path="url(#pb894fa2e3a)" style="fill: url(#h3d9e1170bf)"/>
   </g>
   <g id="matplotlib.axis_3">
    <g id="xtick_5">
     <g id="line2d_9">
      <g>
       <use xlink:href="#md6bb42c8e9" x="415.57337" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_29">
      <!-- Le -->
      <g transform="translate(409.50962 354.143949) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_10">
      <g>
       <use xlink:href="#md6bb42c8e9" x="481.339402" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_30">
      <!-- chat -->
      <g transform="translate(469.850105 354.144769) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_11">
      <g>
       <use xlink:href="#md6bb42c8e9" x="547.105434" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_31">
      <!-- dort -->
      <g transform="translate(536.343755 354.144769) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_8">
     <g id="line2d_12">
      <g>
       <use xlink:href="#md6bb42c8e9" x="612.871467" y="339.16641" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_32">
      <!-- bien -->
      <g transform="translate(601.522443 354.144769) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(152.796875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_33">
     <!-- Keyⱼ -->
     <g transform="translate(504.282575 368.264886) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-2e"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(60.59375 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(122.125 0)"/>
      <use xlink:href="#DejaVuSans-11f6" transform="translate(181.3125 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_4">
    <g id="ytick_5">
     <g id="line2d_13">
      <g>
       <use xlink:href="#md65c6d23bc" x="382.690354" y="108.985297" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_34">
      <!-- Le -->
      <g transform="translate(363.562854 112.974066) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-2f"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(53.96875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_14">
      <g>
       <use xlink:href="#md65c6d23bc" x="382.690354" y="174.751329" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_35">
      <!-- chat -->
      <g transform="translate(352.71176 178.740509) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(118.359375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(179.640625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_15">
      <g>
       <use xlink:href="#md65c6d23bc" x="382.690354" y="240.517361" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_36">
      <!-- dort -->
      <g transform="translate(354.166994 244.506541) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-55" transform="translate(124.671875 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(165.78125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_16">
      <g>
       <use xlink:href="#md65c6d23bc" x="382.690354" y="306.283393" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_37">
      <!-- bien -->
      <g transform="translate(352.992307 310.272573) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-51" transform="translate(152.796875 0)"/>
      </g>
     </g>
    </g>
    <g id="text_38">
     <!-- Queryᵢ -->
     <g transform="translate(346.309416 223.724189) rotate(-90) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-34"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(78.71875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(142.09375 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(203.625 0)"/>
      <use xlink:href="#DejaVuSans-5c" transform="translate(244.734375 0)"/>
      <use xlink:href="#DejaVuSans-8c4" transform="translate(303.921875 0)"/>
     </g>
    </g>
   </g>
   <g id="patch_14">
    <path d="M 382.690354 339.16641 
L 382.690354 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_15">
    <path d="M 645.754483 339.16641 
L 645.754483 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_16">
    <path d="M 382.690354 339.16641 
L 645.754483 339.16641 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_17">
    <path d="M 382.690354 76.102281 
L 645.754483 76.102281 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_39">
    <!-- 1.00 -->
    <g style="fill: #ffffff" transform="translate(404.440557 111.582953) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-14"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_40">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(472.56698 111.712836) scale(0.105 -0.105)">
     <defs>
      <path id="DejaVuSans-c9c" d="M 678 2272 
L 4684 2272 
L 4684 1741 
L 678 1741 
L 678 2272 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-ca8" d="M 2916 1091 
Q 2819 1203 2666 1466 
Q 2456 1091 2272 925 
Q 2041 725 1681 725 
Q 1259 725 981 1041 
Q 688 1372 688 1919 
Q 688 2444 981 2800 
Q 1244 3116 1688 3116 
Q 1916 3116 2084 3022 
Q 2281 2919 2416 2741 
Q 2541 2581 2666 2366 
Q 2875 2741 3059 2906 
Q 3291 3106 3650 3106 
Q 4072 3106 4350 2791 
Q 4644 2459 4644 1913 
Q 4644 1388 4350 1031 
Q 4088 716 3644 716 
Q 3416 716 3247 809 
Q 3078 894 2916 1091 
z
M 1647 1134 
Q 2163 1134 2472 1884 
Q 2075 2703 1647 2703 
Q 1334 2703 1175 2478 
Q 1003 2238 1003 1919 
Q 1003 1569 1175 1353 
Q 1350 1134 1647 1134 
z
M 3684 2697 
Q 3219 2697 2859 1947 
Q 3253 1128 3684 1128 
Q 3997 1128 4156 1353 
Q 4328 1594 4328 1913 
Q 4328 2263 4156 2478 
Q 3981 2697 3684 2697 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_41">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(538.333012 111.712836) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_42">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(604.099045 111.712836) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_43">
    <!-- 0.28 -->
    <g style="fill: #222222" transform="translate(404.440557 177.348985) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1b" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_44">
    <!-- 0.72 -->
    <g style="fill: #ffffff" transform="translate(470.20659 177.348985) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_45">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(538.333012 177.478868) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_46">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(604.099045 177.478868) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_47">
    <!-- 0.53 -->
    <g style="fill: #ffffff" transform="translate(404.440557 243.115017) scale(0.1 -0.1)">
     <defs>
      <path id="DejaVuSans-18" d="M 691 4666 
L 3169 4666 
L 3169 4134 
L 1269 4134 
L 1269 2991 
Q 1406 3038 1543 3061 
Q 1681 3084 1819 3084 
Q 2600 3084 3056 2656 
Q 3513 2228 3513 1497 
Q 3513 744 3044 326 
Q 2575 -91 1722 -91 
Q 1428 -91 1123 -41 
Q 819 9 494 109 
L 494 744 
Q 775 591 1075 516 
Q 1375 441 1709 441 
Q 2250 441 2565 725 
Q 2881 1009 2881 1497 
Q 2881 1984 2565 2268 
Q 2250 2553 1709 2553 
Q 1456 2553 1204 2497 
Q 953 2441 691 2322 
L 691 4666 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_48">
    <!-- 0.15 -->
    <g style="fill: #222222" transform="translate(470.20659 243.115017) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_49">
    <!-- 0.32 -->
    <g style="fill: #222222" transform="translate(535.972622 243.115017) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_50">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(604.099045 243.2449) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_51">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(404.440557 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_52">
    <!-- 0.22 -->
    <g style="fill: #222222" transform="translate(470.20659 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_53">
    <!-- 0.27 -->
    <g style="fill: #222222" transform="translate(535.972622 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_54">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(601.738654 308.88105) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_55">
    <!-- position i ne voit que j ≤ i -->
    <g style="fill: #555555" transform="translate(456.336246 49.795868) scale(0.09 -0.09)">
     <defs>
      <path id="DejaVuSans-53" d="M 1159 525 
L 1159 -1331 
L 581 -1331 
L 581 3500 
L 1159 3500 
L 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
z
M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-4d" d="M 603 3500 
L 1178 3500 
L 1178 -63 
Q 1178 -731 923 -1031 
Q 669 -1331 103 -1331 
L -116 -1331 
L -116 -844 
L 38 -844 
Q 366 -844 484 -692 
Q 603 -541 603 -63 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-cee" d="M 4684 3175 
L 1684 2309 
L 4684 1453 
L 4684 897 
L 678 2047 
L 678 2578 
L 4684 3725 
L 4684 3175 
z
M 678 531 
L 4684 531 
L 4684 0 
L 678 0 
L 678 531 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-53"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(124.671875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(176.765625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(204.546875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(243.75 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(271.53125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(332.71875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(396.09375 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(427.875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(455.65625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(487.4375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(550.8125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(612.34375 0)"/>
     <use xlink:href="#DejaVuSans-59" transform="translate(644.125 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(703.3125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(764.5 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(792.28125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(831.484375 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(863.265625 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(926.75 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(990.125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1051.65625 0)"/>
     <use xlink:href="#DejaVuSans-4d" transform="translate(1083.4375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1111.21875 0)"/>
     <use xlink:href="#DejaVuSans-cee" transform="translate(1143 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1226.796875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1258.578125 0)"/>
    </g>
   </g>
   <g id="text_56">
    <!-- Avec masque causal (correct) -->
    <g transform="translate(432.726168 70.102281) scale(0.11 -0.11)">
     <defs>
      <path id="DejaVuSans-24" d="M 2188 4044 
L 1331 1722 
L 3047 1722 
L 2188 4044 
z
M 1831 4666 
L 2547 4666 
L 4325 0 
L 3669 0 
L 3244 1197 
L 1141 1197 
L 716 0 
L 50 0 
L 1831 4666 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-24"/>
     <use xlink:href="#DejaVuSans-59" transform="translate(62.546875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(121.734375 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(183.265625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(238.25 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(270.03125 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(367.4375 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(428.71875 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(480.8125 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(544.296875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(607.671875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(669.203125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(700.984375 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(755.96875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(817.25 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(880.625 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(932.71875 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(994 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1021.78125 0)"/>
     <use xlink:href="#DejaVuSans-b" transform="translate(1053.5625 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(1092.578125 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1147.5625 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(1208.75 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(1248.109375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1287.015625 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(1348.546875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(1403.53125 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1442.734375 0)"/>
    </g>
   </g>
  </g>
  <g id="text_57">
   <!-- Self-attention du décodeur : pourquoi le masquage est indispensable -->
   <g transform="translate(110.313179 17.198047) scale(0.125 -0.125)">
    <defs>
     <path id="DejaVuSans-10" d="M 313 2009 
L 1997 2009 
L 1997 1497 
L 313 1497 
L 313 2009 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-ab" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
M 2468 5119 
L 3090 5119 
L 2072 3944 
L 1593 3944 
L 2468 5119 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-4a" d="M 2906 1791 
Q 2906 2416 2648 2759 
Q 2391 3103 1925 3103 
Q 1463 3103 1205 2759 
Q 947 2416 947 1791 
Q 947 1169 1205 825 
Q 1463 481 1925 481 
Q 2391 481 2648 825 
Q 2906 1169 2906 1791 
z
M 3481 434 
Q 3481 -459 3084 -895 
Q 2688 -1331 1869 -1331 
Q 1566 -1331 1297 -1286 
Q 1028 -1241 775 -1147 
L 775 -588 
Q 1028 -725 1275 -790 
Q 1522 -856 1778 -856 
Q 2344 -856 2625 -561 
Q 2906 -266 2906 331 
L 2906 616 
Q 2728 306 2450 153 
Q 2172 0 1784 0 
Q 1141 0 747 490 
Q 353 981 353 1791 
Q 353 2603 747 3093 
Q 1141 3584 1784 3584 
Q 2172 3584 2450 3431 
Q 2728 3278 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 434 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-36"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(63.484375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(125.015625 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(152.796875 0)"/>
    <use xlink:href="#DejaVuSans-10" transform="translate(182.53125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(218.609375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(279.890625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(319.09375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(358.296875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(419.828125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(483.203125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(522.40625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(550.1875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(611.375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(674.75 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(706.53125 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(770.015625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(833.390625 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(865.171875 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(928.65625 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(990.1875 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1045.171875 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1106.359375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1169.84375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1231.375 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(1294.75 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1335.859375 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(1367.640625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1401.328125 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1433.109375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1496.59375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1557.78125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(1621.15625 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(1660.515625 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1724 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1787.375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1848.5625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1876.34375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(1908.125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1935.90625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1997.4375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(2029.21875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2126.625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2187.90625 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(2240 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(2303.484375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2366.859375 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(2428.140625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2491.625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2553.15625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2584.9375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2646.46875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2698.5625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2737.765625 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2769.546875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2797.328125 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(2860.703125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2924.1875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2951.96875 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(3004.0625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3067.546875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3129.078125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(3192.453125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(3244.546875 0)"/>
    <use xlink:href="#DejaVuSans-45" transform="translate(3305.828125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(3369.3125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3397.09375 0)"/>
   </g>
  </g>
  <g id="text_58">
   <!-- (exemple : génération de « Le chat dort bien ») -->
   <g transform="translate(213.801304 33.696) scale(0.095 -0.095)">
    <defs>
     <path id="DejaVuSans-5b" d="M 3513 3500 
L 2247 1797 
L 3578 0 
L 2900 0 
L 1881 1375 
L 863 0 
L 184 0 
L 1544 1831 
L 300 3500 
L 978 3500 
L 1906 2253 
L 2834 3500 
L 3513 3500 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-6d" d="M 3316 3309 
L 3316 2713 
L 2375 1875 
L 3316 1038 
L 3316 441 
L 1850 1747 
L 1850 2003 
L 3316 3309 
z
M 1959 3309 
L 1959 2713 
L 1019 1875 
L 1959 1038 
L 1959 441 
L 494 1747 
L 494 2003 
L 1959 3309 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-7d" d="M 603 3309 
L 2069 2003 
L 2069 1747 
L 603 441 
L 603 1038 
L 1544 1875 
L 603 2713 
L 603 3309 
z
M 1959 3309 
L 3425 2003 
L 3425 1747 
L 1959 441 
L 1959 1038 
L 2900 1875 
L 1959 2713 
L 1959 3309 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-b"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(39.015625 0)"/>
    <use xlink:href="#DejaVuSans-5b" transform="translate(98.796875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(154.90625 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(216.4375 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(313.84375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(377.328125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(405.109375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(466.640625 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(498.421875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(532.109375 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(563.890625 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(627.375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(688.90625 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(752.28125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(813.8125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(854.921875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(916.203125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(955.40625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(983.1875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1044.375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1107.75 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1139.53125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1203.015625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1264.546875 0)"/>
    <use xlink:href="#DejaVuSans-6d" transform="translate(1296.328125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1357.515625 0)"/>
    <use xlink:href="#DejaVuSans-2f" transform="translate(1389.296875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1443.265625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1504.796875 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1536.578125 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1591.5625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1654.9375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1716.21875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1755.421875 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1787.203125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1850.6875 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(1911.875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1952.984375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1992.1875 0)"/>
    <use xlink:href="#DejaVuSans-45" transform="translate(2023.96875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2087.453125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2115.234375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2176.765625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2240.140625 0)"/>
    <use xlink:href="#DejaVuSans-7d" transform="translate(2271.921875 0)"/>
    <use xlink:href="#DejaVuSans-c" transform="translate(2333.109375 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p945314bdd1">
   <rect x="51.178594" y="76.102281" width="263.064129" height="263.064129"/>
  </clipPath>
  <clipPath id="pb894fa2e3a">
   <rect x="382.690354" y="76.102281" width="263.064129" height="263.064129"/>
  </clipPath>
 </defs>
 <defs>
  <pattern id="h3d9e1170bf" patternUnits="userSpaceOnUse" x="0" y="0" width="72" height="72">
   <rect x="0" y="0" width="73" height="73" fill="#dddddd"/>
   <path d="M -36 36 
L 36 -36 
M -33 39 
L 39 -33 
M -30 42 
L 42 -30 
M -27 45 
L 45 -27 
M -24 48 
L 48 -24 
M -21 51 
L 51 -21 
M -18 54 
L 54 -18 
M -15 57 
L 57 -15 
M -12 60 
L 60 -12 
M -9 63 
L 63 -9 
M -6 66 
L 66 -6 
M -3 69 
L 69 -3 
M 0 72 
L 72 0 
M 3 75 
L 75 3 
M 6 78 
L 78 6 
M 9 81 
L 81 9 
M 12 84 
L 84 12 
M 15 87 
L 87 15 
M 18 90 
L 90 18 
M 21 93 
L 93 21 
M 24 96 
L 96 24 
M 27 99 
L 99 27 
M 30 102 
L 102 30 
M 33 105 
L 105 33 
M 36 108 
L 108 36 
" style="fill: #000000; stroke: #000000; stroke-width: 1.0; stroke-linecap: butt; stroke-linejoin: miter"/>
  </pattern>
 </defs>
</svg>
</div>

## Assembler le tout : l'architecture complète

L'encodeur (6 couches identiques) transforme la phrase source en représentations continues. Chaque couche a deux sous-couches : self-attention multi-têtes, puis un réseau feed-forward, chacune entourée d'une connexion résiduelle et d'une layer normalization (`LayerNorm(x + Sublayer(x))`).

Le décodeur (6 couches aussi) fait la même chose, avec une sous-couche supplémentaire : l'**encoder-decoder attention**, où les queries viennent du décodeur et les keys/values de la sortie de l'encodeur. Logique : le décodeur *pose la question* (« qu'est-ce qui, dans la phrase source, est pertinent maintenant ?»), et l'encodeur *fournit la banque d'information* dans laquelle chercher la réponse.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="640.8pt" height="511.157969pt" viewBox="0 0 640.8 511.157969" xmlns="http://www.w3.org/2000/svg" version="1.1">
<metadata>
<rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<cc:Work>
<dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
<dc:date>2026-08-26T22:54:26.050814</dc:date>
<dc:format>image/svg+xml</dc:format>
<dc:creator>
<cc:Agent>
<dc:title>Matplotlib v3.10.8, https://matplotlib.org/</dc:title>
</cc:Agent>
</dc:creator>
</cc:Work>
</rdf:RDF>
</metadata>
<defs>
<style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
</defs>
<g id="figure_1">
<g id="patch_1">
<path d="M 0 511.157969 L 640.8 511.157969 L 640.8 0 L 0 0 z " style="fill: #ffffff"/>
</g>
<g id="axes_1">
<g id="patch_2">
<path d="M 47.061818 451.016004 L 212.203636 451.016004 Q 215.050909 451.016004 215.050909 448.898326 L 215.050909 417.133147 Q 215.050909 415.015469 212.203636 415.015469 L 47.061818 415.015469 Q 44.214545 415.015469 44.214545 417.133147 L 44.214545 448.898326 Q 44.214545 451.016004 47.061818 451.016004 z " clip-path="url(#p726c9cd962)" style="fill: #fbe3c8; stroke: #e08a2c; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_3">
<path d="M 47.061818 407.391826 L 212.203636 407.391826 Q 215.050909 407.391826 215.050909 405.274147 L 215.050909 373.508969 Q 215.050909 371.39129 212.203636 371.39129 L 47.061818 371.39129 Q 44.214545 371.39129 44.214545 373.508969 L 44.214545 405.274147 Q 44.214545 407.391826 47.061818 407.391826 z " clip-path="url(#p726c9cd962)" style="fill: #cfe0f0; stroke: #3b6ea5; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_4">
<path d="M 47.061818 363.767647 L 212.203636 363.767647 Q 215.050909 363.767647 215.050909 361.649969 L 215.050909 329.88479 Q 215.050909 327.767112 212.203636 327.767112 L 47.061818 327.767112 Q 44.214545 327.767112 44.214545 329.88479 L 44.214545 361.649969 Q 44.214545 363.767647 47.061818 363.767647 z " clip-path="url(#p726c9cd962)" style="fill: #eaeaea; stroke: #888888; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_5">
<path d="M 47.061818 320.143469 L 212.203636 320.143469 Q 215.050909 320.143469 215.050909 318.02579 L 215.050909 286.260612 Q 215.050909 284.142933 212.203636 284.142933 L 47.061818 284.142933 Q 44.214545 284.142933 44.214545 286.260612 L 44.214545 318.02579 Q 44.214545 320.143469 47.061818 320.143469 z " clip-path="url(#p726c9cd962)" style="fill: #d9ecdf; stroke: #4c9a6f; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_6">
<path d="M 47.061818 276.51929 L 212.203636 276.51929 Q 215.050909 276.51929 215.050909 274.401612 L 215.050909 242.636433 Q 215.050909 240.518754 212.203636 240.518754 L 47.061818 240.518754 Q 44.214545 240.518754 44.214545 242.636433 L 44.214545 274.401612 Q 44.214545 276.51929 47.061818 276.51929 z " clip-path="url(#p726c9cd962)" style="fill: #eaeaea; stroke: #888888; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_7">
<path d="M 38.52 410.356576 L 220.745455 410.356576 L 220.745455 237.554004 L 38.52 237.554004 L 38.52 410.356576 z " clip-path="url(#p726c9cd962)" style="fill: none; stroke-dasharray: 4.44,1.92; stroke-dashoffset: 0; stroke: #999999; stroke-width: 1.2; stroke-linejoin: miter"/>
</g>
<g id="patch_8">
<path d="M 371.650909 451.016004 L 536.792727 451.016004 Q 539.64 451.016004 539.64 448.898326 L 539.64 417.133147 Q 539.64 415.015469 536.792727 415.015469 L 371.650909 415.015469 Q 368.803636 415.015469 368.803636 417.133147 L 368.803636 448.898326 Q 368.803636 451.016004 371.650909 451.016004 z " clip-path="url(#p726c9cd962)" style="fill: #fbe3c8; stroke: #e08a2c; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_9">
<path d="M 371.650909 407.391826 L 536.792727 407.391826 Q 539.64 407.391826 539.64 405.274147 L 539.64 373.508969 Q 539.64 371.39129 536.792727 371.39129 L 371.650909 371.39129 Q 368.803636 371.39129 368.803636 373.508969 L 368.803636 405.274147 Q 368.803636 407.391826 371.650909 407.391826 z " clip-path="url(#p726c9cd962)" style="fill: #f9d8d6; stroke: #c0504d; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_10">
<path d="M 371.650909 363.767647 L 536.792727 363.767647 Q 539.64 363.767647 539.64 361.649969 L 539.64 329.88479 Q 539.64 327.767112 536.792727 327.767112 L 371.650909 327.767112 Q 368.803636 327.767112 368.803636 329.88479 L 368.803636 361.649969 Q 368.803636 363.767647 371.650909 363.767647 z " clip-path="url(#p726c9cd962)" style="fill: #eaeaea; stroke: #888888; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_11">
<path d="M 371.650909 320.143469 L 536.792727 320.143469 Q 539.64 320.143469 539.64 318.02579 L 539.64 286.260612 Q 539.64 284.142933 536.792727 284.142933 L 371.650909 284.142933 Q 368.803636 284.142933 368.803636 286.260612 L 368.803636 318.02579 Q 368.803636 320.143469 371.650909 320.143469 z " clip-path="url(#p726c9cd962)" style="fill: #e9dff2; stroke: #7b5ea3; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_12">
<path d="M 371.650909 276.51929 L 536.792727 276.51929 Q 539.64 276.51929 539.64 274.401612 L 539.64 242.636433 Q 539.64 240.518754 536.792727 240.518754 L 371.650909 240.518754 Q 368.803636 240.518754 368.803636 242.636433 L 368.803636 274.401612 Q 368.803636 276.51929 371.650909 276.51929 z " clip-path="url(#p726c9cd962)" style="fill: #eaeaea; stroke: #888888; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_13">
<path d="M 371.650909 232.895112 L 536.792727 232.895112 Q 539.64 232.895112 539.64 230.777433 L 539.64 199.012254 Q 539.64 196.894576 536.792727 196.894576 L 371.650909 196.894576 Q 368.803636 196.894576 368.803636 199.012254 L 368.803636 230.777433 Q 368.803636 232.895112 371.650909 232.895112 z " clip-path="url(#p726c9cd962)" style="fill: #d9ecdf; stroke: #4c9a6f; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_14">
<path d="M 371.650909 189.270933 L 536.792727 189.270933 Q 539.64 189.270933 539.64 187.153254 L 539.64 155.388076 Q 539.64 153.270397 536.792727 153.270397 L 371.650909 153.270397 Q 368.803636 153.270397 368.803636 155.388076 L 368.803636 187.153254 Q 368.803636 189.270933 371.650909 189.270933 z " clip-path="url(#p726c9cd962)" style="fill: #eaeaea; stroke: #888888; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="patch_15">
<path d="M 363.109091 410.356576 L 545.334545 410.356576 L 545.334545 150.305647 L 363.109091 150.305647 L 363.109091 410.356576 z " clip-path="url(#p726c9cd962)" style="fill: none; stroke-dasharray: 4.44,1.92; stroke-dashoffset: 0; stroke: #999999; stroke-width: 1.2; stroke-linejoin: miter"/>
</g>
<g id="patch_16">
<path d="M 371.650909 103.716719 L 536.792727 103.716719 Q 539.64 103.716719 539.64 101.59904 L 539.64 76.186897 Q 539.64 74.069219 536.792727 74.069219 L 371.650909 74.069219 Q 368.803636 74.069219 368.803636 76.186897 L 368.803636 101.59904 Q 368.803636 103.716719 371.650909 103.716719 z " clip-path="url(#p726c9cd962)" style="fill: #fde9c8; stroke: #c99a2e; stroke-width: 1.4; stroke-linejoin: miter"/>
</g>
<g id="line2d_1">
<path d="M 243.523636 246.87179 L 243.523636 302.143201 " clip-path="url(#p726c9cd962)" style="fill: none; stroke: #7b5ea3; stroke-width: 1.6; stroke-linecap: square"/>
</g>
<g id="line2d_2">
<path d="M 243.523636 246.87179 L 129.632727 246.87179 " clip-path="url(#p726c9cd962)" style="fill: none"/>
</g>
<g id="text_1">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="435.361205" transform="rotate(-0 129.632727 435.361205)">Embedding + Positional Encoding</text>
</g>
<g id="text_2">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="391.874996" transform="rotate(-0 129.632727 391.874996)">Multi-Head Self-Attention</text>
</g>
<g id="patch_17">
<path d="M 129.632727 415.13252 Q 129.632727 411.202168 129.632727 408.613456 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 412.613456 L 129.632727 408.613456 L 131.632727 412.613456 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_3">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="347.974879" transform="rotate(-0 129.632727 347.974879)">Add &amp; Norm</text>
</g>
<g id="patch_18">
<path d="M 129.632727 371.508342 Q 129.632727 367.577989 129.632727 364.989278 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 368.989278 L 129.632727 364.989278 L 131.632727 368.989278 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_4">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="304.626638" transform="rotate(-0 129.632727 304.626638)">Feed Forward</text>
</g>
<g id="patch_19">
<path d="M 129.632727 327.884163 Q 129.632727 323.953811 129.632727 321.365099 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 325.365099 L 129.632727 321.365099 L 131.632727 325.365099 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_5">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="260.726522" transform="rotate(-0 129.632727 260.726522)">Add &amp; Norm</text>
</g>
<g id="patch_20">
<path d="M 129.632727 284.259984 Q 129.632727 280.329632 129.632727 277.740921 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 281.740921 L 129.632727 277.740921 L 131.632727 281.740921 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_6">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; fill: #555555" transform="translate(234.214233 316.584196) rotate(-270)">×6</text>
</g>
<g id="text_7">
<text style="font-weight: 700; font-size: 12px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #3b6ea5" x="129.632727" y="222.730254" transform="rotate(-0 129.632727 222.730254)">ENCODEUR</text>
</g>
<g id="patch_21">
<path d="M 129.632727 474.428606 Q 129.632727 462.662301 129.632727 452.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 456.237636 L 129.632727 452.237636 L 131.632727 456.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_8">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="489.134219" transform="rotate(-0 129.632727 489.134219)">Séquence source (ex. anglais)</text>
</g>
<g id="text_9">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="435.361205" transform="rotate(-0 454.221818 435.361205)">Embedding + Positional Encoding</text>
</g>
<g id="text_10">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="391.599058" transform="rotate(-0 454.221818 391.599058)">Masked Multi-Head Self-Attention</text>
</g>
<g id="patch_22">
<path d="M 454.221818 415.13252 Q 454.221818 411.202168 454.221818 408.613456 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 412.613456 L 454.221818 408.613456 L 456.221818 412.613456 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_11">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="347.974879" transform="rotate(-0 454.221818 347.974879)">Add &amp; Norm</text>
</g>
<g id="patch_23">
<path d="M 454.221818 371.508342 Q 454.221818 367.577989 454.221818 364.989278 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 368.989278 L 454.221818 364.989278 L 456.221818 368.989278 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_12">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="304.48867" transform="rotate(-0 454.221818 304.48867)">Encoder-Decoder Attention</text>
</g>
<g id="patch_24">
<path d="M 454.221818 327.884163 Q 454.221818 323.953811 454.221818 321.365099 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 325.365099 L 454.221818 321.365099 L 456.221818 325.365099 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_13">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="260.726522" transform="rotate(-0 454.221818 260.726522)">Add &amp; Norm</text>
</g>
<g id="patch_25">
<path d="M 454.221818 284.259984 Q 454.221818 280.329632 454.221818 277.740921 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 281.740921 L 454.221818 277.740921 L 456.221818 281.740921 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_14">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="217.378281" transform="rotate(-0 454.221818 217.378281)">Feed Forward</text>
</g>
<g id="patch_26">
<path d="M 454.221818 240.635806 Q 454.221818 236.705454 454.221818 234.116742 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 238.116742 L 454.221818 234.116742 L 456.221818 238.116742 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_15">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="173.478165" transform="rotate(-0 454.221818 173.478165)">Add &amp; Norm</text>
</g>
<g id="patch_27">
<path d="M 454.221818 197.011627 Q 454.221818 193.081275 454.221818 190.492564 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 194.492564 L 454.221818 190.492564 L 456.221818 194.492564 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_16">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; fill: #555555" transform="translate(558.803324 272.960018) rotate(-270)">×6</text>
</g>
<g id="patch_28">
<path d="M 454.221818 474.428606 Q 454.221818 462.662301 454.221818 452.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 456.237636 L 454.221818 452.237636 L 456.221818 456.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_17">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(412.31549 479.616078)">Sortie déjà générée</text>
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(425.005724 489.134219)">(shifted right)</text>
</g>
<g id="patch_29">
<path d="M 245.522584 302.143201 Q 304.737942 302.143201 362.164445 302.143201 " style="fill: none; stroke: #7b5ea3; stroke-width: 1.6; stroke-linecap: round"/>
<path d="M 358.164445 300.143201 L 362.164445 302.143201 L 358.164445 304.143201 " style="fill: none; stroke: #7b5ea3; stroke-width: 1.6; stroke-linecap: round"/>
</g>
<g id="text_18">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: start; fill: #7b5ea3" x="252.065455" y="274.507496" transform="rotate(-0 252.065455 274.507496)">K, V</text>
</g>
<g id="text_19">
<text style="font-weight: 700; font-size: 12px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #c0504d" x="454.221818" y="137.599576" transform="rotate(-0 454.221818 137.599576)">DÉCODEUR</text>
</g>
<g id="patch_30">
<path d="M 454.221818 120.777577 Q 454.221818 112.187971 454.221818 104.940006 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 108.940006 L 454.221818 104.940006 L 456.221818 108.940006 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_20">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="88.892969" transform="rotate(-0 454.221818 88.892969)">Linear + Softmax</text>
</g>
<g id="text_21">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="61.363147" transform="rotate(-0 454.221818 61.363147)">Probabilités sur le prochain token</text>
</g>
<g id="text_22">
<text style="font-size: 13px; font-family: 'DejaVu Sans'; text-anchor: middle" x="320.4" y="17.597969" transform="rotate(-0 320.4 17.597969)">Vue d'ensemble simplifiée de l'architecture Transformer</text>
</g>
</g>
</g>
<defs>
<clipPath id="p726c9cd962">
<rect x="7.2" y="29.597969" width="626.4" height="474.36"/>
</clipPath>
</defs>
</svg>
</div>

## Feed-Forward et embeddings : les détails qui comptent

Chaque couche contient aussi un petit réseau feed-forward, appliqué séparément à chaque position :

**FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**

Rôle complémentaire à l'attention : l'attention **communique** entre positions (une moyenne pondérée reste une opération linéaire), le FFN **calcule** au sein d'une position, avec une vraie non-linéarité (ReLU). C'est le seul endroit d'une couche où le modèle peut créer de nouvelles features plutôt que de simplement recombiner celles qui existent déjà.

Autre détail sympa : le papier **partage la même matrice de poids** entre les deux couches d'embedding et la transformation pré-softmax finale, une seule « table de correspondance » réutilisée dans les deux sens (token → vecteur, et vecteur → token).

## Positional Encoding : injecter l'ordre sans récurrence

Problème subtil : sans RNN ni convolution, **rien** dans l'architecture ne connaît l'ordre des tokens. La self-attention traite la séquence comme un ensemble : permuter les mots donnerait exactement les mêmes scores de compatibilité entre les mêmes paires.

La solution : ajouter directement à l'embedding un vecteur de position, construit à partir de sinus et cosinus de fréquences différentes :

**PE(pos, 2i) = sin(pos / 10000^(2i/d_model))**
**PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))**

Pourquoi pas juste le numéro de position (0, 1, 2...) directement ? Deux problèmes : la magnitude explose sans limite (un pos de 499 écraserait un embedding de magnitude ~1), et un simple scalaire ne peut pas remplir un vecteur de 512 dimensions de façon riche. Les sinusoïdes, elles, remplissent chaque dimension à une fréquence différente, et possèdent une propriété utile : PE(pos+k) s'exprime comme une fonction linéaire de PE(pos), ce qui pourrait faciliter l'apprentissage de positions relatives.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="734.4pt" height="316.8pt" viewBox="0 0 734.4 316.8" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:27:53.434438</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.11.1, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 316.8 
L 734.4 316.8 
L 734.4 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 37.726099 281.798901 
L 347.152029 281.798901 
L 347.152029 52.674206 
L 37.726099 52.674206 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#pb03cf739f7)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAa4AAAE+CAYAAADVk/TZAAAkqUlEQVR4nO3daZxddZXu8afmIZUaUkkqc0JCgDAPgZAIGJHhaqMS5KIiU7d2QG0VhG4V0St9VWguTm230jgAakBAQQWlJYBAUGRICBAIZA5kDklVpea5X/D6Wedzd3Gq6p/8vm8f9j77DHVWDv+117/gck0fEAAAiSgc7gsAAOD/B4ULAJAUChcAICkULgBAUgo62ttpzgAAjCwD/TbiFxcAICkULgBAUihcAICkULgAAEkpYHIGACAl/OICACSFwgUASAqFCwCQFAoXACApTM4AcOAIpjEgHfziAgAkhcIFAEgKhQsAkBQKFwAgKUzOAAAkhV9cAICkULgAAEmhcAEAksINyEBquIkWBzh+cQEAkkLhAgAkhcIFAEgKhQsAkBRuQAYAJIVfXACApFC4AABJoXABAJJC4QIAJIXJGdh/MFEC+xM+zxa/uAAASaFwAQCSQuECACSFwgUASAqTMwAASeEXFwAgKRQuAEBSKFwAgKRQuAAASWFyBmLcvY9c+IxgiD8D/OICACSFwgUASAqFCwCQFAoXACApTM4AACSFX1wAgKRQuAAASaFwAQCSQuECACSFyRkjEZMIRhbej5GF92PEKWByBgAAHoULAJAUChcAICkULgBAUpicAQBICr+4AABJoXABAJJC4QIAJIXCBQBICpMzBoM7+GO8Ptnx2llDPaVhv9K/f7x2/OICACSFwgUASAqFCwCQFAoXACApTM4AACSFX1wAgKRQuAAASaFwAQCSQuECACTlwJmccSDcbc9z3G8cENMh9pMpDgfEZ3KEPUd+cQEAkkLhAgAkhcIFAEgKhQsAkBQmZwAAksIvLgBAUihcAICkULgAAEmhcAEAkpLW5IwRdvd2iGvNi6QmSqQ0GSKl13WEXWvBQDpfoSPttQsF18ovLgBAUihcAICkULgAAEmhcAEAksLkDABAUvjFBQBICoULAJAUChcAICkULgBAUkbe5IzhuLN7pN1NPsTXMyzTKEbaVImhfg1G2GduWKY/HOh/6wf681f27x5+cQEAkkLhAgAkhcIFAEgKhQsAkBQmZwDAAWJ/+aWyvzwPAMABgsIFAEgKhQsAkBQKFwAgKcMzOSNfd2/n47zDcKd5XiZZ5GtSxX4ycSIvkyNS+pwPx+Pl4bzDMgUmX4+Zj7/ZYfk+e+f/tvjFBQBICoULAJAUChcAICkULgBAUpicAWBY8K/m/UdRwdA+Hp8dAEBSKFwAgKRQuAAASaFwAQCSkr/JGVnv0B5hd+nn7U78rHfFj7DpICNq4sQI++zkb8rHyLqekfRZHtTncYT9bY2oySIj7G+LX1wAgKRQuAAASaFwAQCSQuECACSFyRlAnvCvwvwY6ikNqSkq2P9fIP62AABJoXABAJJC4QIAJIXCBQBIyuAmZwz1XdiDmvCQ8dh8TQXIPMkj49s1HHfMD/Fj5nyPR9pnIHAgvM/h+5XU88j2Xg309WU6TpLUn+3YgYzHZf4bkPJyrfziAgAkhcIFAEgKhQsAkBQKFwAgKUzOwIjDv6b2n+kQKU1xGGmv+Uh67Ubaa8N3BAAgKRQuAEBSKFwAgKRQuAAAScnf5Iw83PmeczJCPiYc5Otu+qGeGjAc0yjy8H5kniiR47xD/trla8JDHl67nBMe8jHFIetnJ8e1DPVj5m06RsZrjZ//YK7VX08+HpNfXACApFC4AABJoXABAJJC4QIAJIXJGQeI/eVfKCPtDv6RNN1AGp7XJx+vAc8j13lH1rUO9fXsL99nAIADBIULAJAUChcAICkULgBAUoZnckbW6Ri57hbPx0SBfE1byPoaZH198vH8c+WDmYLyDj9e7uvxr0E4/SCamjCYyQf5OG/WCQ8jbaJC1nMO5rz5+AzkkvW9HOrPh6SBvug9yfa33h+ck19cAICkULgAAEmhcAEAkkLhAgAkhckZw2Qk/Ythf5lSkPsx83HObCcdaRMVhmNqQtbHHMxnZ6gfM3q8kfY8BvOdlI/nyeQMAMB+g8IFAEgKhQsAkBQKFwAgKbknZ+RjOkZf7zv/eMoxAaL/nX/Mguicg5jykY/nkX0aR/B4ytPEid6ebMflnJoQ3N3f053pMTNPcYieY65j8zD9YaA3ev5DP+Ujej8GM6UhmsaQ9bzRcf09/u8nupac15P5Wv37Eb42OV7X8DXo9q9BdF4mZwAA9hsULgBAUihcAICkULgAAElhckYOw1HZmfAQHffOnzPXeePj/IH5mkYx1Nea63Ud6ikO+brWfLw+4fSHwvhaC4v8q1AQnLggOG/WcxbmeJOLSosyXU/8mP5a+cUFAEgKhQsAkBQKFwAgKRQuAEBSBjc5I5iqkHU6RkF4p308xSHzlIvgWsPrGczEiawTMKIpBdG0gYxTHMIpFlI8NSG6nqzTMTIe9/b1RM8z2+SIzMdFkzoUTyLoi6YxBFm+JjxknYzQ1x28H9FEhejxck2jCK8n44SHPEyGePt6Mk7r6PNf6f090eSMaApOXCai55KPx+QXFwAgKRQuAEBSKFwAgKRQuAAASTlgJmdkrdAHwuSIkTY1IR9TCobjMbNOhpCk0mDaQNbHzHrOXBMesk5NiI6LJjVkfbycx0aPWeKPy3zO4Li3H9N/SrI/j2KbFZYGWXBcvs5bVFrijwuvBgCAEYbCBQBICoULAJAUChcAICm5J2dE0zGyTqPoi6YfZHy84XjMKOvu8pniCRDhNIbouGhSRXdnpmuJHi/nY+bhOfYH0xZ6O+NpFOF0iCCLJirExwXPIzhOyj7lIuu1xueMJ5JEedYpDtHUiKxTGgZ33mzHxdMv4q/evuC5RMf2Rq9r8JB9Az7s7s9xrRnPGx0XzRXhFxcAICkULgBAUihcAICkULgAAElJanLGYKrsSJrwMJjriacfDP00iqzXEx0Xvc9Zpz/kOjbKioOpCYUZpxsUlcaf5qyTGqLzFmY8Z3F5jqkJGa+nuLzUZtFEhfC4YEpDrmOLyst8Fpy3KDhn1sd7+2A/OaKgtNxmhWUVmY4LszKfDea8Kg1eg6LgfQ6vBgCAEYbCBQBICoULAJAUChcAICm5J2f0+WkEBcH0g3BSRT7OKakgOG/0mNGUi3DiRJQFEyUGdd6ujMcF2WCmUfQFeXRsNKmht9O/H/3BZIjBXGvWyRE9HdE0imyTISSpt9O/J9Gx0bVGkxiynlOKp0pEUxyiaQzRRIXouFwTHqJXPb6ed/5ao3PmOm/W68nHcbmOjY/Ldj384gIAJIXCBQBICoULAJAUChcAICnDMjkja7XMNRlhqCdHZJ0Mkeu8WSc8RFlJxmvN9TyivCyYqBBNnCipiKYmBBMMgsfLdWxxxscsrfLTDeKpCT6TpJJRftpAdN7iUX5qQnRc9HhFlZU2k3JMcagY5Y8r9+ctKPfHxef0mSQpuNaB4iAr8RMewuOKg+NK4mkUfQX+cxc2qARdDfnIJKknuJ72oLmnPWjeiTJ+cQEAkkLhAgAkhcIFAEgKhQsAkJSckzMKev3EhYIeP+FA4XFB1uvPGV2LJA10tmfLujoyHdff0RYc57Ncx0YTIHrb/GvQE2TRNIpookR0zlzHRtMfutuCaR3BNIqsEyXePtafN5qAER0XTZXo7PXXk2vCQ5T35GGqxGAmPORjAka+JjxEovPGn6z9Q762jcpH8xu/uAAASaFwAQCSQuECACSFwgUASEreJmfkYzpG7mkU7/x0iKxZRY4xH9Eki+jYiiL/ypZnnf4wKpj+EJzz7WOjaQz+vOU1fqJAcaWfKFBa7actlARTIySpdLQ/NjpvUVWVzQorq302anSQ+eMkqaDSH1sQPOZAiX8e/aX+9Rko9RMnBkrjyRld/f7z2hl0PHT1+qwjaGzpCs7Z0uUbaSSpNdgWpzlowmkNmnBagnM2tftmotag0UiSmoNj4/P66+kIXp+uDn/Onq54a5to+5qe4DG7g/P2Bq85v7gAAEmhcAEAkkLhAgAkJfcNyN3+BtyCHn/jbkFXq80Kg+PUFdzw29rkj5PU37bPZgPtLZmO62v1x3W3+Gvt3uczSepp969BdGx0A3J3m78ZuGufvwG5JzguujH37WOz3WTcE/1/+GANo6PPr310DuKm3ugxh+Om3nxsv57STbTRv6iz3rQqxevSQ71jQ7ReLUnleVhfryx753dByHVstNZdWuXXyMuqfcYvLgBAUihcAICkULgAAEmhcAEAkjIsNyDnY6EzVx4tWFYVBzf1BueMjhtVGS9mRjf9llX7m3OjrLzO37hbXudvIi2r9TfYltX6G2FzHVs6ptZmRTX1Nius8scVBscVjPLHSVJ/RY3Pyvzz7C/zz7Gtx7c8tAdj3KPjpPhm2OYu3/TSGDTT7A1uMN3d6pt39gaZJO1p9c09e4OsNWgK6g5uzu2MGoJy3IDcHbw+PZ2+Yaqn0zeb9QY7PfR1+3P2Bpkk9QW7VvT3+tduoN/fuBsfN7LadwoK/fcrv7gAAEmhcAEAkkLhAgAkhcIFAEhK7skZnX6qRGGXnypR2OmzgbYmm/U37/HZvr02k6S+4NjuJv88Ovf4rKvJP4+uJr9g27Uv3vK+s9Hn0ZSLOPMLr+3BonVrMIm7LRrFoHiSRdZpFNE5s24FL42syRG5/sU4knYziJqQpPw0MEVTE7I2L+XK89HAVF7vJ/mX1MQ7BIQNTNU+i44bqPQNSgPlUYNS3KQVNTC1Bo1IcXOTz/jFBQBICoULAJAUChcAICkULgBAUgY1OSMf0zEGs0hcU1Jks+rg2OpgikVFMI2iot5vhV45Nt7uvHK8X5itHF/nH3Ncrc3Kxo+3WVH9hEyZqv05Jam/0l9rX5A1d/mF133Blt17232Tyc62eMLDrmBSw7Zm3yyzo8lPMNje5I/bG5yzI5go8Xbun0s0OaKrtdlm3cH2Pb3BZIieDp9J8XSIaFLDcExxCKcxFPuGkKJS/7deVOazknLftFBc4TNJKq303xFlVb5ZoiLYKqQ8+K4bVe2/68bV+EySJtb6fGKtf30mjPbNMuOrfMYvLgBAUihcAICkULgAAEmhcAEAkpJzckZhe6PP2vykCrW8ZaO+PTsyZV27dvnHk9S+w19P+64mm3Xs8YvPbTv9lgXte9pt1tkYNwq0BNtWNPb45oRoykWUZZ1iMRzTKPLR9CPFzT1RU9BQN/1I2Rt/sjb9VE4YY7PScXGDTnH9RJtFjT8DVX7CQ/8on/WU++kPLd05tovp8n9b0ZYwUeNP1qafrXv994cUN/40t/jriRp/8tH0I+Wn8Sdq+uEXFwAgKRQuAEBSKFwAgKRQuAAASRnU5IySYC28osjXxJqSKPML4WNLfSZJdcHd26Ma/IJ29RS/oD16sl+0Hj2twWaV06bYTJKKJ86wWdEEn/WN9o/ZWe4X3/dEC8+tflH2jWa/QCpJm4OpEht2+YXXDbt800tzo1+0bg0WrNuChXBJ6ty322bdLb4JqbvNL0xHC8/5mgwRTX+IJjwUB1McSkf5hofS0f5zJUnl1WNtVhX8TUZZ/Rj/9zpz/Kggi6dRTKvxr8+MYMJDfUVxkPnvpZI2/5kratlpM0nq27bRZj3bfda+xTe4tbzhH7N5s29ua90eT09pCfLGoFnkrWBKTjPbmgAA9hcULgBAUihcAICkULgAAEnJPTkjWEAs3ucXAXuChcXe7Zts1rJpm8+ChUVJatna5LNtLTZr3e4bBfa0+8aFeGHRZ1L2KRdZl/SzTqOIJkpIcTPNmKCZJmq0qR7rF8mrJvrF95qpvslGkqqm+gkQUaNN+ZTpNouabDRmso36qoOtZCS1FPjGhT0d/rO1PVgIfzNotNkQTIGJmmwk6Y3d/u+nZa9/zLZ9vpmmvdk/Zmezn8rT1brXZpLUG0xqiBtt/PdAVoXFfrKKlKPRJtgSpbTSN9qU1/hGmopa34QzqtpvMSJJo+uCrUvqo0Yb/zxmjfNNOPziAgAkhcIFAEgKhQsAkBQKFwAgKcVXVs4J/4OoskUL99HC/Lgyfxf65KpSm9VMjxff62f7KReT5h9qs+pDZtqsdOYRNhuYcLDNOqr8Vg+StD2YVrE2WChfvdsvIL/4RpPNNm3z2w40BYvr+3b7iRKS1NHoG2Y6m/3UgJ52P40iXAgPdtIpbI4Xu4vX+YXg8prRNquo8+cdPc43GNSO881L0yfFW1ocOcUvsB/e4K/10GDLk2OC6TGjOnxDUOEu3zAlST0bXrHZvt61Nmva8abN9m7yb3TjxiabbWuOtxPa2eUnyESTGqJmqqwNU6MG0fg0rsxnUyr957V2uv9cjZntmzPqGqbaTJJqps+wWenMI/2BE2tt1F3rm6n4xQUASAqFCwCQFAoXACApFC4AQFJyTs4o3rPJZgPb/cJrd7Bgu/eV9TZrXOMXghs3xI0CTW/4BoQ3g209dmdcsO3sz7wjTPgvhqpin0ZNLw3Bgu2EYAuJupm1/vFm+zvtJWnMYX6qRNWhviGm5CDf9NI39iCb7Svxi8vbW/37KEmv7/FNKK/s8JNVXgqaXrYFx8VNL75xRZI6g6aXrhY/HSIf0x+KSv1nR4q3SymvGWezyvpJNqse59/numDawsGT4wauY6bV2uywsX5Sw6xoMkSVb4Yoa/INKNrpvwclqXvdSzZrfn2TzRrXBE0va/13aNNm3zC1JZggJEm7u7JNEWoLpgRF+MUFAEgKhQsAkBQKFwAgKRQuAEBSCi7X9HB1rCS4uTtqFJhc4RcsZzT4RdCGo/xibsPc2f5iJNWecLzNig89yWbtY/zkjPWN/k78Z4NtVJat8VsvSNKajX6B/a2tfsG/ZfsGm7Xv2WqznnbfuBLJtTBfFiy+j26YYbP6KX4bkUnBAvq7DvHNIvOn+Tv/JemwYLuUsb3+/SjYtNJmrSuettmu5a/bbMfK7TaTpC0bmmy2KVgojxbJszYTVQfNQpI0ucJPwpkZTMIZd7h/LyccP80fd6Jv7Ck/+hSbSVLPZD/FYUu7/7Jbsd3/Tf5tk//sPL/WTwDZvcU3Q0hS41bfZNH+lv9b7w62dokadKJtVkqr/FQiSaoc67fwqZvsp26MCybEzJ1dbzN+cQEAkkLhAgAkhcIFAEgKhQsAkJSckzNKdrxqs+5VfmH6redW2mzXC5tstvNlP1FgQ6PfQkKStnb4hcdoAka0LUHUnBJtzzI1WLCWpGlT/NYUE471jQsNJ/hpFKOPm2uzwoNPsFlzlV9YfX1P/Jo/v80vMD/5mn8vN27yd/Dv2eIXtFt3brJZR6PfRkSSejv9JItIcblvJqqom2Cz0RNn2WzMpNrwMQ+a4RtNTjvMN8TMneQXuw+t9402Na1+sb9/3XKbSVLLC8/bbGfYoOKng7yxxTdDZJ2CI0k9wbdd9K/4mhKfRo1oM+v8ax41oknS+ONm2GzsicfarPTI+TbrnnC4zTY2d9tsebAtkiQtW+ub0VZt8M0iu7f48zZv3WgzfnEBAJJC4QIAJIXCBQBICoULAJCUnJMzygt9d8JBo/yi5JxJvvlg+ml+K4wpZ86zWdnJ59hMknZV+fM+9YZvIvj9S36KwapXd9lsxzo/xaJl2zqbSfEki+gO9vKgGWDMQUfbbEpwF/r7T/DNGWfPjheQ59T6z0fR6idstuexpTbbtPRlm60PFvRXt/jFZUl6q9tPlYhE28UcEkyGmBU02cw486jwMetPP9NmfXPebbPVTf7P+Y+v+8/yn17w2wltCaY/SNLejX77jY5gmstAv2+LKqn025OMnnSwzSYc7KfgSNKRh4+32bnH+m1WFkzx1zO+xTcRdD3zR5ttWfqMzSRp05832ey1nX77mo1tvkktmp6S9bteyv59P/XsBTYrPel9NuMXFwAgKRQuAEBSKFwAgKRQuAAASck5OaN4zTKbNT3+J5ttfuRFm214xi8ERwvsOzrju+KjCRh1wZ3v0QL77GDrhemnH2az8We8N7gaSUedbqP1Xf5u+6Xr/UL575b7hfA3X/d3tu/d6KejtO1+w2ZSvE1CNHGiquEgm40/2G9fc+icYHH9OL+4LkmnTq+12eQu36DTu/xhm21d+pTNNj++3marN8eTCDa2+b+Dtj7/JxtNeplW6RfYjxhbabPpp/ltKSRp+lkn2qziXb6hqrHe//38LZic8fuX/Xv1/CrfvCNJO9Ztttm+LWts1t3qJ70UFPrvlqiZqm6632JFkiYd7Buj3hM0knxgjm8KOrbBb+2T9bteGvrve35xAQCSQuECACSFwgUASAqFCwCQlJyTM6KmhhPr/ELfER/y22/MuPA8m/We9GGb/WGtH48vSbcu83ewr13uF2XfWvOczaIJF2XVvnFj3GF+AogkHT3XT6tYfKpvXDh9im/cGHjiTputv/NBm734Jz8BZEVTvK1J1CgQTZw4aZxv3Jhzvl+0nnrhR23WctgZNpOk377mG1Ruf8K/Butf8FNQGjf4ReloG5WK+riRpGGO36Jm/sm+WWLxghk2O7G6y2Y9j/3CZq8vedRmkrTiCd/As2qff8xoikO0LdCJ0/zWLXMuOM5mkjTxIx+32a5JvsnkvtV+i54lf/ZNOBtXvmazpk1+QowUNz5VNcyw2aQj/WuwcP40my2e7885p9A/f0nqeOgOm726xDd9vPC8b7R5Nfjs8IsLAJAUChcAICkULgBAUihcAICk5JycUfD0vTbbuOQ+m738gL8L/blGv+C/r9fPv4gaRSTp5DG+WWTOojk2m/4x3xDSM/dcmz24xjeL3PqkX+yXpHXLfSPJnnUr/PVkbBZpOPxkmx0TNIpcETSKSNJpDb4Bo//Ju2y2donf7mHlUv/avNjsPztRo4gUN4vMm1Blszkf9luQTLnwYzZrnr3QZvcHjSKS9POoWWSF/9uKmkX6uv1rFzWLTDzcNy1I0oL5QbNIsOB/fFW7zXoe9c0ir/3SN4u88NQWm0n5aRaZd1CtzeZccLzNJlxwkc0kaVuDP/beV/yEkHuCz87GFX5KTvObPosaRaS4WWTy0f55nB40i3xynt8OhV9cAICkULgAAEmhcAEAkkLhAgAkpfjKSt+0IEkzgq0QTj/FL8p+8J5rbTbvqHNt9o1H1trszv/2i4eS9N1VT9qsYIVfmJ/Y77c1+XCzvyv+mtN848KHtq6ymSQtv2eJzR5bscNm24JR/9H2LO8d/5LNDj/60zbbUBN/Pi6+z08G+MsjfsrHW2t9M0DRIbNsNmWu3y7m4x+Mr/XKoImg+KEf2Oy5G++32U//43M2e6u7z2bH1JTZTJJuXeQnz8y+7iqbvVrnJ4t86Xev2GzFY74haMOy39lMkra8UGezZ5cvtNk/LjrCZpd/wL+us9t9U8eetb7pR5LWtPptNLqD5oyGMt+cMX3hTJtNuORym/1NM2wmSdf93L8nqx57xmbNb662WXmt3/Jkzlnn2uyz5/r3SpI+Pts3xrXccaPNnr3qEZvdvsU3ovGLCwCQFAoXACApFC4AQFIoXACApOScnNFzj19Ye+amP9js4WCqRDQdY26tX9A/7aN+uwtJmnXNP9vsuaLZNrv2937RetVjfsuTps2+ASOaYiFJsxYstNlngkXrSw71Ex5afxG8Vzc/bLNHg0XQjhzTKKJpJad9wk9cmPLZL9rszx3jbHb9/f41X/3E0zaTpJbtvtEmmhwxe8GpNrvmPP+ZPG+Kf+323HqDzSTp2X9/3GZ/3u2bE/oG/GOeNrbSZgs+c4rNxl/xJZtJ0oO7/Wfghvv81h1rnvqLzdp3v2mzaErDoacusJkkfWWRf7/Ormmy2fYf3WSzv/7X32z2lz3+vSoqKLCZJJ05abTNTvqCb1KqudT/bd29yU/A+M5v/N/W+r/6xjdJ6mzykzxqpvqmqSMWnmSz/xs0hPCLCwCQFAoXACApFC4AQFIoXACApOScnDGvzjdLnHvDIpsdcv5XbHbRbc/b7Je/e8BmfSs6bCZJc25vtNmPF/uF0D/M8ouSj9z4M3/c9labRVMsJOn8U/22FmOOfI/NLr7bL3Y/8qBfzG2r940S0z9wjs3+7fJ5NpOkc3pW2uypf/i6zb5989/ZbEypn3Ly44/5BdtZP/+hzSTpa8/4JpTbb3/MZi89cLfN/mWzX5R+63Lf1HHFRVfYTJImPeUnkpQG275EvTRTjx5vs2jCw082xf++/dYt/rXbvtJvQVI73TdKXPo1v9XQTQv99IdNX/yUzSTp7hP8BJk/BlNpPnyUbxg674Fv2qy61jeLXH2Lb+qQpB8t89+FFQ9V22xh1Sab3faxo2323orf2+ze5f59lKTVLX4iyfu2+ykfZ/69n0iyvvxYm/GLCwCQFAoXACApFC4AQFIoXACApOScnLHj//hF23t/6CcVrG/zd2h/cHqNzU6/9bM2W3nIeTaTpCuCO9hfe8RvzVBS6a9n7iLfuPDLS4+3WfmS620mSfd/1S+8Pt/UabNT6/2Ugg/8v/Nttvucf7HZpT/z00FWPuino0hSf69flD3i7A/Y7GeLfdPHrOdut9nSy2+12UM722wmSXNG+4aZ869eaLOqq79ns0vv8ov9j9/7kM069my1mSTNPNW/dt++4mSbndH0V5stW/wtm/3m5d02m1DuF9Al6SMX+wX/Gf/2I5t9+cldNltyh2/4aNz4os0mHusnSkjStVe8y2afbPDNXcsvv9pmdz/qm2Wi6Rj/+0y/LZIkHfej79jslq2+OeOm/1pmsx0vPW6zMQf777PLLjvdZpL0ryfX2mz9Nb4R6Vd3+alFe4NtgfjFBQBICoULAJAUChcAICkULgBAUnJOzvjIsf4u9c9vfsJmZ9y2xma33fVLmzUs8Qvo938j3gbggZL7bPaDlX6xN1p8/tQ102x222s+uy6YYiFJ3Uf6yRGLr7nMZhdMecNmd511jc2e+8QvbPbjS/3i+qjf3mkzSTrren9H/Yu/+5XNLunxC69PXLvYZpOO93f36yG/bYkkzagssVk0OeKi3/gF5D/95Oc2q558qM1u+048OeO9z/kpIN87zjcKPBNsa/L57/rmna73fMFm//DVe20mSTe8+KrNzv61b1z49dmjbPbef/qpzR4ItuG5rH6DzSTpqKN8g9e8G1+32Uu7Zthsxqc+Z7OHvuabRXq/cqnNJOmqgz5ks2Nq/ESjdQ/5LXOub/RbP/37jbfZ7Jbv++kxkjSx1j+Xyz/2QZuN+Y2fENMabH/FLy4AQFIoXACApFC4AABJoXABAJKSc3LGC+//Xza74/HNNoumY5y1zC/4L/qjX3hd+pPbbSZJdTOPsdnt/+q3SZi79Gabffdq3/BRWuibRa665eM2k6Q/He+bAT79Vd/U0LrTL3af92m/QHrHu/zb/OCCy2yWaxrFP55zsM1m3uUbKU6/wTf2vPJH3www691+wfrhYCFckvZd+VGb/ecv/NY2J4/x00oufOz7Nrt6wwSb/fQ7d9hMkkpH+b+fG66/xGYX7/ATYm656D9ttiPY0uPz151hM0na+vc32uz8rz5os52rnrTZ/I9fbLOHL5lls2Wn+62WJOneYELIR4/379f8YPLOGT/xzSnP/mqJzSYc7bcvkqQHvumn9tT+8Cqbff/GP9tsWtCgtPguf86f1vk6IEnXff12m/V1+e2orrjGf5a/NcE32vCLCwCQFAoXACApFC4AQFIoXACApOScnPHFL5xis1lf93e3L/qCX7A94Zc7bPb4qXtsdt3KpTaTpDNanrfZzLqLbDY7mHJRcf6Xbbb+J74B467Jx9lMknZ2+QaVba/7poYjb/ITSe6/xU8kufgk/zwmBYvShTmmURxyvt8m4hP3+oaHqAHjgi982mb/0XW/za6t/7zNJOmTHzrEZu9/xW+Jc8EnvmmzG+/wW9C8dKVv3hm/3C/2S/GUj3OP8tMPDv6en6jQdapv3nn+5376w6r577aZJD36jbk2W7fs2zY77ZGP2OzpJX7Sy4/mXWuzv1voGzckSUFzxkFnHWazm1c02+yZO/21zrvQN5k8cmK8tc2104612fsPH2uzxVv9ti/zLvbvx82/8p+dtbf4x5Ok8uf8ZI3GYErOlSd80mZH3MDkDADAfoLCBQBICoULAJAUChcAICnF32tfHf4H90w53mZ9t7zPZm2v/dZmR317nc3qvvhXm9236mmbSVLdVRfa7DsNfuuOzbf7RevF8tM4qhd8xmYXfN83SkjSD3v94vznpvrXdUkwqWLnT/1d+osu+4bNps/zC/Mv3zLTZpJ08xy/TcQ55ffY7LY3V9hs9if8ViGzmitt9vzul20mSWuCJoMnj1lgs6agweDdj/rrGbXIT9W4aenDNpOk+b++zmZXj/INVU9/aaHN7jrrKzY77Gw/NeHET/rJMpL06PydNvvyKX67lG8cWm+zqY/5bYjCBoOGs20mSWtbf2yzu6aeaLPSH/htiNrW/MFmR9zgp2qM+a1vJpOkB1Y/a7Oqf/KNLd+f4L/rtt7pmyEuaT/cZtWnxI1PF/3QN5t9e59vxPrs5LNsdvcivy0Qv7gAAEmhcAEAkkLhAgAkhcIFAEjK/wBH0XIGYQwLhgAAAABJRU5ErkJggg==" id="imagec676129752" transform="scale(1 -1) translate(0 -228.96)" x="37.44" y="-52.56" width="309.6" height="228.96"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="m9f67ababa1" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m9f67ababa1" x="37.726099" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- 0 -->
      <g transform="translate(34.544849 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-13" d="M 2034 4250 
Q 1547 4250 1301 3770 
Q 1056 3291 1056 2328 
Q 1056 1369 1301 889 
Q 1547 409 2034 409 
Q 2525 409 2770 889 
Q 3016 1369 3016 2328 
Q 3016 3291 2770 3770 
Q 2525 4250 2034 4250 
z
M 2034 4750 
Q 2819 4750 3233 4129 
Q 3647 3509 3647 2328 
Q 3647 1150 3233 529 
Q 2819 -91 2034 -91 
Q 1250 -91 836 529 
Q 422 1150 422 2328 
Q 422 3509 836 4129 
Q 1250 4750 2034 4750 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-13"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#m9f67ababa1" x="89.297088" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- 10 -->
      <g transform="translate(82.934588 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-14" d="M 794 531 
L 1825 531 
L 1825 4091 
L 703 3866 
L 703 4441 
L 1819 4666 
L 2450 4666 
L 2450 531 
L 3481 531 
L 3481 0 
L 794 0 
L 794 531 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#m9f67ababa1" x="140.868076" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- 20 -->
      <g transform="translate(134.505576 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-15" d="M 1228 531 
L 3431 531 
L 3431 0 
L 469 0 
L 469 531 
Q 828 903 1448 1529 
Q 2069 2156 2228 2338 
Q 2531 2678 2651 2914 
Q 2772 3150 2772 3378 
Q 2772 3750 2511 3984 
Q 2250 4219 1831 4219 
Q 1534 4219 1204 4116 
Q 875 4013 500 3803 
L 500 4441 
Q 881 4594 1212 4672 
Q 1544 4750 1819 4750 
Q 2544 4750 2975 4387 
Q 3406 4025 3406 3419 
Q 3406 3131 3298 2873 
Q 3191 2616 2906 2266 
Q 2828 2175 2409 1742 
Q 1991 1309 1228 531 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-15"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#m9f67ababa1" x="192.439064" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- 30 -->
      <g transform="translate(186.076564 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-16" d="M 2597 2516 
Q 3050 2419 3304 2112 
Q 3559 1806 3559 1356 
Q 3559 666 3084 287 
Q 2609 -91 1734 -91 
Q 1441 -91 1130 -33 
Q 819 25 488 141 
L 488 750 
Q 750 597 1062 519 
Q 1375 441 1716 441 
Q 2309 441 2620 675 
Q 2931 909 2931 1356 
Q 2931 1769 2642 2001 
Q 2353 2234 1838 2234 
L 1294 2234 
L 1294 2753 
L 1863 2753 
Q 2328 2753 2575 2939 
Q 2822 3125 2822 3475 
Q 2822 3834 2567 4026 
Q 2313 4219 1838 4219 
Q 1578 4219 1281 4162 
Q 984 4106 628 3988 
L 628 4550 
Q 988 4650 1302 4700 
Q 1616 4750 1894 4750 
Q 2613 4750 3031 4423 
Q 3450 4097 3450 3541 
Q 3450 3153 3228 2886 
Q 3006 2619 2597 2516 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-16"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_5">
      <g>
       <use xlink:href="#m9f67ababa1" x="244.010052" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- 40 -->
      <g transform="translate(237.647552 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-17" d="M 2419 4116 
L 825 1625 
L 2419 1625 
L 2419 4116 
z
M 2253 4666 
L 3047 4666 
L 3047 1625 
L 3713 1625 
L 3713 1100 
L 3047 1100 
L 3047 0 
L 2419 0 
L 2419 1100 
L 313 1100 
L 313 1709 
L 2253 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-17"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_6">
      <g>
       <use xlink:href="#m9f67ababa1" x="295.581041" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- 50 -->
      <g transform="translate(289.218541 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-18" d="M 691 4666 
L 3169 4666 
L 3169 4134 
L 1269 4134 
L 1269 2991 
Q 1406 3038 1543 3061 
Q 1681 3084 1819 3084 
Q 2600 3084 3056 2656 
Q 3513 2228 3513 1497 
Q 3513 744 3044 326 
Q 2575 -91 1722 -91 
Q 1428 -91 1123 -41 
Q 819 9 494 109 
L 494 744 
Q 775 591 1075 516 
Q 1375 441 1709 441 
Q 2250 441 2565 725 
Q 2881 1009 2881 1497 
Q 2881 1984 2565 2268 
Q 2250 2553 1709 2553 
Q 1456 2553 1204 2497 
Q 953 2441 691 2322 
L 691 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-18"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_7">
      <g>
       <use xlink:href="#m9f67ababa1" x="347.152029" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- 60 -->
      <g transform="translate(340.789529 296.396557) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-19" d="M 2113 2584 
Q 1688 2584 1439 2293 
Q 1191 2003 1191 1497 
Q 1191 994 1439 701 
Q 1688 409 2113 409 
Q 2538 409 2786 701 
Q 3034 994 3034 1497 
Q 3034 2003 2786 2293 
Q 2538 2584 2113 2584 
z
M 3366 4563 
L 3366 3988 
Q 3128 4100 2886 4159 
Q 2644 4219 2406 4219 
Q 1781 4219 1451 3797 
Q 1122 3375 1075 2522 
Q 1259 2794 1537 2939 
Q 1816 3084 2150 3084 
Q 2853 3084 3261 2657 
Q 3669 2231 3669 1497 
Q 3669 778 3244 343 
Q 2819 -91 2113 -91 
Q 1303 -91 875 529 
Q 447 1150 447 2328 
Q 447 3434 972 4092 
Q 1497 4750 2381 4750 
Q 2619 4750 2861 4703 
Q 3103 4656 3366 4563 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-19"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="text_8">
     <!-- position pos -->
     <g transform="translate(159.18383 311.157182) scale(0.11 -0.11)">
      <defs>
       <path id="DejaVuSans-53" d="M 1159 525 
L 1159 -1331 
L 581 -1331 
L 581 3500 
L 1159 3500 
L 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
z
M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-52" d="M 1959 3097 
Q 1497 3097 1228 2736 
Q 959 2375 959 1747 
Q 959 1119 1226 758 
Q 1494 397 1959 397 
Q 2419 397 2687 759 
Q 2956 1122 2956 1747 
Q 2956 2369 2687 2733 
Q 2419 3097 1959 3097 
z
M 1959 3584 
Q 2709 3584 3137 3096 
Q 3566 2609 3566 1747 
Q 3566 888 3137 398 
Q 2709 -91 1959 -91 
Q 1206 -91 779 398 
Q 353 888 353 1747 
Q 353 2609 779 3096 
Q 1206 3584 1959 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-56" d="M 2834 3397 
L 2834 2853 
Q 2591 2978 2328 3040 
Q 2066 3103 1784 3103 
Q 1356 3103 1142 2972 
Q 928 2841 928 2578 
Q 928 2378 1081 2264 
Q 1234 2150 1697 2047 
L 1894 2003 
Q 2506 1872 2764 1633 
Q 3022 1394 3022 966 
Q 3022 478 2636 193 
Q 2250 -91 1575 -91 
Q 1294 -91 989 -36 
Q 684 19 347 128 
L 347 722 
Q 666 556 975 473 
Q 1284 391 1588 391 
Q 1994 391 2212 530 
Q 2431 669 2431 922 
Q 2431 1156 2273 1281 
Q 2116 1406 1581 1522 
L 1381 1569 
Q 847 1681 609 1914 
Q 372 2147 372 2553 
Q 372 3047 722 3315 
Q 1072 3584 1716 3584 
Q 2034 3584 2315 3537 
Q 2597 3491 2834 3397 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-4c" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-57" d="M 1172 4494 
L 1172 3500 
L 2356 3500 
L 2356 3053 
L 1172 3053 
L 1172 1153 
Q 1172 725 1289 603 
Q 1406 481 1766 481 
L 2356 481 
L 2356 0 
L 1766 0 
Q 1100 0 847 248 
Q 594 497 594 1153 
L 594 3053 
L 172 3053 
L 172 3500 
L 594 3500 
L 594 4494 
L 1172 4494 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-51" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-3" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-53"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(124.671875 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(176.765625 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(204.546875 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(243.75 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(271.53125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(332.71875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(396.09375 0)"/>
      <use xlink:href="#DejaVuSans-53" transform="translate(427.875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(491.359375 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(552.546875 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_8">
      <defs>
       <path id="mbe5dce66a2" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="52.674206" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- 0 -->
      <g transform="translate(24.363599 56.473034) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_9">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="88.47494" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- 10 -->
      <g transform="translate(18.001099 92.273768) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_10">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="124.275673" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- 20 -->
      <g transform="translate(18.001099 128.074501) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-15"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_11">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="160.076407" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_12">
      <!-- 30 -->
      <g transform="translate(18.001099 163.875235) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-16"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_12">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="195.87714" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_13">
      <!-- 40 -->
      <g transform="translate(18.001099 199.675968) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-17"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_13">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="231.677874" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_14">
      <!-- 50 -->
      <g transform="translate(18.001099 235.476702) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-18"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_14">
      <g>
       <use xlink:href="#mbe5dce66a2" x="37.726099" y="267.478607" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_15">
      <!-- 60 -->
      <g transform="translate(18.001099 271.277435) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-19"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="text_16">
     <!-- dimension i -->
     <g transform="translate(11.358521 199.00335) rotate(-90) scale(0.11 -0.11)">
      <defs>
       <path id="DejaVuSans-47" d="M 2906 2969 
L 2906 4863 
L 3481 4863 
L 3481 0 
L 2906 0 
L 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
z
M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-50" d="M 3328 2828 
Q 3544 3216 3844 3400 
Q 4144 3584 4550 3584 
Q 5097 3584 5394 3201 
Q 5691 2819 5691 2113 
L 5691 0 
L 5113 0 
L 5113 2094 
Q 5113 2597 4934 2840 
Q 4756 3084 4391 3084 
Q 3944 3084 3684 2787 
Q 3425 2491 3425 1978 
L 3425 0 
L 2847 0 
L 2847 2094 
Q 2847 2600 2669 2842 
Q 2491 3084 2119 3084 
Q 1678 3084 1418 2786 
Q 1159 2488 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1356 3278 1631 3431 
Q 1906 3584 2284 3584 
Q 2666 3584 2933 3390 
Q 3200 3197 3328 2828 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-48" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(250.203125 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(313.578125 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(365.671875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(393.453125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(454.640625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(518.015625 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(549.796875 0)"/>
     </g>
    </g>
   </g>
   <g id="patch_3">
    <path d="M 37.726099 281.798901 
L 37.726099 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 347.152029 281.798901 
L 347.152029 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 37.726099 281.798901 
L 347.152029 281.798901 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 37.726099 52.674206 
L 347.152029 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_17">
    <!-- Encodage positionnel complet -->
    <g transform="translate(37.726099 32.411062) scale(0.115 -0.115)">
     <defs>
      <path id="DejaVuSans-28" d="M 628 4666 
L 3578 4666 
L 3578 4134 
L 1259 4134 
L 1259 2753 
L 3481 2753 
L 3481 2222 
L 1259 2222 
L 1259 531 
L 3634 531 
L 3634 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-46" d="M 3122 3366 
L 3122 2828 
Q 2878 2963 2633 3030 
Q 2388 3097 2138 3097 
Q 1578 3097 1268 2742 
Q 959 2388 959 1747 
Q 959 1106 1268 751 
Q 1578 397 2138 397 
Q 2388 397 2633 464 
Q 2878 531 3122 666 
L 3122 134 
Q 2881 22 2623 -34 
Q 2366 -91 2075 -91 
Q 1284 -91 818 406 
Q 353 903 353 1747 
Q 353 2603 823 3093 
Q 1294 3584 2113 3584 
Q 2378 3584 2631 3529 
Q 2884 3475 3122 3366 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-44" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-4a" d="M 2906 1791 
Q 2906 2416 2648 2759 
Q 2391 3103 1925 3103 
Q 1463 3103 1205 2759 
Q 947 2416 947 1791 
Q 947 1169 1205 825 
Q 1463 481 1925 481 
Q 2391 481 2648 825 
Q 2906 1169 2906 1791 
z
M 3481 434 
Q 3481 -459 3084 -895 
Q 2688 -1331 1869 -1331 
Q 1566 -1331 1297 -1286 
Q 1028 -1241 775 -1147 
L 775 -588 
Q 1028 -725 1275 -790 
Q 1522 -856 1778 -856 
Q 2344 -856 2625 -561 
Q 2906 -266 2906 331 
L 2906 616 
Q 2728 306 2450 153 
Q 2172 0 1784 0 
Q 1141 0 747 490 
Q 353 981 353 1791 
Q 353 2603 747 3093 
Q 1141 3584 1784 3584 
Q 2172 3584 2450 3431 
Q 2728 3278 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 434 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-28"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(63.1875 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(126.5625 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(181.546875 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(242.734375 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(306.21875 0)"/>
     <use xlink:href="#DejaVuSans-4a" transform="translate(367.5 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(430.984375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(492.515625 0)"/>
     <use xlink:href="#DejaVuSans-53" transform="translate(524.296875 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(587.78125 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(648.96875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(701.0625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(728.84375 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(768.046875 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(795.828125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(857.015625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(920.390625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(983.765625 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(1045.296875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1073.078125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(1104.859375 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1159.84375 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(1221.03125 0)"/>
     <use xlink:href="#DejaVuSans-53" transform="translate(1318.4375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(1381.921875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1409.703125 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(1471.234375 0)"/>
    </g>
    <!-- (d_model=64, positions 0 à 59) -->
    <g transform="translate(37.726099 46.674206) scale(0.115 -0.115)">
     <defs>
      <path id="DejaVuSans-b" d="M 1984 4856 
Q 1566 4138 1362 3434 
Q 1159 2731 1159 2009 
Q 1159 1288 1364 580 
Q 1569 -128 1984 -844 
L 1484 -844 
Q 1016 -109 783 600 
Q 550 1309 550 2009 
Q 550 2706 781 3412 
Q 1013 4119 1484 4856 
L 1984 4856 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-42" d="M 3263 -1063 
L 3263 -1509 
L -63 -1509 
L -63 -1063 
L 3263 -1063 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-20" d="M 678 2906 
L 4684 2906 
L 4684 2381 
L 678 2381 
L 678 2906 
z
M 678 1631 
L 4684 1631 
L 4684 1100 
L 678 1100 
L 678 1631 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-f" d="M 750 794 
L 1409 794 
L 1409 256 
L 897 -744 
L 494 -744 
L 750 256 
L 750 794 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-a2" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
M 1403 5119 
L 2284 3950 
L 1806 3950 
L 787 5119 
L 1403 5119 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-1c" d="M 703 97 
L 703 672 
Q 941 559 1184 500 
Q 1428 441 1663 441 
Q 2288 441 2617 861 
Q 2947 1281 2994 2138 
Q 2813 1869 2534 1725 
Q 2256 1581 1919 1581 
Q 1219 1581 811 2004 
Q 403 2428 403 3163 
Q 403 3881 828 4315 
Q 1253 4750 1959 4750 
Q 2769 4750 3195 4129 
Q 3622 3509 3622 2328 
Q 3622 1225 3098 567 
Q 2575 -91 1691 -91 
Q 1453 -91 1209 -44 
Q 966 3 703 97 
z
M 1959 2075 
Q 2384 2075 2632 2365 
Q 2881 2656 2881 3163 
Q 2881 3666 2632 3958 
Q 2384 4250 1959 4250 
Q 1534 4250 1286 3958 
Q 1038 3666 1038 3163 
Q 1038 2656 1286 2365 
Q 1534 2075 1959 2075 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-c" d="M 513 4856 
L 1013 4856 
Q 1481 4119 1714 3412 
Q 1947 2706 1947 2009 
Q 1947 1309 1714 600 
Q 1481 -109 1013 -844 
L 513 -844 
Q 928 -128 1133 580 
Q 1338 1288 1338 2009 
Q 1338 2731 1133 3434 
Q 928 4138 513 4856 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-b"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(39.015625 0)"/>
     <use xlink:href="#DejaVuSans-42" transform="translate(102.5 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(152.5 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(249.90625 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(311.09375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(374.578125 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(436.109375 0)"/>
     <use xlink:href="#DejaVuSans-20" transform="translate(463.890625 0)"/>
     <use xlink:href="#DejaVuSans-19" transform="translate(547.6875 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(611.3125 0)"/>
     <use xlink:href="#DejaVuSans-f" transform="translate(674.9375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(706.71875 0)"/>
     <use xlink:href="#DejaVuSans-53" transform="translate(738.5 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(801.984375 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(863.171875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(915.265625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(943.046875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(982.25 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1010.03125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1071.21875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1134.59375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1186.6875 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(1218.46875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1282.09375 0)"/>
     <use xlink:href="#DejaVuSans-a2" transform="translate(1313.875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1375.15625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(1406.9375 0)"/>
     <use xlink:href="#DejaVuSans-1c" transform="translate(1470.5625 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1534.1875 0)"/>
    </g>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_7">
    <path d="M 359.515009 281.798901 
L 368.797787 281.798901 
L 368.797787 52.674206 
L 359.515009 52.674206 
z
" style="fill: #ffffff"/>
   </g>
   <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAA0AAAE+CAYAAABWcprLAAABnElEQVR4nO2aQQ7DIBADF0L/0y/1/28I9EJ734lkWWi5W85ObRJQ2+v9WZFco/UrqxGKelpBnUa7gKifCKLnzSoRz5xkIGSBNRdVjPYSgui9ESciankR7ZNsJiLqMhDCmZhTWkNBAOTmMYLVICC8YyRMBAmsLkZHVuNSiRg92eMx0TgSBDgKHZkI0lxzECSwtUfIRQViLwgCvXNR3fMa5VdYgfg56WZSvTWE51x7EGmNf5/QUUiYiAIRETGAUYwe6H6POMmqUSC2EzCi90YMeX4pbz7ITAXi55TXaPcIcxCiEkJ6K/1PhxixJnECIuLEQESB4E5QNEX0pCCcY2SfCHMQJLCViEeieedFC/xONBHg8dhMN3KyBrEYPRAj6JQXCTcWmAhrEKS5yhJ696k2li1CdUfZW7cKBHHSzSQEMUlgTwSBZpqqmdxByGYa031jqT5F4GoIQeQPklIQ4Jw7wUzSajjPNCZALgUhSnlV4+90YjXYZ451NVjdzatBLjpvJkprhE6MXoF44GROr0DsBTcWIjoRRCXi75QXfQG/mU4OB1g+/wAAAABJRU5ErkJggg==" id="imaged301aecfdf" transform="scale(1 -1) translate(0 -228.96)" x="359.28" y="-52.56" width="9.36" height="228.96"/>
   <g id="matplotlib.axis_3"/>
   <g id="matplotlib.axis_4">
    <g id="ytick_8">
     <g id="line2d_15">
      <defs>
       <path id="m323ed5a368" d="M 0 0 
L 3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_18">
      <!-- −1.00 -->
      <g transform="translate(375.797787 285.217846) scale(0.09 -0.09)">
       <defs>
        <path id="DejaVuSans-c9c" d="M 678 2272 
L 4684 2272 
L 4684 1741 
L 678 1741 
L 678 2272 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-11" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-14" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_9">
     <g id="line2d_16">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="253.158314" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_19">
      <!-- −0.75 -->
      <g transform="translate(375.797787 256.577259) scale(0.09 -0.09)">
       <defs>
        <path id="DejaVuSans-1a" d="M 525 4666 
L 3525 4666 
L 3525 4397 
L 1831 0 
L 1172 0 
L 2766 4134 
L 525 4134 
L 525 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-1a" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_10">
     <g id="line2d_17">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="224.517727" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_20">
      <!-- −0.50 -->
      <g transform="translate(375.797787 227.936672) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_11">
     <g id="line2d_18">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="195.87714" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_21">
      <!-- −0.25 -->
      <g transform="translate(375.797787 199.296086) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-15" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_12">
     <g id="line2d_19">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="167.236553" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_22">
      <!-- 0.00 -->
      <g transform="translate(375.797787 170.655499) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_13">
     <g id="line2d_20">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="138.595967" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_23">
      <!-- 0.25 -->
      <g transform="translate(375.797787 142.014912) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_14">
     <g id="line2d_21">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="109.95538" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_24">
      <!-- 0.50 -->
      <g transform="translate(375.797787 113.374325) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_15">
     <g id="line2d_22">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="81.314793" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_25">
      <!-- 0.75 -->
      <g transform="translate(375.797787 84.733738) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-1a" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_16">
     <g id="line2d_23">
      <g>
       <use xlink:href="#m323ed5a368" x="368.797787" y="52.674206" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_26">
      <!-- 1.00 -->
      <g transform="translate(375.797787 56.093152) scale(0.09 -0.09)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
   </g>
   <g id="LineCollection_1"/>
   <g id="patch_8">
    <path d="M 359.515009 281.798901 
L 364.156398 281.798901 
L 368.797787 281.798901 
L 368.797787 52.674206 
L 364.156398 52.674206 
L 359.515009 52.674206 
L 359.515009 281.798901 
z
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
  </g>
  <g id="axes_3">
   <g id="patch_9">
    <path d="M 462.02522 281.798901 
L 725.03726 281.798901 
L 725.03726 52.674206 
L 462.02522 52.674206 
z
" style="fill: #ffffff"/>
   </g>
   <g id="matplotlib.axis_5">
    <g id="xtick_8">
     <g id="line2d_24">
      <g>
       <use xlink:href="#m9f67ababa1" x="462.02522" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_27">
      <!-- 0 -->
      <g transform="translate(458.84397 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
      </g>
     </g>
    </g>
    <g id="xtick_9">
     <g id="line2d_25">
      <g>
       <use xlink:href="#m9f67ababa1" x="505.86056" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_28">
      <!-- 10 -->
      <g transform="translate(499.49806 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_10">
     <g id="line2d_26">
      <g>
       <use xlink:href="#m9f67ababa1" x="549.6959" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_29">
      <!-- 20 -->
      <g transform="translate(543.3334 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-15"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_11">
     <g id="line2d_27">
      <g>
       <use xlink:href="#m9f67ababa1" x="593.53124" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_30">
      <!-- 30 -->
      <g transform="translate(587.16874 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-16"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_12">
     <g id="line2d_28">
      <g>
       <use xlink:href="#m9f67ababa1" x="637.36658" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_31">
      <!-- 40 -->
      <g transform="translate(631.00408 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-17"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_13">
     <g id="line2d_29">
      <g>
       <use xlink:href="#m9f67ababa1" x="681.20192" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_32">
      <!-- 50 -->
      <g transform="translate(674.83942 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-18"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_14">
     <g id="line2d_30">
      <g>
       <use xlink:href="#m9f67ababa1" x="725.03726" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_33">
      <!-- 60 -->
      <g transform="translate(718.67476 296.396557) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-19"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="text_34">
     <!-- position pos -->
     <g transform="translate(560.276006 311.157182) scale(0.11 -0.11)">
      <use xlink:href="#DejaVuSans-53"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(124.671875 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(176.765625 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(204.546875 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(243.75 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(271.53125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(332.71875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(396.09375 0)"/>
      <use xlink:href="#DejaVuSans-53" transform="translate(427.875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(491.359375 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(552.546875 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_6">
    <g id="ytick_17">
     <g id="line2d_31">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="271.385162" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_35">
      <!-- −1.00 -->
      <g transform="translate(424.379907 275.18399) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-14" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_18">
     <g id="line2d_32">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="245.348137" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_36">
      <!-- −0.75 -->
      <g transform="translate(424.379907 249.146965) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-1a" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_19">
     <g id="line2d_33">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="219.311113" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_37">
      <!-- −0.50 -->
      <g transform="translate(424.379907 223.109941) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_20">
     <g id="line2d_34">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="193.274088" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_38">
      <!-- −0.25 -->
      <g transform="translate(424.379907 197.072916) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-c9c"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(83.796875 0)"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(147.421875 0)"/>
       <use xlink:href="#DejaVuSans-15" transform="translate(179.203125 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(242.828125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_21">
     <g id="line2d_35">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="167.237063" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_39">
      <!-- 0.00 -->
      <g transform="translate(432.759595 171.035892) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_22">
     <g id="line2d_36">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="141.200039" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_40">
      <!-- 0.25 -->
      <g transform="translate(432.759595 144.998867) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_23">
     <g id="line2d_37">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="115.163014" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_41">
      <!-- 0.50 -->
      <g transform="translate(432.759595 118.961842) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_24">
     <g id="line2d_38">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="89.12599" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_42">
      <!-- 0.75 -->
      <g transform="translate(432.759595 92.924818) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-1a" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_25">
     <g id="line2d_39">
      <g>
       <use xlink:href="#mbe5dce66a2" x="462.02522" y="63.088965" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_43">
      <!-- 1.00 -->
      <g transform="translate(432.759595 66.887793) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="text_44">
     <!-- valeur -->
     <g transform="translate(417.737329 184.521163) rotate(-90) scale(0.11 -0.11)">
      <defs>
       <path id="DejaVuSans-59" d="M 191 3500 
L 800 3500 
L 1894 563 
L 2988 3500 
L 3597 3500 
L 2284 0 
L 1503 0 
L 191 3500 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-58" d="M 544 1381 
L 544 3500 
L 1119 3500 
L 1119 1403 
Q 1119 906 1312 657 
Q 1506 409 1894 409 
Q 2359 409 2629 706 
Q 2900 1003 2900 1516 
L 2900 3500 
L 3475 3500 
L 3475 0 
L 2900 0 
L 2900 538 
Q 2691 219 2414 64 
Q 2138 -91 1772 -91 
Q 1169 -91 856 284 
Q 544 659 544 1381 
z
M 1991 3584 
L 1991 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-55" d="M 2631 2963 
Q 2534 3019 2420 3045 
Q 2306 3072 2169 3072 
Q 1681 3072 1420 2755 
Q 1159 2438 1159 1844 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1341 3275 1631 3429 
Q 1922 3584 2338 3584 
Q 2397 3584 2469 3576 
Q 2541 3569 2628 3553 
L 2631 2963 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-59"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(59.1875 0)"/>
      <use xlink:href="#DejaVuSans-4f" transform="translate(120.46875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(148.25 0)"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(209.78125 0)"/>
      <use xlink:href="#DejaVuSans-55" transform="translate(273.15625 0)"/>
     </g>
    </g>
   </g>
   <g id="line2d_40">
    <path d="M 462.02522 167.237063 
L 466.408754 79.599461 
L 470.792288 72.535466 
L 475.175822 152.539683 
L 479.559356 246.056604 
L 483.94289 267.107203 
L 488.326424 196.337656 
L 492.709958 98.813159 
L 497.093492 64.197284 
L 501.477026 124.315707 
L 505.86056 223.895828 
L 510.244094 271.384142 
L 514.627628 223.120112 
L 519.011162 123.477466 
L 523.394696 64.067191 
L 527.77823 99.510822 
L 532.161764 197.221646 
L 536.545298 267.364784 
L 540.928832 245.450957 
L 545.312366 151.627637 
L 549.6959 72.155552 
L 554.079434 80.10097 
L 558.462968 168.15891 
L 562.846502 255.369309 
L 567.230036 261.551328 
L 571.61357 181.021247 
L 575.997104 87.818051 
L 580.380638 67.632329 
L 584.764172 139.022741 
L 589.147706 236.35327 
L 593.53124 270.138678 
L 597.914774 209.316816 
L 602.298308 109.807023 
L 606.681842 63.098145 
L 611.065376 112.134108 
L 615.44891 211.831474 
L 619.832444 270.528945 
L 624.215978 234.260336 
L 628.599512 136.37084 
L 632.983046 66.859607 
L 637.36658 89.634945 
L 641.750114 183.757313 
L 646.133648 262.69104 
L 650.517182 253.864821 
L 654.900716 165.393442 
L 659.28425 78.61708 
L 663.667784 73.317522 
L 668.051318 154.367158 
L 672.434852 247.249325 
L 676.818386 266.568588 
L 681.20192 194.562906 
L 685.585454 97.433969 
L 689.968988 64.481676 
L 694.352522 126.002212 
L 698.736056 225.43388 
L 703.11959 271.359664 
L 707.503124 221.555608 
L 711.886658 121.811334 
L 716.270192 63.831265 
L 720.653726 100.922011 
" clip-path="url(#pcc53a218ab)" style="fill: none; stroke: #3b6ea5; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_41">
    <path d="M 462.02522 63.088965 
L 466.408754 110.965606 
L 470.792288 210.577965 
L 475.175822 270.342899 
L 479.559356 235.312804 
L 483.94289 137.694186 
L 488.326424 67.237154 
L 492.709958 88.719577 
L 497.093492 182.390615 
L 501.477026 262.129548 
L 505.86056 254.624768 
L 510.244094 166.776135 
L 514.627628 79.351278 
L 519.011162 72.728207 
L 523.394696 152.996142 
L 527.77823 246.357115 
L 532.161764 266.975477 
L 536.545298 195.894802 
L 540.928832 98.466334 
L 545.312366 64.265358 
L 549.6959 124.736093 
L 554.079434 224.282024 
L 558.462968 271.381082 
L 562.846502 222.730609 
L 567.230036 123.059627 
L 571.61357 64.005176 
L 575.997104 99.861646 
L 580.380638 197.662765 
L 584.764172 267.490634 
L 589.147706 245.145833 
L 593.53124 151.172068 
L 597.914774 71.968386 
L 602.298308 80.354287 
L 606.681842 168.619811 
L 611.065376 255.614044 
L 615.44891 261.354888 
L 619.832444 180.564238 
L 624.215978 87.520646 
L 628.599512 67.76796 
L 632.983046 139.466709 
L 637.36658 236.697394 
L 641.750114 270.066572 
L 646.133648 208.894773 
L 650.517182 109.423069 
L 654.900716 63.105284 
L 659.28425 112.525777 
L 663.667784 212.247575 
L 668.051318 270.586915 
L 672.434852 233.906879 
L 676.818386 135.930922 
L 681.20192 66.737687 
L 685.585454 89.943115 
L 689.968988 184.212243 
L 694.352522 262.874469 
L 698.736056 253.608105 
L 703.11959 164.932604 
L 707.503124 78.375812 
L 711.886658 73.517645 
L 716.270192 154.824679 
L 720.653726 247.543602 
" clip-path="url(#pcc53a218ab)" style="fill: none; stroke: #e08a2c; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_42">
    <path d="M 462.02522 167.237063 
L 466.408754 142.77048 
L 470.792288 119.673319 
L 475.175822 99.238352 
L 479.559356 82.609349 
L 483.94289 70.717052 
L 488.326424 64.227086 
L 492.709958 63.502703 
L 497.093492 68.584446 
L 501.477026 79.187885 
L 505.86056 94.719533 
L 510.244094 114.310068 
L 514.627628 136.862985 
L 519.011162 161.115973 
L 523.394696 185.711565 
L 527.77823 209.273118 
L 532.161764 230.481868 
L 536.545298 248.150735 
L 540.928832 261.290776 
L 545.312366 269.166526 
L 549.6959 271.337172 
L 554.079434 267.68122 
L 558.462968 258.403298 
L 562.846502 244.022701 
L 567.230036 225.344328 
L 571.61357 203.413626 
L 575.997104 179.458083 
L 580.380638 154.818515 
L 584.764172 130.874028 
L 589.147706 108.964819 
L 593.53124 90.317173 
L 597.914774 75.974817 
L 602.298308 66.74051 
L 606.681842 63.131106 
L 611.065376 65.348628 
L 615.44891 73.268957 
L 619.832444 86.448785 
L 624.215978 104.150422 
L 628.599512 125.383088 
L 632.983046 148.958367 
L 637.36658 173.556725 
L 641.750114 197.801364 
L 646.133648 220.335286 
L 650.517182 239.897242 
L 654.900716 255.392327 
L 659.28425 265.953265 
L 663.667784 270.988949 
L 668.051318 270.217525 
L 672.434852 263.682171 
L 676.818386 251.748678 
L 681.20192 235.084976 
L 685.585454 214.623753 
L 689.968988 191.510244 
L 694.352522 167.038138 
L 698.736056 142.577166 
L 703.11959 119.496436 
L 707.503124 99.087802 
L 711.886658 82.493557 
L 716.270192 70.642499 
L 720.653726 64.197946 
" clip-path="url(#pcc53a218ab)" style="fill: none; stroke: #4c9a6f; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_43">
    <path d="M 462.02522 63.088965 
L 466.408754 66.003607 
L 470.792288 74.584397 
L 475.175822 88.351059 
L 479.559356 106.533058 
L 483.94289 128.112727 
L 488.326424 151.882228 
L 492.709958 176.511156 
L 497.093492 200.621003 
L 501.477026 222.862314 
L 505.86056 241.990219 
L 510.244094 256.934107 
L 514.627628 266.857552 
L 519.011162 271.205129 
L 523.394696 269.733498 
L 527.77823 262.52503 
L 532.161764 249.983188 
L 536.545298 232.809955 
L 540.928832 211.966534 
L 545.312366 188.619556 
L 549.6959 164.075776 
L 554.079434 139.708937 
L 558.462968 116.882878 
L 562.846502 96.875197 
L 567.230036 80.805748 
L 571.61357 69.573955 
L 575.997104 63.808474 
L 580.380638 63.832005 
L 584.764172 69.643232 
L 589.147706 80.916892 
L 593.53124 97.021988 
L 597.914774 117.057099 
L 602.298308 139.900838 
L 606.681842 164.274616 
L 611.065376 188.814205 
L 615.44891 212.146098 
L 619.832444 232.964382 
L 624.215978 250.103836 
L 628.599512 262.605146 
L 632.983046 269.768598 
L 637.36658 271.193248 
L 641.750114 266.799355 
L 646.133648 256.832851 
L 650.517182 241.851572 
L 654.900716 222.694037 
L 659.28425 200.432513 
L 663.667784 176.313004 
L 668.051318 151.685504 
L 672.434852 127.928443 
L 676.818386 106.371528 
L 681.20192 88.221324 
L 685.585454 74.493718 
L 689.968988 65.95706 
L 694.352522 63.089155 
L 698.736056 66.050524 
L 703.11959 74.675414 
L 707.503124 88.481082 
L 711.886658 106.69481 
L 716.270192 128.297154 
L 720.653726 152.079008 
" clip-path="url(#pcc53a218ab)" style="fill: none; stroke: #c0504d; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_44">
    <path d="M 462.02522 167.237063 
L 725.03726 167.237063 
" clip-path="url(#pcc53a218ab)" style="fill: none; stroke: #cccccc; stroke-width: 0.8; stroke-linecap: square"/>
   </g>
   <g id="patch_10">
    <path d="M 462.02522 281.798901 
L 462.02522 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_11">
    <path d="M 725.03726 281.798901 
L 725.03726 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_12">
    <path d="M 462.02522 281.798901 
L 725.03726 281.798901 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_13">
    <path d="M 462.02522 52.674206 
L 725.03726 52.674206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_45">
    <!-- Quelques dimensions individuelles -->
    <g transform="translate(462.02522 32.411062) scale(0.115 -0.115)">
     <defs>
      <path id="DejaVuSans-34" d="M 2522 4238 
Q 1834 4238 1429 3725 
Q 1025 3213 1025 2328 
Q 1025 1447 1429 934 
Q 1834 422 2522 422 
Q 3209 422 3611 934 
Q 4013 1447 4013 2328 
Q 4013 3213 3611 3725 
Q 3209 4238 2522 4238 
z
M 3406 84 
L 4238 -825 
L 3475 -825 
L 2784 -78 
Q 2681 -84 2626 -87 
Q 2572 -91 2522 -91 
Q 1538 -91 948 567 
Q 359 1225 359 2328 
Q 359 3434 948 4092 
Q 1538 4750 2522 4750 
Q 3503 4750 4090 4092 
Q 4678 3434 4678 2328 
Q 4678 1516 4351 937 
Q 4025 359 3406 84 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-54" d="M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
M 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 -1331 
L 2906 -1331 
L 2906 525 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-34"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(78.71875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(142.09375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(203.625 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(231.40625 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(294.890625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(358.265625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(419.796875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(471.890625 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(503.671875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(567.15625 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(594.9375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(692.34375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(753.875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(817.25 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(869.34375 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(897.125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(958.3125 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1021.6875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1073.78125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1105.5625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1133.34375 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(1196.71875 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1260.203125 0)"/>
     <use xlink:href="#DejaVuSans-59" transform="translate(1287.984375 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1347.171875 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(1374.953125 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(1438.4375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1501.8125 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(1563.34375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(1591.125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1618.90625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1680.4375 0)"/>
    </g>
    <!-- (fréquences différentes) -->
    <g transform="translate(462.02522 46.674206) scale(0.115 -0.115)">
     <defs>
      <path id="DejaVuSans-49" d="M 2375 4863 
L 2375 4384 
L 1825 4384 
Q 1516 4384 1395 4259 
Q 1275 4134 1275 3809 
L 1275 3500 
L 2222 3500 
L 2222 3053 
L 1275 3053 
L 1275 0 
L 697 0 
L 697 3053 
L 147 3053 
L 147 3500 
L 697 3500 
L 697 3744 
Q 697 4328 969 4595 
Q 1241 4863 1831 4863 
L 2375 4863 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-ab" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
M 2468 5119 
L 3090 5119 
L 2072 3944 
L 1593 3944 
L 2468 5119 
z
" transform="scale(0.015625)"/>
      <path id="DejaVuSans-13ae" d="M 4531 4863 
L 4531 4384 
L 3981 4384 
Q 3672 4384 3551 4259 
Q 3431 4134 3431 3809 
L 3431 3500 
L 4378 3500 
L 4378 3053 
L 3431 3053 
L 3431 0 
L 2853 0 
L 2853 3053 
L 1275 3053 
L 1275 0 
L 697 0 
L 697 3053 
L 147 3053 
L 147 3500 
L 697 3500 
L 697 3744 
Q 697 4328 969 4595 
Q 1241 4863 1831 4863 
L 2375 4863 
L 2375 4384 
L 1825 4384 
Q 1516 4384 1394 4259 
Q 1275 4134 1275 3809 
L 1275 3500 
L 2853 3500 
L 2853 3744 
Q 2853 4328 3125 4595 
Q 3397 4863 3988 4863 
L 4531 4863 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-b"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(39.015625 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(74.21875 0)"/>
     <use xlink:href="#DejaVuSans-ab" transform="translate(113.125 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(174.65625 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(238.140625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(301.515625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(363.046875 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(426.421875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(481.40625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(542.9375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(595.03125 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(626.8125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(690.296875 0)"/>
     <use xlink:href="#DejaVuSans-13ae" transform="translate(718.078125 0)"/>
     <use xlink:href="#DejaVuSans-ab" transform="translate(786.96875 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(848.5 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(887.40625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(948.9375 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(1012.3125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1051.515625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1113.046875 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1165.140625 0)"/>
    </g>
   </g>
   <g id="legend_1">
    <g id="patch_14">
     <path d="M 658.643979 113.877019 
L 718.73726 113.877019 
Q 720.53726 113.877019 720.53726 112.077019 
L 720.53726 58.974206 
Q 720.53726 57.174206 718.73726 57.174206 
L 658.643979 57.174206 
Q 656.843979 57.174206 656.843979 58.974206 
L 656.843979 112.077019 
Q 656.843979 113.877019 658.643979 113.877019 
z
" style="fill: #ffffff; opacity: 0.85; stroke: #cccccc; stroke-linejoin: miter"/>
    </g>
    <g id="line2d_45">
     <path d="M 660.443979 64.4628 
L 669.443979 64.4628 
L 678.443979 64.4628 
" style="fill: none; stroke: #3b6ea5; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_46">
     <!-- dim 0 -->
     <g transform="translate(685.643979 67.6128) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-13" transform="translate(220.453125 0)"/>
     </g>
    </g>
    <g id="line2d_46">
     <path d="M 660.443979 77.963503 
L 669.443979 77.963503 
L 678.443979 77.963503 
" style="fill: none; stroke: #e08a2c; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_47">
     <!-- dim 1 -->
     <g transform="translate(685.643979 81.113503) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(220.453125 0)"/>
     </g>
    </g>
    <g id="line2d_47">
     <path d="M 660.443979 91.464206 
L 669.443979 91.464206 
L 678.443979 91.464206 
" style="fill: none; stroke: #4c9a6f; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_48">
     <!-- dim 10 -->
     <g transform="translate(685.643979 94.614206) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(220.453125 0)"/>
      <use xlink:href="#DejaVuSans-13" transform="translate(284.078125 0)"/>
     </g>
    </g>
    <g id="line2d_48">
     <path d="M 660.443979 104.964909 
L 669.443979 104.964909 
L 678.443979 104.964909 
" style="fill: none; stroke: #c0504d; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_49">
     <!-- dim 11 -->
     <g transform="translate(685.643979 108.114909) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(220.453125 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(284.078125 0)"/>
     </g>
    </g>
   </g>
  </g>
  <g id="text_50">
   <!-- Positional Encoding : chaque dimension oscille à sa propre fréquence -->
   <g transform="translate(142.02375 13.398209) scale(0.13 -0.13)">
    <defs>
     <path id="DejaVuSans-33" d="M 1259 4147 
L 1259 2394 
L 2053 2394 
Q 2494 2394 2734 2622 
Q 2975 2850 2975 3272 
Q 2975 3691 2734 3919 
Q 2494 4147 2053 4147 
L 1259 4147 
z
M 628 4666 
L 2053 4666 
Q 2838 4666 3239 4311 
Q 3641 3956 3641 3272 
Q 3641 2581 3239 2228 
Q 2838 1875 2053 1875 
L 1259 1875 
L 1259 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-1d" d="M 750 794 
L 1409 794 
L 1409 0 
L 750 0 
L 750 794 
z
M 750 3309 
L 1409 3309 
L 1409 2516 
L 750 2516 
L 750 3309 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-4b" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-33"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(56.734375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(117.921875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(170.015625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(197.796875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(237 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(264.78125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(325.96875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(389.34375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(450.625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(478.40625 0)"/>
    <use xlink:href="#DejaVuSans-28" transform="translate(510.1875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(573.375 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(636.75 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(691.734375 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(752.921875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(816.40625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(844.1875 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(907.5625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(971.046875 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(1002.828125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1036.515625 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1068.296875 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1123.28125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1186.65625 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(1247.9375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1311.421875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1374.796875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1436.328125 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1468.109375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1531.59375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(1559.375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1656.78125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1718.3125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1781.6875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1833.78125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1861.5625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1922.75 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1986.125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2017.90625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2079.09375 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(2131.1875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2186.171875 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2213.953125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2241.734375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2269.515625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2331.046875 0)"/>
    <use xlink:href="#DejaVuSans-a2" transform="translate(2362.828125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2424.109375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2455.890625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2507.984375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2569.265625 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(2601.046875 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2664.53125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2703.4375 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(2764.625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2828.109375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2867.015625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2928.546875 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(2960.328125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2995.53125 0)"/>
    <use xlink:href="#DejaVuSans-ab" transform="translate(3034.4375 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(3095.96875 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(3159.453125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3222.828125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3284.359375 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(3347.734375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3402.71875 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="pb03cf739f7">
   <rect x="37.726099" y="52.674206" width="309.42593" height="229.124694"/>
  </clipPath>
  <clipPath id="pcc53a218ab">
   <rect x="462.02522" y="52.674206" width="263.01204" height="229.124694"/>
  </clipPath>
 </defs>
</svg>
</div>

## Pourquoi l'attention plutôt que le récurrent ou le convolutionnel ?

Le papier compare les trois approches sur trois critères : complexité par couche, opérations séquentielles nécessaires, et longueur de chemin maximale entre deux positions.

| Type de couche | Complexité | Opér. séquentielles | Chemin max. |
|---|---|---|---|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Récurrent | O(n·d²) | O(n) | O(n) |
| Convolutionnel | O(k·n·d²) | O(1) | O(logₖ n) |

La self-attention connecte **toutes** les positions en une seule opération, contre *n* opérations séquentielles pour un RNN, et plusieurs couches empilées pour un CNN.

Attention toutefois : ce n'est pas gratuit. La complexité par couche de la self-attention est **quadratique** en *n* (à cause de la matrice *n×n*), contre linéaire pour le récurrent. Pour une séquence de 10 000 tokens avec d=512, la self-attention finit par coûter environ **20 fois plus cher** par couche qu'un RNN. C'est précisément pour ça que le papier évoque en perspective une « self-attention restreinte » (limiter chaque position à un voisinage local), l'ancêtre conceptuel de toute une famille de méthodes d'attention efficaces qui viendront plus tard (sparse attention, linear attention...).

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="748.433125pt" height="321.72425pt" viewBox="0 0 748.433125 321.72425" xmlns="http://www.w3.org/2000/svg" version="1.1">
<metadata>
<rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<cc:Work>
<dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
<dc:date>2026-08-26T22:54:26.786516</dc:date>
<dc:format>image/svg+xml</dc:format>
<dc:creator>
<cc:Agent>
<dc:title>Matplotlib v3.10.8, https://matplotlib.org/</dc:title>
</cc:Agent>
</dc:creator>
</cc:Work>
</rdf:RDF>
</metadata>
<defs>
<style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
</defs>
<g id="figure_1">
<g id="patch_1">
<path d="M 0 321.72425 L 748.433125 321.72425 L 748.433125 0 L 0 0 z " style="fill: #ffffff"/>
</g>
<g id="axes_1">
<g id="patch_2">
<path d="M 40.603125 283.768 L 365.691783 283.768 L 365.691783 65.688 L 40.603125 65.688 z " style="fill: #ffffff"/>
</g>
<g id="matplotlib.axis_1">
<g id="xtick_1">
<g id="line2d_1">
<defs>
<path id="m1fdfde7ee8" d="M 0 0 L 0 3.5 " style="stroke: #000000; stroke-width: 0.8"/>
</defs>
<g>
<use xlink:href="#m1fdfde7ee8" x="52.394679" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_1">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="52.394679" y="298.366437" transform="rotate(-0 52.394679 298.366437)">0</text>
</g>
</g>
<g id="xtick_2">
<g id="line2d_2">
<g>
<use xlink:href="#m1fdfde7ee8" x="89.709722" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_2">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="89.709722" y="298.366437" transform="rotate(-0 89.709722 298.366437)">25</text>
</g>
</g>
<g id="xtick_3">
<g id="line2d_3">
<g>
<use xlink:href="#m1fdfde7ee8" x="127.024765" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_3">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="127.024765" y="298.366437" transform="rotate(-0 127.024765 298.366437)">50</text>
</g>
</g>
<g id="xtick_4">
<g id="line2d_4">
<g>
<use xlink:href="#m1fdfde7ee8" x="164.339809" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_4">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="164.339809" y="298.366437" transform="rotate(-0 164.339809 298.366437)">75</text>
</g>
</g>
<g id="xtick_5">
<g id="line2d_5">
<g>
<use xlink:href="#m1fdfde7ee8" x="201.654852" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_5">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="201.654852" y="298.366437" transform="rotate(-0 201.654852 298.366437)">100</text>
</g>
</g>
<g id="xtick_6">
<g id="line2d_6">
<g>
<use xlink:href="#m1fdfde7ee8" x="238.969895" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_6">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="238.969895" y="298.366437" transform="rotate(-0 238.969895 298.366437)">125</text>
</g>
</g>
<g id="xtick_7">
<g id="line2d_7">
<g>
<use xlink:href="#m1fdfde7ee8" x="276.284939" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_7">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="276.284939" y="298.366437" transform="rotate(-0 276.284939 298.366437)">150</text>
</g>
</g>
<g id="xtick_8">
<g id="line2d_8">
<g>
<use xlink:href="#m1fdfde7ee8" x="313.599982" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_8">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="313.599982" y="298.366437" transform="rotate(-0 313.599982 298.366437)">175</text>
</g>
</g>
<g id="xtick_9">
<g id="line2d_9">
<g>
<use xlink:href="#m1fdfde7ee8" x="350.915025" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_9">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="350.915025" y="298.366437" transform="rotate(-0 350.915025 298.366437)">200</text>
</g>
</g>
<g id="text_10">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="203.147454" y="312.444562" transform="rotate(-0 203.147454 312.444562)">longueur de séquence n</text>
</g>
</g>
<g id="matplotlib.axis_2">
<g id="ytick_1">
<g id="line2d_10">
<defs>
<path id="mc1f461f8c3" d="M 0 0 L -3.5 0 " style="stroke: #000000; stroke-width: 0.8"/>
</defs>
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_11">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="287.567219" transform="rotate(-0 33.603125 287.567219)">0</text>
</g>
</g>
<g id="ytick_2">
<g id="line2d_11">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="247.421333" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_12">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="251.220552" transform="rotate(-0 33.603125 251.220552)">10</text>
</g>
</g>
<g id="ytick_3">
<g id="line2d_12">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="211.074667" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_13">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="214.873885" transform="rotate(-0 33.603125 214.873885)">20</text>
</g>
</g>
<g id="ytick_4">
<g id="line2d_13">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="174.728" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_14">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="178.527219" transform="rotate(-0 33.603125 178.527219)">30</text>
</g>
</g>
<g id="ytick_5">
<g id="line2d_14">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="138.381333" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_15">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="142.180552" transform="rotate(-0 33.603125 142.180552)">40</text>
</g>
</g>
<g id="ytick_6">
<g id="line2d_15">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="102.034667" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_16">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="105.833885" transform="rotate(-0 33.603125 105.833885)">50</text>
</g>
</g>
<g id="ytick_7">
<g id="line2d_16">
<g>
<use xlink:href="#mc1f461f8c3" x="40.603125" y="65.688" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_17">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: end" x="33.603125" y="69.487219" transform="rotate(-0 33.603125 69.487219)">60</text>
</g>
</g>
<g id="text_18">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="14.798438" y="174.728" transform="rotate(-90 14.798438 174.728)">longueur de chemin maximale</text>
</g>
</g>
<g id="line2d_17">
<path d="M 55.379882 276.498667 L 169.336685 -1 L 169.336685 -1 " clip-path="url(#p0d00ad708b)" style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="line2d_18">
<path d="M 55.379882 281.474781 L 56.872484 280.133333 L 58.365086 279.181561 L 59.857687 278.44331 L 62.842891 277.330119 L 65.828094 276.498667 L 70.3059 275.546895 L 76.276306 274.595123 L 83.739315 273.695453 L 92.694926 272.864 L 104.635739 272.005429 L 119.561757 271.173976 L 138.965579 270.334365 L 164.339809 269.483952 L 195.684445 268.667237 L 235.984692 267.847291 L 288.225753 267.018829 L 350.915025 266.238961 L 350.915025 266.238961 " clip-path="url(#p0d00ad708b)" style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="line2d_19">
<path d="M 55.379882 280.133333 L 350.915025 280.133333 L 350.915025 280.133333 " clip-path="url(#p0d00ad708b)" style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="patch_3">
<path d="M 40.603125 283.768 L 40.603125 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_4">
<path d="M 365.691783 283.768 L 365.691783 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_5">
<path d="M 40.603125 283.768 L 365.691783 283.768 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_6">
<path d="M 40.603125 65.688 L 365.691783 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="text_19">
<text style="font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="203.147454" y="59.688" transform="rotate(-0 203.147454 59.688)">Échelle linéaire</text>
</g>
<g id="legend_1">
<g id="patch_7">
<path d="M 194.030298 109.917219 L 359.741783 109.917219 Q 361.441783 109.917219 361.441783 108.217219 L 361.441783 71.638 Q 361.441783 69.938 359.741783 69.938 L 194.030298 69.938 Q 192.330298 69.938 192.330298 71.638 L 192.330298 108.217219 Q 192.330298 109.917219 194.030298 109.917219 z " style="fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
</g>
<g id="line2d_20">
<path d="M 195.730298 76.821672 L 204.230298 76.821672 L 212.730298 76.821672 " style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_20">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="79.796672" transform="rotate(-0 219.530298 79.796672)">Récurrent : O(n)</text>
</g>
<g id="line2d_21">
<path d="M 195.730298 89.298078 L 204.230298 89.298078 L 212.730298 89.298078 " style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_21">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="92.273078" transform="rotate(-0 219.530298 92.273078)">Convolutionnel : O(logₖ n), k=3</text>
</g>
<g id="line2d_22">
<path d="M 195.730298 101.774484 L 204.230298 101.774484 L 212.730298 101.774484 " style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="text_22">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="104.749484" transform="rotate(-0 219.530298 104.749484)">Self-Attention : O(1)</text>
</g>
</g>
</g>
<g id="axes_2">
<g id="patch_8">
<path d="M 416.144467 283.768 L 741.233125 283.768 L 741.233125 65.688 L 416.144467 65.688 z " style="fill: #ffffff"/>
</g>
<g id="matplotlib.axis_3">
<g id="xtick_10">
<g id="line2d_23">
<g>
<use xlink:href="#m1fdfde7ee8" x="427.936021" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_23">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="427.936021" y="298.366437" transform="rotate(-0 427.936021 298.366437)">0</text>
</g>
</g>
<g id="xtick_11">
<g id="line2d_24">
<g>
<use xlink:href="#m1fdfde7ee8" x="465.251064" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_24">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="465.251064" y="298.366437" transform="rotate(-0 465.251064 298.366437)">25</text>
</g>
</g>
<g id="xtick_12">
<g id="line2d_25">
<g>
<use xlink:href="#m1fdfde7ee8" x="502.566108" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_25">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="502.566108" y="298.366437" transform="rotate(-0 502.566108 298.366437)">50</text>
</g>
</g>
<g id="xtick_13">
<g id="line2d_26">
<g>
<use xlink:href="#m1fdfde7ee8" x="539.881151" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_26">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="539.881151" y="298.366437" transform="rotate(-0 539.881151 298.366437)">75</text>
</g>
</g>
<g id="xtick_14">
<g id="line2d_27">
<g>
<use xlink:href="#m1fdfde7ee8" x="577.196194" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_27">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="577.196194" y="298.366437" transform="rotate(-0 577.196194 298.366437)">100</text>
</g>
</g>
<g id="xtick_15">
<g id="line2d_28">
<g>
<use xlink:href="#m1fdfde7ee8" x="614.511238" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_28">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="614.511238" y="298.366437" transform="rotate(-0 614.511238 298.366437)">125</text>
</g>
</g>
<g id="xtick_16">
<g id="line2d_29">
<g>
<use xlink:href="#m1fdfde7ee8" x="651.826281" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_29">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="651.826281" y="298.366437" transform="rotate(-0 651.826281 298.366437)">150</text>
</g>
</g>
<g id="xtick_17">
<g id="line2d_30">
<g>
<use xlink:href="#m1fdfde7ee8" x="689.141324" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_30">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="689.141324" y="298.366437" transform="rotate(-0 689.141324 298.366437)">175</text>
</g>
</g>
<g id="xtick_18">
<g id="line2d_31">
<g>
<use xlink:href="#m1fdfde7ee8" x="726.456368" y="283.768" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_31">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="726.456368" y="298.366437" transform="rotate(-0 726.456368 298.366437)">200</text>
</g>
</g>
<g id="text_32">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="578.688796" y="312.444562" transform="rotate(-0 578.688796 312.444562)">longueur de séquence n</text>
</g>
</g>
<g id="matplotlib.axis_4">
<g id="ytick_8">
<g id="line2d_32">
<g>
<use xlink:href="#mc1f461f8c3" x="416.144467" y="227.663946" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_33">
<g transform="translate(391.544467 231.463165)">
<text>
<tspan x="0" y="-0.976562" style="font-size: 10px; font-family: 'DejaVu Sans'">1</tspan>
<tspan x="6.362305" y="-0.976562" style="font-size: 10px; font-family: 'DejaVu Sans'">0</tspan>
<tspan x="12.820312" y="-4.804688" style="font-size: 7px; font-family: 'DejaVu Sans'">0</tspan>
</text>
</g>
</g>
</g>
<g id="ytick_9">
<g id="line2d_33">
<g>
<use xlink:href="#mc1f461f8c3" x="416.144467" y="161.579097" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_34">
<g transform="translate(391.544467 165.378316)">
<text>
<tspan x="0" y="-0.064063" style="font-size: 10px; font-family: 'DejaVu Sans'">1</tspan>
<tspan x="6.362305" y="-0.064063" style="font-size: 10px; font-family: 'DejaVu Sans'">0</tspan>
<tspan x="12.820312" y="-3.892188" style="font-size: 7px; font-family: 'DejaVu Sans'">1</tspan>
</text>
</g>
</g>
</g>
<g id="ytick_10">
<g id="line2d_34">
<g>
<use xlink:href="#mc1f461f8c3" x="416.144467" y="95.494249" style="stroke: #000000; stroke-width: 0.8"/>
</g>
</g>
<g id="text_35">
<g transform="translate(391.544467 99.293468)">
<text>
<tspan x="0" y="-0.976562" style="font-size: 10px; font-family: 'DejaVu Sans'">1</tspan>
<tspan x="6.362305" y="-0.976562" style="font-size: 10px; font-family: 'DejaVu Sans'">0</tspan>
<tspan x="12.820312" y="-4.804688" style="font-size: 7px; font-family: 'DejaVu Sans'">2</tspan>
</text>
</g>
</g>
</g>
<g id="ytick_11">
<g id="line2d_35">
<defs>
<path id="m5c3f018d7c" d="M 0 0 L -2 0 " style="stroke: #000000; stroke-width: 0.6"/>
</defs>
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="273.855273" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_12">
<g id="line2d_36">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="262.218309" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_13">
<g id="line2d_37">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="253.961751" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_14">
<g id="line2d_38">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="247.557468" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_15">
<g id="line2d_39">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="242.324787" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_16">
<g id="line2d_40">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="237.900618" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_17">
<g id="line2d_41">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="234.068229" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_18">
<g id="line2d_42">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="230.687823" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_19">
<g id="line2d_43">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="207.770424" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_20">
<g id="line2d_44">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="196.13346" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_21">
<g id="line2d_45">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="187.876903" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_22">
<g id="line2d_46">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="181.472619" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_23">
<g id="line2d_47">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="176.239938" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_24">
<g id="line2d_48">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="171.81577" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_25">
<g id="line2d_49">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="167.983381" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_26">
<g id="line2d_50">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="164.602974" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_27">
<g id="line2d_51">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="141.685576" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_28">
<g id="line2d_52">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="130.048612" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_29">
<g id="line2d_53">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="121.792054" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_30">
<g id="line2d_54">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="115.387771" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_31">
<g id="line2d_55">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="110.15509" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_32">
<g id="line2d_56">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="105.730921" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_33">
<g id="line2d_57">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="101.898532" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_34">
<g id="line2d_58">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="98.518126" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="ytick_35">
<g id="line2d_59">
<g>
<use xlink:href="#m5c3f018d7c" x="416.144467" y="75.600727" style="stroke: #000000; stroke-width: 0.6"/>
</g>
</g>
</g>
<g id="text_36">
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="385.46478" y="174.728" transform="rotate(-90 385.46478 174.728)">longueur de chemin maximale (échelle log)</text>
</g>
</g>
<g id="line2d_60">
<path d="M 430.921225 207.770424 L 432.413826 196.13346 L 433.906428 187.876903 L 435.39903 181.472619 L 436.891631 176.239938 L 438.384233 171.81577 L 439.876835 167.983381 L 441.369437 164.602974 L 442.862038 161.579097 L 444.35464 158.843668 L 447.339844 154.049168 L 450.325047 149.942133 L 453.310251 146.349915 L 456.295454 143.157708 L 459.280657 140.285284 L 462.265861 137.674368 L 465.251064 135.281292 L 469.72887 132.028727 L 474.206675 129.107534 L 478.68448 126.456394 L 483.162285 124.029573 L 489.132692 121.083369 L 495.103099 118.411647 L 501.073506 115.967594 L 508.536515 113.178967 L 515.999523 110.637459 L 524.955134 107.857841 L 533.910744 105.323818 L 544.358956 102.625161 L 554.807168 100.158589 L 566.747982 97.577048 L 580.181398 94.925908 L 593.614814 92.499087 L 608.540831 90.02339 L 624.95945 87.526139 L 642.870671 85.028888 L 662.274493 82.54825 L 683.170918 80.096736 L 707.052545 77.529638 L 726.456368 75.600727 L 726.456368 75.600727 " clip-path="url(#p2a222495cd)" style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="line2d_61">
<path d="M 430.921225 240.882171 L 432.413826 227.663946 L 433.906428 220.988649 L 435.39903 216.70511 L 436.891631 213.625194 L 438.384233 211.25651 L 439.876835 209.351685 L 441.369437 207.770424 L 442.862038 206.426176 L 444.35464 205.262123 L 447.339844 203.329235 L 450.325047 201.771092 L 453.310251 200.474338 L 457.788056 198.873517 L 462.265861 197.564838 L 466.743666 196.464001 L 472.714073 195.230336 L 480.177082 193.958184 L 489.132692 192.708537 L 499.580904 191.51545 L 511.521718 190.394776 L 526.447736 189.246591 L 542.866355 188.209579 L 563.762779 187.126519 L 589.137008 186.056983 L 618.989043 185.033995 L 654.811485 184.035069 L 698.096935 183.054469 L 726.456368 182.508597 L 726.456368 182.508597 " clip-path="url(#p2a222495cd)" style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="line2d_62">
<path d="M 430.921225 227.663946 L 726.456368 227.663946 L 726.456368 227.663946 " clip-path="url(#p2a222495cd)" style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="line2d_63">
<path d="M 430.921225 273.855273 L 432.413826 262.218309 L 433.906428 253.961751 L 435.39903 247.557468 L 436.891631 242.324787 L 438.384233 237.900618 L 439.876835 234.068229 L 441.369437 230.687823 L 442.862038 227.663946 L 444.35464 224.928517 L 447.339844 220.134017 L 450.325047 216.026982 L 453.310251 212.434764 L 456.295454 209.242556 L 459.280657 206.370133 L 462.265861 203.759217 L 465.251064 201.366141 L 469.72887 198.113575 L 474.206675 195.192383 L 478.68448 192.541242 L 483.162285 190.114421 L 489.132692 187.168218 L 495.103099 184.496496 L 501.073506 182.052443 L 508.536515 179.263815 L 515.999523 176.722308 L 524.955134 173.94269 L 533.910744 171.408667 L 544.358956 168.710009 L 554.807168 166.243437 L 566.747982 163.661897 L 580.181398 161.010756 L 593.614814 158.583935 L 608.540831 156.108239 L 624.95945 153.610987 L 642.870671 151.113736 L 662.274493 148.633099 L 683.170918 146.181585 L 707.052545 143.614486 L 726.456368 141.685576 L 726.456368 141.685576 " clip-path="url(#p2a222495cd)" style="fill: none; stroke-dasharray: 6.66,2.88; stroke-dashoffset: 0; stroke: #e08a2c; stroke-width: 1.8"/>
</g>
<g id="patch_9">
<path d="M 416.144467 283.768 L 416.144467 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_10">
<path d="M 741.233125 283.768 L 741.233125 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_11">
<path d="M 416.144467 283.768 L 741.233125 283.768 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="patch_12">
<path d="M 416.144467 65.688 L 741.233125 65.688 " style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
</g>
<g id="text_37">
<text style="font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="578.688796" y="59.688" transform="rotate(-0 578.688796 59.688)">Échelle logarithmique</text>
</g>
<g id="legend_2">
<g id="patch_13">
<path d="M 552.609375 279.768 L 735.633125 279.768 Q 737.233125 279.768 737.233125 278.168 L 737.233125 231.998 Q 737.233125 230.398 735.633125 230.398 L 552.609375 230.398 Q 551.009375 230.398 551.009375 231.998 L 551.009375 278.168 Q 551.009375 279.768 552.609375 279.768 z " style="fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
</g>
<g id="line2d_64">
<path d="M 554.209375 236.87675 L 562.209375 236.87675 L 570.209375 236.87675 " style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_38">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="239.67675" transform="rotate(-0 576.609375 239.67675)">Récurrent : O(n)</text>
</g>
<g id="line2d_65">
<path d="M 554.209375 248.61925 L 562.209375 248.61925 L 570.209375 248.61925 " style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_39">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="251.41925" transform="rotate(-0 576.609375 251.41925)">Convolutionnel : O(logₖ n)</text>
</g>
<g id="line2d_66">
<path d="M 554.209375 260.36175 L 562.209375 260.36175 L 570.209375 260.36175 " style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="text_40">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="263.16175" transform="rotate(-0 576.609375 263.16175)">Self-Attention : O(1)</text>
</g>
<g id="line2d_67">
<path d="M 554.209375 272.10425 L 562.209375 272.10425 L 570.209375 272.10425 " style="fill: none; stroke-dasharray: 6.66,2.88; stroke-dashoffset: 0; stroke: #e08a2c; stroke-width: 1.8"/>
</g>
<g id="text_41">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="274.90425" transform="rotate(-0 576.609375 274.90425)">Self-Attention restreinte : O(n/r), r=10</text>
</g>
</g>
</g>
<g id="text_42">
<text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="374.033125" y="15.938203" transform="rotate(-0 374.033125 15.938203)">Longueur de chemin maximale entre deux positions, selon le type de couche (Tableau 1)</text>
</g>
</g>
<defs>
<clipPath id="p0d00ad708b">
<rect x="40.603125" y="65.688" width="325.088658" height="218.08"/>
</clipPath>
<clipPath id="p2a222495cd">
<rect x="416.144467" y="65.688" width="325.088658" height="218.08"/>
</clipPath>
</defs>
</svg>
</div>

## Les résultats, en bref

Sur WMT 2014 anglais-allemand, le modèle « big » atteint **28.4 BLEU**, plus de 2 points au-dessus du meilleur résultat précédent, ensembles de modèles inclus. Sur anglais-français, **41.0 BLEU**, à moins d'un quart du coût d'entraînement du précédent état de l'art. Et même le modèle « base » (12h sur 8 GPU) dépasse tout ce qui existait avant lui.

Deux détails d'entraînement qui valent le détour :

- **Le warmup du learning rate** : montée linéaire pendant 4000 pas, puis décroissance en 1/√pas. Avec un réseau aussi profond (6+6 couches, pleines de connexions résiduelles et de layer norms), un learning rate élevé dès le départ produirait des mises à jour instables qui s'amplifient en cascade à travers les couches.
- **Le label smoothing** (ε=0.1) : émousse volontairement les cibles d'entraînement. Ça dégrade la perplexité (le modèle devient « moins sûr de lui ») mais améliore l'accuracy et le BLEU, un compromis délibéré contre le sur-apprentissage.

## Ce qu'il faut retenir

- Le problème des RNN, ce n'est pas la profondeur (ça, tout le monde l'a) : c'est la **dépendance séquentielle horizontale** qui empêche la parallélisation à l'intérieur d'une séquence.
- L'attention, c'est une moyenne pondérée : query interroge des keys, softmax transforme les scores en poids, on combine les values.
- Le scaling par √dₖ empêche le softmax de saturer quand la dimension grandit.
- Multi-head = plusieurs « regards » parallèles sur la même séquence, avec un vrai compromis nombre de têtes / capacité par tête.
- Le masquage causal garde le décodeur honnête pendant l'entraînement parallélisé.
- Tout ça a un prix : une complexité quadratique en longueur de séquence, encore un sujet de recherche actif aujourd'hui.

**La suite :** je passe à l'implémentation, le Transformer complet en PyTorch (encodeur-décodeur, multi-head attention, positional encoding, masquage) pour vérifier que je comprends vraiment chaque étape, pas juste la formule. À suivre.
