---
title: "Attention Is All You Need: When Attention Replaces Recurrence"
date: 2026-08-26 10:00:01 +0000
categories: [Deep Learning, NLP]
tags: [transformer, attention, nlp, deep-learning, week-3]
render_with_liquid: false
hidden: true
sitemap: false
---

*🇫🇷 [Version française](https://laryconseiga.github.io/Learning-by-Building/posts/attention-is-all-you-need-quand-lattention-remplace-la-recurrence/)*

Week 3 of my reading plan. After MLPs, CNNs, and LSTMs, it's time for the paper that redefined modern NLP: **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017). This is the paper that introduces the **Transformer**, the architecture behind pretty much every major language model today.

The paper's pitch is almost provocative: you can throw recurrent networks (RNNs, LSTMs) in the trash and keep "only" attention. The result: a new state-of-the-art in translation, achieved in a fraction of the training time of previous models (3.5 days on 8 GPUs versus several weeks for the competition).

In this article, I rebuild the paper's reasoning step by step: why RNNs are a problem, how attention actually works, and how it all comes together in the full architecture. The accompanying code (a from-scratch Transformer implementation in PyTorch) is available in [this GitHub repository](https://github.com/LaryConseiga/Architecture-Transformer-from-Scratch){:target="_blank"}.

## The problem the Transformer solves

An RNN processes a sequence word by word. To compute its hidden state at position *t*, it **needs** the state at position *t-1*, which needs *t-2*, and so on. This chain of dependencies is strict: it's impossible to compute state 50 without having gone through the previous 49.

The problem is that this **forbids any parallelization within a single sequence**. Even with 8 GPUs, you can't parallelize the computation of states within one sentence, only parallelize across several different sentences. On long sequences, this is a real bottleneck.

CNNs already solve part of the problem: within a layer, each position is computed independently of the others, from an already-available local window. But to connect two distant positions in the sequence, you have to stack several layers: the "receptive field" grows progressively, layer after layer.

The Transformer pushes this idea to its logical conclusion: what if each position could directly consult **every other position**, in a single operation, with no limited window and no stacking needed to widen its reach? That's exactly what attention does.

## The logic of attention, in one picture

Before the formulas, the intuition. The attention mechanism rests on three objects: the **query** (what I'm looking for), the **key** (a label that says "here's what I contain"), and the **value** (the actual content retrieved if the match is good).

It's a bit like how a search engine works: your query is compared against document labels (keys), and you retrieve the content (values) of the documents whose label matches best, except that here, instead of retrieving only the best match, you retrieve a **weighted average of everything**, proportional to the quality of the match.

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
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'" transform="translate(9.3038 76.9335)">("what I'm looking for")</text>
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
    <text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="230.9994" y="98.849083" transform="rotate(-0 230.9994 98.849083)">raw scores</text>
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
    <text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(349.309997 94.090013)">weights</text>
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
    <text style="font-weight: 700; font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="470.05785" y="157.18699" transform="rotate(-0 470.05785 157.18699)">Output</text>
   </g>
   <g id="text_13">
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'" transform="translate(449.793475 169.126656)">(weighted</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'" transform="translate(449.40785 178.468906)">average)</text>
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
    <text style="font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(367.484575 220.556031)">weights applied</text>
    <text style="font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(377.21395 229.514281)">to values</text>
   </g>
   <g id="text_15">
    <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="261.5175" y="15.998281" transform="rotate(-0 261.5175 15.998281)">The general logic of attention: Query + Keys → weights → weighted average of Values</text>
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

## Scaled Dot-Product Attention: the formula

Concretely, here's the paper's central formula:

**Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V**

Broken down into steps:

1. **QKᵀ**: the dot product between every query and every key. For a sequence of *n* tokens, this gives an *n×n* matrix: a compatibility score for every pair of positions.
2. **/ √dₖ**: scaling. Without this, when dimension dₖ is large, the dot products explode in magnitude and push the softmax into a saturated region (near-zero gradient). Dividing by √dₖ exactly compensates for that growth.
3. **softmax**: turns the raw scores into weights that sum to 1 on each row.
4. **× V**: combines the values according to those weights: this is the final output, a weighted average.

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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(40.141518 146.886578)">input</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(40.787143 155.844828)">matrices</text>
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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(128.886429 146.886578)">dot</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(127.757679 155.844828)">product</text>
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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(215.710714 146.886578)">rescales</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(212.434464 155.844828)">the scores</text>
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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(295.8725 146.886578)">decoder</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(293.586875 155.844828)">only</text>
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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(389.693036 146.886578)">weights</text>
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
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(467.424821 146.694578)">weighted</text>
    <text style="font-style: italic; font-size: 8px; font-family: 'DejaVu Sans'; fill: #444444" transform="translate(467.039196 156.036828)">average</text>
   </g>
   <g id="text_13">
    <text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #444444" x="271.842857" y="207.128203" transform="rotate(-0 271.842857 207.128203)">Example: dₖ=64  →  divide by √64=8 to keep the softmax from saturating</text>
   </g>
   <g id="text_14">
    <text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="277.2" y="16.398203" transform="rotate(-0 277.2 16.398203)">Scaled Dot-Product Attention: the full chain of operations</text>
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

## What this looks like, concretely

Here's an illustrative example (values invented for pedagogical purposes, not drawn from a trained model) on the sentence "The black cat sleeps deeply." Each row represents one word's attention distribution over every word in the sentence (including itself), and always sums to 1:

What we see: "cat" pays strong attention to "black" (the noun looks at the adjective describing it), and "deeply" pays strong attention to "sleeps" (the adverb looks at the verb it modifies). This is exactly the kind of syntactic structure the authors report having observed in the attention heads of a real trained model.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="446.4pt" height="374.4pt" viewBox="0 0 446.4 374.4" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:32:45.871124</dc:date>
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
   <g clip-path="url(#p95579c5af2)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAYUAAAGGCAYAAACUt53mAAAFjUlEQVR4nO3XvYkUYBSGUUd2EAYWR9gCVtAqtgMTEwM7MNIeDIxNTTUxsQQ7MNgGDA1ERPxBwR8YsycVTa4fe04Fb3J5uJsPX38dLvFX3n3+Pj1hSfvddnrCkk5vPZyesKQXT+5PT1jS5ekBAPw/RAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoAZPP204/D9IjV3Lz3fHrCkl49vjM9YUmv33+ZnrCkGyfH0xOW5FMAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFAHI0PYCL49n5m+kJS3pwdn16AheITwGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAyNHHbz+nNyxnf7KfnrCk02tXpicsyY3+m/1uOz1hST4FACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCb47tPD9MjVvPy0e3pCUu6uttOTwD+wKcAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAEQUAIgoABBRACCiAEBEAYCIAgARBQAiCgBEFACIKAAQUQAgogBARAGAiAIAEQUAIgoARBQAiCgAEFEAIKIAQEQBgIgCABEFACIKAOQ3igAjTtKTlgMAAAAASUVORK5CYII=" id="imagebf54bad1ad" transform="scale(1 -1) translate(0 -280.8)" x="56.16" y="-52.56" width="280.08" height="280.8"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="mb5648d20e8" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#mb5648d20e8" x="83.841337" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- The -->
      <g transform="translate(64.802657 357.490311) rotate(-30) scale(0.11 -0.11)">
       <defs>
        <path id="DejaVuSans-37" d="M -19 4666 
L 3928 4666 
L 3928 4134 
L 2272 4134 
L 2272 0 
L 1638 0 
L 1638 4134 
L -19 4134 
L -19 4666 
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
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#mb5648d20e8" x="139.924012" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- black -->
      <g transform="translate(113.316405 361.860233) rotate(-30) scale(0.11 -0.11)">
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
        <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
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
        <path id="DejaVuSans-4e" d="M 581 4863 
L 1159 4863 
L 1159 1991 
L 2875 3500 
L 3609 3500 
L 1753 1863 
L 3688 0 
L 2938 0 
L 1159 1709 
L 1159 0 
L 581 0 
L 581 4863 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-46" transform="translate(152.546875 0)"/>
       <use xlink:href="#DejaVuSans-4e" transform="translate(207.53125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#mb5648d20e8" x="196.006687" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- cat -->
      <g transform="translate(179.87501 355.811208) rotate(-30) scale(0.11 -0.11)">
       <defs>
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
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#mb5648d20e8" x="252.089362" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- sleeps -->
      <g transform="translate(220.425384 364.77953) rotate(-30) scale(0.11 -0.11)">
       <defs>
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
       </defs>
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_5">
      <g>
       <use xlink:href="#mb5648d20e8" x="308.172036" y="333.022687" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- deeply -->
      <g transform="translate(274.747186 365.796171) rotate(-30) scale(0.11 -0.11)">
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
       </defs>
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(125.015625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(186.546875 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(250.03125 0)"/>
       <use xlink:href="#DejaVuSans-5c" transform="translate(277.8125 0)"/>
      </g>
     </g>
    </g>
    <g id="text_6">
     <!-- Keyⱼ (who is being attended to) -->
     <g transform="translate(118.251999 379.683148) scale(0.1 -0.1)">
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
       <path id="DejaVuSans-5a" d="M 269 3500 
L 844 3500 
L 1563 769 
L 2278 3500 
L 2956 3500 
L 3675 769 
L 4391 3500 
L 4966 3500 
L 4050 0 
L 3372 0 
L 2619 2869 
L 1863 0 
L 1184 0 
L 269 3500 
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
      <use xlink:href="#DejaVuSans-5a" transform="translate(269.59375 0)"/>
      <use xlink:href="#DejaVuSans-4b" transform="translate(351.375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(414.75 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(475.9375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(507.71875 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(535.5 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(587.59375 0)"/>
      <use xlink:href="#DejaVuSans-45" transform="translate(619.375 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(682.859375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(744.390625 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(772.171875 0)"/>
      <use xlink:href="#DejaVuSans-4a" transform="translate(835.546875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(899.03125 0)"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(930.8125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(992.09375 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1031.296875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1070.5 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(1132.03125 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(1195.40625 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1258.890625 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(1320.421875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(1383.90625 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1415.6875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(1454.890625 0)"/>
      <use xlink:href="#DejaVuSans-c" transform="translate(1516.078125 0)"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_6">
      <defs>
       <path id="m196d198ad6" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m196d198ad6" x="55.8" y="80.65065" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- The -->
      <g transform="translate(28.341719 84.829791) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_7">
      <g>
       <use xlink:href="#m196d198ad6" x="55.8" y="136.733325" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- black -->
      <g transform="translate(19.601875 140.912466) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-45"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(91.265625 0)"/>
       <use xlink:href="#DejaVuSans-46" transform="translate(152.546875 0)"/>
       <use xlink:href="#DejaVuSans-4e" transform="translate(207.53125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_8">
      <g>
       <use xlink:href="#m196d198ad6" x="55.8" y="192.816" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- cat -->
      <g transform="translate(31.698438 196.994711) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_9">
      <g>
       <use xlink:href="#m196d198ad6" x="55.8" y="248.898675" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- sleeps -->
      <g transform="translate(13.763281 253.077815) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_10">
      <g>
       <use xlink:href="#m196d198ad6" x="55.8" y="304.98135" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- deeply -->
      <g transform="translate(11.73 309.16049) scale(0.11 -0.11)">
       <use xlink:href="#DejaVuSans-47"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(63.484375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(125.015625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(186.546875 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(250.03125 0)"/>
       <use xlink:href="#DejaVuSans-5c" transform="translate(277.8125 0)"/>
      </g>
     </g>
    </g>
    <g id="text_12">
     <!-- Queryᵢ (the position doing the looking) -->
     <g transform="translate(5.327656 289.069125) rotate(-90) scale(0.1 -0.1)">
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
      <use xlink:href="#DejaVuSans-3" transform="translate(321.796875 0)"/>
      <use xlink:href="#DejaVuSans-b" transform="translate(353.578125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(392.59375 0)"/>
      <use xlink:href="#DejaVuSans-4b" transform="translate(431.796875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(495.171875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(556.703125 0)"/>
      <use xlink:href="#DejaVuSans-53" transform="translate(588.484375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(651.96875 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(713.15625 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(765.25 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(793.03125 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(832.234375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(860.015625 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(921.203125 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(984.578125 0)"/>
      <use xlink:href="#DejaVuSans-47" transform="translate(1016.359375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(1079.84375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(1141.03125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(1168.8125 0)"/>
      <use xlink:href="#DejaVuSans-4a" transform="translate(1232.1875 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(1295.671875 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(1327.453125 0)"/>
      <use xlink:href="#DejaVuSans-4b" transform="translate(1366.65625 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(1430.03125 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(1491.5625 0)"/>
      <use xlink:href="#DejaVuSans-4f" transform="translate(1523.34375 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(1551.125 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(1612.3125 0)"/>
      <use xlink:href="#DejaVuSans-4e" transform="translate(1673.5 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(1731.40625 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(1759.1875 0)"/>
      <use xlink:href="#DejaVuSans-4a" transform="translate(1822.5625 0)"/>
      <use xlink:href="#DejaVuSans-c" transform="translate(1886.046875 0)"/>
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
iVBORw0KGgoAAAANSUhEUgAAABMAAAGGCAYAAACZjLqIAAABwElEQVR4nO2cW4oEMQwDvUPf/7r7MZDMGWQEXSDpACKFnMTOPP7+v/eOSR+X0czM41vXzHPH5+bFNC5sHqNX01yoaW7MsGk6Ma0rC8G0lgYa02jmPIKapq6muTEDYzrNsJjTNHUzMKYxTfIR5DQDY1Ib5I47C7Omqapp6mqaG7Omqapp6mqaG7OINNsg6+q4szEDYzZN1ewY4wRjhhTtycBsmrLAmOA0sSs7IQFkYMak6TTDNi7HacbFbJq6GRfT/OLikzlNIycY01ln1h0ALlrwJIy90a3bCTzuWOsM+xxt3k4+M3DRkvem0YzbuFgxuUWb8rCUcaOb96bPrCetLnCDnBJAr7pXzTq86vJikvdmBGbT1M1S3s+waYYUbciNzsVM+cZSSJoZM3pImt4jKONtOyVNLGaPoI1ZAia3c8zBTKizYuoqpq5irsyKKaqYutoe6ALX2TEOT2DMkDSLKYt7apB/jZsyO0Vgev9fA4vZNHUzLmbGysCYrbN3zZqmrpQ024a+a1ZMXT2CFmaDxZzr+xZyCKZ7ZdY642K2znSzYooqpq5ibsywH8iY0+ReKCH3ZjFFobdTMUUVU5cV8wcYxRvIsD0NiAAAAABJRU5ErkJggg==" id="image8e087fe748" transform="scale(1 -1) translate(0 -280.8)" x="388.08" y="-52.56" width="13.68" height="280.8"/>
   <g id="matplotlib.axis_3"/>
   <g id="matplotlib.axis_4">
    <g id="ytick_6">
     <g id="line2d_11">
      <defs>
       <path id="m4b466afe82" d="M 0 0 
L 3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m4b466afe82" x="401.76" y="333.216" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="290.016" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="246.816" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="203.616" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="160.416" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="117.216" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m4b466afe82" x="401.76" y="74.016" style="stroke: #000000; stroke-width: 0.8"/>
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
     <!-- attention weights -->
     <g transform="translate(433.911406 232.288734) rotate(-90) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-44"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(61.28125 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(100.484375 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(139.6875 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(201.21875 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(264.59375 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(303.796875 0)"/>
      <use xlink:href="#DejaVuSans-52" transform="translate(331.578125 0)"/>
      <use xlink:href="#DejaVuSans-51" transform="translate(392.765625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(456.140625 0)"/>
      <use xlink:href="#DejaVuSans-5a" transform="translate(487.921875 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(569.703125 0)"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(631.234375 0)"/>
      <use xlink:href="#DejaVuSans-4a" transform="translate(659.015625 0)"/>
      <use xlink:href="#DejaVuSans-4b" transform="translate(722.5 0)"/>
      <use xlink:href="#DejaVuSans-57" transform="translate(785.875 0)"/>
      <use xlink:href="#DejaVuSans-56" transform="translate(825.078125 0)"/>
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
   <!-- Illustrative example: attention weights for "The black cat sleeps deeply" -->
   <g transform="translate(33.291914 7.978359) scale(0.105 -0.105)">
    <defs>
     <path id="DejaVuSans-2c" d="M 628 4666 
L 1259 4666 
L 1259 0 
L 628 0 
L 628 4666 
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
     <path id="DejaVuSans-5" d="M 1147 4666 
L 1147 2931 
L 616 2931 
L 616 4666 
L 1147 4666 
z
M 2328 4666 
L 2328 2931 
L 1797 2931 
L 1797 4666 
L 2328 4666 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-2c"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(29.5 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(57.28125 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(85.0625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(148.4375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(200.53125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(239.734375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(280.84375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(342.125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(381.328125 0)"/>
    <use xlink:href="#DejaVuSans-59" transform="translate(409.109375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(468.296875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(529.828125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(561.609375 0)"/>
    <use xlink:href="#DejaVuSans-5b" transform="translate(621.390625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(680.578125 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(741.859375 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(839.265625 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(902.75 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(930.53125 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(992.0625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1025.75 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1057.53125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1118.8125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1158.015625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1197.21875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1258.75 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1322.125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1361.328125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1389.109375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1450.296875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1513.671875 0)"/>
    <use xlink:href="#DejaVuSans-5a" transform="translate(1545.453125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1627.234375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1688.765625 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(1716.546875 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1780.03125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1843.40625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1882.609375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1934.703125 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(1966.484375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2001.6875 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2062.875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2103.984375 0)"/>
    <use xlink:href="#DejaVuSans-5" transform="translate(2135.765625 0)"/>
    <use xlink:href="#DejaVuSans-37" transform="translate(2181.765625 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(2242.84375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2306.21875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2367.75 0)"/>
    <use xlink:href="#DejaVuSans-45" transform="translate(2399.53125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2463.015625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2490.796875 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(2552.078125 0)"/>
    <use xlink:href="#DejaVuSans-4e" transform="translate(2607.0625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2664.96875 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(2696.75 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2751.734375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2813.015625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2852.21875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2884 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2936.09375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2963.875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3025.40625 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(3086.9375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(3150.421875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(3202.515625 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(3234.296875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3297.78125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3359.3125 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(3420.84375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(3484.328125 0)"/>
    <use xlink:href="#DejaVuSans-5c" transform="translate(3512.109375 0)"/>
    <use xlink:href="#DejaVuSans-5" transform="translate(3571.296875 0)"/>
   </g>
  </g>
  <g id="text_52">
   <!-- (illustrative values for teaching purposes, not from a trained model) -->
   <g transform="translate(69.885703 16.848) scale(0.09 -0.09)">
    <defs>
     <path id="DejaVuSans-f" d="M 750 794 
L 1409 794 
L 1409 256 
L 897 -744 
L 494 -744 
L 750 256 
L 750 794 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-b"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(39.015625 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(66.796875 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(94.578125 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(122.359375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(185.734375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(237.828125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(277.03125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(318.140625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(379.421875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(418.625 0)"/>
    <use xlink:href="#DejaVuSans-59" transform="translate(446.40625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(505.59375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(567.125 0)"/>
    <use xlink:href="#DejaVuSans-59" transform="translate(598.90625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(658.09375 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(719.375 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(747.15625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(810.53125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(872.0625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(924.15625 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(955.9375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(991.140625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(1052.328125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1093.4375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1125.21875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1164.421875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1225.953125 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1287.234375 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1342.21875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1405.59375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1433.375 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(1496.75 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1560.234375 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1592.015625 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(1655.5 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(1718.875 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1759.984375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1823.46875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1884.65625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1936.75 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1998.28125 0)"/>
    <use xlink:href="#DejaVuSans-f" transform="translate(2050.375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2082.15625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2113.9375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2177.3125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2238.5 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2277.703125 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(2309.484375 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2344.6875 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2383.59375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(2444.78125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2542.1875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2573.96875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2635.25 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2667.03125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2706.234375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2747.34375 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2808.625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2836.40625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2899.78125 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(2961.3125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(3024.796875 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(3056.578125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(3153.984375 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(3215.171875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3278.65625 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(3340.1875 0)"/>
    <use xlink:href="#DejaVuSans-c" transform="translate(3367.96875 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p95579c5af2">
   <rect x="55.8" y="52.609313" width="280.413374" height="280.413374"/>
  </clipPath>
 </defs>
</svg>
</div>

## Multi-Head Attention: several viewpoints in parallel

A single attention function is a single "weighted average," a single point of view on the sentence. The problem: this average can dilute information that should have stayed sharp (a pronoun that should point very precisely to a single noun, for instance).

The paper's solution: project Q, K, V several times with **different, learned** projection matrices, compute attention in parallel on each projection, then concatenate and project one last time.

**MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ**, with **headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)**

In the paper: **h = 8 heads**, each of dimension dₖ = dᵥ = 512/8 = 64. An important trick: thanks to this reduced per-head dimension, the total computational cost stays comparable to that of a single full-dimension head: the same compute budget, simply split across 8 subspaces instead of one.

Each head can then specialize: one captures nearby syntactic relations, another captures long-distance references, and so on. This is exactly what the authors observe when inspecting the heads of a trained model (see the paper's appendix).

> **The hidden trade-off:** the paper's ablation study (Table 3) shows that a single head gives 24.9 BLEU, 8 heads give 25.8, and 32 heads drop back down to 25.4. Too few heads means not enough different perspectives. Too many heads means each head becomes too narrow (dₖ=16) to represent anything useful. There's a real optimum to find.
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
<text style="font-weight: 700; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(413.760206 212.282186)">output d_model=512</text>
</g>
<g id="text_12">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #444444" x="277.2" y="50.901733" transform="rotate(-0 277.2 50.901733)">In practice, this paper uses h=8 heads (4 shown here for readability)</text>
</g>
<g id="text_13">
<text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="277.2" y="16.398203" transform="rotate(-0 277.2 16.398203)">Multi-Head Attention: h attentions in parallel, then concatenation and projection</text>
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

## Why the decoder must be masked

The decoder generates its output word by word, in order. When predicting word 3, it **logically has no right** to know word 4: that word doesn't exist yet at the actual moment of generation.

The problem: during training, to parallelize computation, the model is given the **entire** target sentence at once. Without precaution, the decoder's self-attention could "cheat" by looking directly at future words, which it will never have access to during real inference.

The solution: a **causal mask**. All compatibility scores pointing to a future position are forced to −∞ before the softmax, which gives them a weight of 0 after exponentiation. Each position can then only "see" itself and the positions that precede it.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="670.79952pt" height="381.033698pt" viewBox="0 0 670.79952 381.033698" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:44:31.024266</dc:date>
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
   <path d="M 0 381.033698 
L 670.79952 381.033698 
L 670.79952 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 61.644141 342.332877 
L 332.08776 342.332877 
L 332.08776 71.889258 
L 61.644141 71.889258 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#pb6d38f8c11)">
    <image xlink:href="data:image/png;base64,
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" id="imagead561ef696" transform="scale(1 -1) translate(0 -270)" x="61.92" y="-72.153698" width="270" height="270"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="mbb6309f273" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#mbb6309f273" x="95.449593" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- The -->
      <g transform="translate(85.685413 357.311237) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-37" d="M -19 4666 
L 3928 4666 
L 3928 4134 
L 2272 4134 
L 2272 0 
L 1638 0 
L 1638 4134 
L -19 4134 
L -19 4666 
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
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#mbb6309f273" x="163.060498" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- cat -->
      <g transform="translate(154.898389 357.310416) scale(0.105 -0.105)">
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
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#mbb6309f273" x="230.671403" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- sleeps -->
      <g transform="translate(213.949332 357.311237) scale(0.105 -0.105)">
       <defs>
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
        <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
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
       </defs>
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#mbb6309f273" x="298.282308" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- well -->
      <g transform="translate(287.84137 357.311237) scale(0.105 -0.105)">
       <defs>
        <path id="DejaVuSans-5a" d="M 269 3500 
L 844 3500 
L 1563 769 
L 2278 3500 
L 2956 3500 
L 3675 769 
L 4391 3500 
L 4966 3500 
L 4050 0 
L 3372 0 
L 2619 2869 
L 1863 0 
L 1184 0 
L 269 3500 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-5a"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(81.78125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(143.3125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(171.09375 0)"/>
      </g>
     </g>
    </g>
    <g id="text_5">
     <!-- Keyⱼ -->
     <g transform="translate(186.926107 371.431354) scale(0.1 -0.1)">
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
       <path id="m583309ef95" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m583309ef95" x="61.644141" y="105.69471" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- The -->
      <g transform="translate(35.115781 109.68389) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_6">
      <g>
       <use xlink:href="#m583309ef95" x="61.644141" y="173.305615" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- cat -->
      <g transform="translate(38.319922 177.294385) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_7">
      <g>
       <use xlink:href="#m583309ef95" x="61.644141" y="240.91652" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- sleeps -->
      <g transform="translate(21.2 244.9057) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_8">
      <g>
       <use xlink:href="#m583309ef95" x="61.644141" y="308.527425" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- well -->
      <g transform="translate(33.762266 312.516605) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-5a"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(81.78125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(143.3125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(171.09375 0)"/>
      </g>
     </g>
    </g>
    <g id="text_10">
     <!-- Queryᵢ -->
     <g transform="translate(14.797656 223.200911) rotate(-90) scale(0.1 -0.1)">
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
    <path d="M 61.644141 342.332877 
L 61.644141 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 332.08776 342.332877 
L 332.08776 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 61.644141 342.332877 
L 332.08776 342.332877 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 61.644141 71.889258 
L 332.08776 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_11">
    <!-- 0.10 -->
    <g style="fill: #222222" transform="translate(84.316781 108.292367) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(151.927685 108.292367) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(219.53859 108.292367) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(287.149495 108.292367) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(84.316781 175.903272) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_16">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(151.927685 175.903272) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_17">
    <!-- 0.36 -->
    <g style="fill: #222222" transform="translate(219.53859 175.903272) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-19" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_18">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(287.149495 175.903272) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(84.316781 243.514176) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(151.927685 243.514176) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_21">
    <!-- 0.23 -->
    <g style="fill: #222222" transform="translate(219.53859 243.514176) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_22">
    <!-- 0.28 -->
    <g style="fill: #222222" transform="translate(287.149495 243.514176) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1b" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_23">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(84.316781 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_24">
    <!-- 0.22 -->
    <g style="fill: #222222" transform="translate(151.927685 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_25">
    <!-- 0.27 -->
    <g style="fill: #222222" transform="translate(219.53859 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_26">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(287.149495 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_27">
    <!-- cheats: sees the future -->
    <g style="fill: #555555" transform="translate(144.91556 44.844896) scale(0.09 -0.09)">
     <defs>
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
      <path id="DejaVuSans-3" transform="scale(0.015625)"/>
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
     <use xlink:href="#DejaVuSans-46"/>
     <use xlink:href="#DejaVuSans-4b" transform="translate(54.984375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(118.359375 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(179.890625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(241.171875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(280.375 0)"/>
     <use xlink:href="#DejaVuSans-1d" transform="translate(332.46875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(366.15625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(397.9375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(450.03125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(511.5625 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(573.09375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(625.1875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(656.96875 0)"/>
     <use xlink:href="#DejaVuSans-4b" transform="translate(696.171875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(759.546875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(821.078125 0)"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(852.859375 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(888.0625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(951.4375 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(990.640625 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(1054.015625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1092.921875 0)"/>
    </g>
   </g>
   <g id="text_28">
    <!-- Without mask (forbidden) -->
    <g transform="translate(126.371419 65.889258) scale(0.11 -0.11)">
     <defs>
      <path id="DejaVuSans-3a" d="M 213 4666 
L 850 4666 
L 1831 722 
L 2809 4666 
L 3519 4666 
L 4500 722 
L 5478 4666 
L 6119 4666 
L 4947 0 
L 4153 0 
L 3169 4050 
L 2175 0 
L 1381 0 
L 213 4666 
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
      <path id="DejaVuSans-4e" d="M 581 4863 
L 1159 4863 
L 1159 1991 
L 2875 3500 
L 3609 3500 
L 1753 1863 
L 3688 0 
L 2938 0 
L 1159 1709 
L 1159 0 
L 581 0 
L 581 4863 
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
     <use xlink:href="#DejaVuSans-3a"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(96.671875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(124.453125 0)"/>
     <use xlink:href="#DejaVuSans-4b" transform="translate(163.65625 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(227.03125 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(288.21875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(351.59375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(390.796875 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(422.578125 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(519.984375 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(581.265625 0)"/>
     <use xlink:href="#DejaVuSans-4e" transform="translate(633.359375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(691.265625 0)"/>
     <use xlink:href="#DejaVuSans-b" transform="translate(723.046875 0)"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(762.0625 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(797.265625 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(858.453125 0)"/>
     <use xlink:href="#DejaVuSans-45" transform="translate(899.5625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(963.046875 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(990.828125 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(1054.3125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1117.796875 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1179.328125 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1242.703125 0)"/>
    </g>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_7">
    <path d="M 393.155901 342.332877 
L 663.59952 342.332877 
L 663.59952 71.889258 
L 393.155901 71.889258 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#p7ce7e7cea0)">
    <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAXgAAAF3CAYAAACvy1BzAAAFrElEQVR4nO3WPS4EUBhGYSMKe1CqJZagU1oIC9BrlTZgHyqtiEp0OoUoFCbjJ5KxCnOTc59nBW++4uRbPL4s11ts3Or7d/SEKS1/3H2E19XX6AlT2h49AID/IfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPELVzefs8esOUzo/2R0+Y0svH5+gJUzo52Bs9YUo+eIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAKIEHiBJ4gCiBB4gSeIAogQeIEniAqMXN09t69IgZnV3fj54wpYeL49ETYGN88ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8QJfAAUQIPECXwAFECDxAl8ABRAg8Qtdg9PF2PHjGj97ur0ROAOB88QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPECUwANECTxAlMADRAk8QJTAA0QJPEDUHwF5IJSPgCOpAAAAAElFTkSuQmCC" id="image2295cf99cd" transform="scale(1 -1) translate(0 -270)" x="393.12" y="-72.153698" width="270.72" height="270"/>
   </g>
   <g id="patch_8">
    <path d="M 460.766805 71.889258 
L 528.37771 71.889258 
L 528.37771 139.500163 
L 460.766805 139.500163 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="patch_9">
    <path d="M 528.37771 71.889258 
L 595.988615 71.889258 
L 595.988615 139.500163 
L 528.37771 139.500163 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="patch_10">
    <path d="M 595.988615 71.889258 
L 663.59952 71.889258 
L 663.59952 139.500163 
L 595.988615 139.500163 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="patch_11">
    <path d="M 528.37771 139.500163 
L 595.988615 139.500163 
L 595.988615 207.111068 
L 528.37771 207.111068 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="patch_12">
    <path d="M 595.988615 139.500163 
L 663.59952 139.500163 
L 663.59952 207.111068 
L 595.988615 207.111068 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="patch_13">
    <path d="M 595.988615 207.111068 
L 663.59952 207.111068 
L 663.59952 274.721973 
L 595.988615 274.721973 
z
" clip-path="url(#p7ce7e7cea0)" style="fill: url(#h36290a37ce)"/>
   </g>
   <g id="matplotlib.axis_3">
    <g id="xtick_5">
     <g id="line2d_9">
      <g>
       <use xlink:href="#mbb6309f273" x="426.961353" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_29">
      <!-- The -->
      <g transform="translate(417.197173 357.311237) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_10">
      <g>
       <use xlink:href="#mbb6309f273" x="494.572258" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_30">
      <!-- cat -->
      <g transform="translate(486.410149 357.310416) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_11">
      <g>
       <use xlink:href="#mbb6309f273" x="562.183163" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_31">
      <!-- sleeps -->
      <g transform="translate(545.461092 357.311237) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="xtick_8">
     <g id="line2d_12">
      <g>
       <use xlink:href="#mbb6309f273" x="629.794068" y="342.332877" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_32">
      <!-- well -->
      <g transform="translate(619.35313 357.311237) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-5a"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(81.78125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(143.3125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(171.09375 0)"/>
      </g>
     </g>
    </g>
    <g id="text_33">
     <!-- Keyⱼ -->
     <g transform="translate(518.437867 371.431354) scale(0.1 -0.1)">
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
       <use xlink:href="#m583309ef95" x="393.155901" y="105.69471" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_34">
      <!-- The -->
      <g transform="translate(366.627541 109.68389) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_14">
      <g>
       <use xlink:href="#m583309ef95" x="393.155901" y="173.305615" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_35">
      <!-- cat -->
      <g transform="translate(369.831682 177.294385) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-46"/>
       <use xlink:href="#DejaVuSans-44" transform="translate(54.984375 0)"/>
       <use xlink:href="#DejaVuSans-57" transform="translate(116.265625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_15">
      <g>
       <use xlink:href="#m583309ef95" x="393.155901" y="240.91652" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_36">
      <!-- sleeps -->
      <g transform="translate(352.71176 244.9057) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-56"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(52.09375 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(79.875 0)"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(141.40625 0)"/>
       <use xlink:href="#DejaVuSans-53" transform="translate(202.9375 0)"/>
       <use xlink:href="#DejaVuSans-56" transform="translate(266.421875 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_16">
      <g>
       <use xlink:href="#m583309ef95" x="393.155901" y="308.527425" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_37">
      <!-- well -->
      <g transform="translate(365.274026 312.516605) scale(0.105 -0.105)">
       <use xlink:href="#DejaVuSans-5a"/>
       <use xlink:href="#DejaVuSans-48" transform="translate(81.78125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(143.3125 0)"/>
       <use xlink:href="#DejaVuSans-4f" transform="translate(171.09375 0)"/>
      </g>
     </g>
    </g>
    <g id="text_38">
     <!-- Queryᵢ -->
     <g transform="translate(346.309416 223.200911) rotate(-90) scale(0.1 -0.1)">
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
    <path d="M 393.155901 342.332877 
L 393.155901 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_15">
    <path d="M 663.59952 342.332877 
L 663.59952 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_16">
    <path d="M 393.155901 342.332877 
L 663.59952 342.332877 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_17">
    <path d="M 393.155901 71.889258 
L 663.59952 71.889258 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_39">
    <!-- 1.00 -->
    <g style="fill: #ffffff" transform="translate(415.828541 108.292367) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-14"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_40">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(485.799836 108.422249) scale(0.105 -0.105)">
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
    <g style="fill: #888888" transform="translate(553.410741 108.422249) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_42">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(621.021646 108.422249) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_43">
    <!-- 0.28 -->
    <g style="fill: #222222" transform="translate(415.828541 175.903272) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1b" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_44">
    <!-- 0.72 -->
    <g style="fill: #ffffff" transform="translate(483.439445 175.903272) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_45">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(553.410741 176.033154) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_46">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(621.021646 176.033154) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_47">
    <!-- 0.53 -->
    <g style="fill: #ffffff" transform="translate(415.828541 243.514176) scale(0.1 -0.1)">
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
    <g style="fill: #222222" transform="translate(483.439445 243.514176) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_49">
    <!-- 0.32 -->
    <g style="fill: #222222" transform="translate(551.05035 243.514176) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_50">
    <!-- −∞ -->
    <g style="fill: #888888" transform="translate(621.021646 243.644059) scale(0.105 -0.105)">
     <use xlink:href="#DejaVuSans-c9c"/>
     <use xlink:href="#DejaVuSans-ca8" transform="translate(83.796875 0)"/>
    </g>
   </g>
   <g id="text_51">
    <!-- 0.17 -->
    <g style="fill: #222222" transform="translate(415.828541 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-14" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_52">
    <!-- 0.22 -->
    <g style="fill: #222222" transform="translate(483.439445 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_53">
    <!-- 0.27 -->
    <g style="fill: #222222" transform="translate(551.05035 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-15" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-1a" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_54">
    <!-- 0.34 -->
    <g style="fill: #222222" transform="translate(618.661255 311.125081) scale(0.1 -0.1)">
     <use xlink:href="#DejaVuSans-13"/>
     <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
     <use xlink:href="#DejaVuSans-16" transform="translate(95.40625 0)"/>
     <use xlink:href="#DejaVuSans-17" transform="translate(159.03125 0)"/>
    </g>
   </g>
   <g id="text_55">
    <!-- position i only sees j ≤ i -->
    <g style="fill: #555555" transform="translate(474.70607 44.844896) scale(0.09 -0.09)">
     <defs>
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
     <use xlink:href="#DejaVuSans-52" transform="translate(487.4375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(548.625 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(612 0)"/>
     <use xlink:href="#DejaVuSans-5c" transform="translate(639.78125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(698.96875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(730.75 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(782.84375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(844.375 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(905.90625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(958 0)"/>
     <use xlink:href="#DejaVuSans-4d" transform="translate(989.78125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1017.5625 0)"/>
     <use xlink:href="#DejaVuSans-cee" transform="translate(1049.34375 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1133.140625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1164.921875 0)"/>
    </g>
   </g>
   <g id="text_56">
    <!-- With causal mask (correct) -->
    <g transform="translate(454.675132 65.889258) scale(0.11 -0.11)">
     <use xlink:href="#DejaVuSans-3a"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(96.671875 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(124.453125 0)"/>
     <use xlink:href="#DejaVuSans-4b" transform="translate(163.65625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(227.03125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(258.8125 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(313.796875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(375.078125 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(438.453125 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(490.546875 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(551.828125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(579.609375 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(611.390625 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(708.796875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(770.078125 0)"/>
     <use xlink:href="#DejaVuSans-4e" transform="translate(822.171875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(880.078125 0)"/>
     <use xlink:href="#DejaVuSans-b" transform="translate(911.859375 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(950.875 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1005.859375 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(1067.046875 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(1106.40625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1145.3125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(1206.84375 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(1261.828125 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1301.03125 0)"/>
    </g>
   </g>
  </g>
  <g id="text_57">
   <!-- The decoder's self-attention: why masking is essential -->
   <g transform="translate(165.723002 16.698047) scale(0.125 -0.125)">
    <defs>
     <path id="DejaVuSans-a" d="M 1147 4666 
L 1147 2931 
L 616 2931 
L 616 4666 
L 1147 4666 
z
" transform="scale(0.015625)"/>
     <path id="DejaVuSans-10" d="M 313 2009 
L 1997 2009 
L 1997 1497 
L 313 1497 
L 313 2009 
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
    <use xlink:href="#DejaVuSans-37"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(61.078125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(124.453125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(185.984375 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(217.765625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(281.25 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(342.78125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(397.765625 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(458.953125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(522.4375 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(583.96875 0)"/>
    <use xlink:href="#DejaVuSans-a" transform="translate(625.078125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(652.5625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(704.65625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(736.4375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(788.53125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(850.0625 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(877.84375 0)"/>
    <use xlink:href="#DejaVuSans-10" transform="translate(907.578125 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(943.65625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1004.9375 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1044.140625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1083.34375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1144.875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1208.25 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1247.453125 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1275.234375 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1336.421875 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(1399.796875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1433.484375 0)"/>
    <use xlink:href="#DejaVuSans-5a" transform="translate(1465.265625 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1547.046875 0)"/>
    <use xlink:href="#DejaVuSans-5c" transform="translate(1610.421875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1669.609375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(1701.390625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1798.796875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1860.078125 0)"/>
    <use xlink:href="#DejaVuSans-4e" transform="translate(1912.171875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1970.078125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1997.859375 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(2061.234375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2124.71875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2156.5 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2184.28125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2236.375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2268.15625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2329.6875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2381.78125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2433.875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2495.40625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2558.78125 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2597.984375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2625.765625 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2687.046875 0)"/>
   </g>
  </g>
  <g id="text_58">
   <!-- (example: generating "The cat sleeps well") -->
   <g transform="translate(230.991049 33.696) scale(0.095 -0.095)">
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
     <path id="DejaVuSans-5" d="M 1147 4666 
L 1147 2931 
L 616 2931 
L 616 4666 
L 1147 4666 
z
M 2328 4666 
L 2328 2931 
L 1797 2931 
L 1797 4666 
L 2328 4666 
z
" transform="scale(0.015625)"/>
    </defs>
    <use xlink:href="#DejaVuSans-b"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(39.015625 0)"/>
    <use xlink:href="#DejaVuSans-5b" transform="translate(98.796875 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(157.984375 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(219.265625 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(316.671875 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(380.15625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(407.9375 0)"/>
    <use xlink:href="#DejaVuSans-1d" transform="translate(469.46875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(503.15625 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(534.9375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(598.421875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(659.953125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(723.328125 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(784.859375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(825.96875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(887.25 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(926.453125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(954.234375 0)"/>
    <use xlink:href="#DejaVuSans-4a" transform="translate(1017.609375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1081.09375 0)"/>
    <use xlink:href="#DejaVuSans-5" transform="translate(1112.875 0)"/>
    <use xlink:href="#DejaVuSans-37" transform="translate(1158.875 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1219.953125 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1283.328125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1344.859375 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1376.640625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1431.625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(1492.90625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1532.109375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1563.890625 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(1615.984375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1643.765625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1705.296875 0)"/>
    <use xlink:href="#DejaVuSans-53" transform="translate(1766.828125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1830.3125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1882.40625 0)"/>
    <use xlink:href="#DejaVuSans-5a" transform="translate(1914.1875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1995.96875 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2057.5 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2085.28125 0)"/>
    <use xlink:href="#DejaVuSans-5" transform="translate(2113.0625 0)"/>
    <use xlink:href="#DejaVuSans-c" transform="translate(2159.0625 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="pb6d38f8c11">
   <rect x="61.644141" y="71.889258" width="270.443619" height="270.443619"/>
  </clipPath>
  <clipPath id="p7ce7e7cea0">
   <rect x="393.155901" y="71.889258" width="270.443619" height="270.443619"/>
  </clipPath>
 </defs>
 <defs>
  <pattern id="h36290a37ce" patternUnits="userSpaceOnUse" x="0" y="0" width="72" height="72">
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

## Putting it all together: the full architecture

The encoder (6 identical layers) transforms the source sentence into continuous representations. Each layer has two sublayers: multi-head self-attention, then a feed-forward network, each wrapped in a residual connection and layer normalization (`LayerNorm(x + Sublayer(x))`).

The decoder (also 6 layers) does the same thing, with one extra sublayer: **encoder-decoder attention**, where the queries come from the decoder and the keys/values from the encoder's output. The logic: the decoder *asks the question* ("what, in the source sentence, is relevant right now?"), and the encoder *supplies the information bank* to search for the answer.

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
<text style="font-weight: 700; font-size: 12px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #3b6ea5" x="129.632727" y="222.730254" transform="rotate(-0 129.632727 222.730254)">ENCODER</text>
</g>
<g id="patch_21">
<path d="M 129.632727 474.428606 Q 129.632727 462.662301 129.632727 452.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 127.632727 456.237636 L 129.632727 452.237636 L 131.632727 456.237636 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_8">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="129.632727" y="489.134219" transform="rotate(-0 129.632727 489.134219)">Source sequence (e.g. English)</text>
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
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'" transform="translate(412.31549 479.616078)">Output generated so far</text>
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
<text style="font-weight: 700; font-size: 12px; font-family: 'DejaVu Sans'; text-anchor: middle; fill: #c0504d" x="454.221818" y="137.599576" transform="rotate(-0 454.221818 137.599576)">DECODER</text>
</g>
<g id="patch_30">
<path d="M 454.221818 120.777577 Q 454.221818 112.187971 454.221818 104.940006 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
<path d="M 452.221818 108.940006 L 454.221818 104.940006 L 456.221818 108.940006 " style="fill: none; stroke: #555555; stroke-width: 1.2; stroke-linecap: round"/>
</g>
<g id="text_20">
<text style="font-size: 9px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="88.892969" transform="rotate(-0 454.221818 88.892969)">Linear + Softmax</text>
</g>
<g id="text_21">
<text style="font-style: italic; font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="454.221818" y="61.363147" transform="rotate(-0 454.221818 61.363147)">Probabilities over the next token</text>
</g>
<g id="text_22">
<text style="font-size: 13px; font-family: 'DejaVu Sans'; text-anchor: middle" x="320.4" y="17.597969" transform="rotate(-0 320.4 17.597969)">Simplified overview of the Transformer architecture</text>
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

## Feed-forward and embeddings: the details that matter

Each layer also contains a small feed-forward network, applied separately to each position:

**FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**

Its role is complementary to attention: attention **communicates** between positions (a weighted average is still a linear operation), the FFN **computes** within a position, with a real non-linearity (ReLU). It's the only place in a layer where the model can create new features rather than simply recombining ones that already exist.

Another nice detail: the paper **shares the same weight matrix** between the two embedding layers and the final pre-softmax transformation, a single "lookup table" reused in both directions (token → vector, and vector → token).

## Positional encoding: injecting order without recurrence

A subtle problem: with no RNN and no convolution, **nothing** in the architecture knows the order of the tokens. Self-attention treats the sequence as a set: permuting the words would give exactly the same compatibility scores between the same pairs.

The solution: add a position vector directly to the embedding, built from sines and cosines of different frequencies:

**PE(pos, 2i) = sin(pos / 10000^(2i/d_model))**
**PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))**

Why not just use the raw position number (0, 1, 2...) directly? Two problems: the magnitude grows without bound (a position of 499 would swamp an embedding of magnitude ~1), and a single scalar can't richly fill a 512-dimensional vector. Sinusoids, on the other hand, fill each dimension at a different frequency, and have a useful property: PE(pos+k) can be expressed as a linear function of PE(pos), which could make it easier to learn relative positions.

<div markdown="0">
<svg style="max-width:100%;height:auto;display:block;margin:1.5em auto;" xmlns:xlink="http://www.w3.org/1999/xlink" width="734.4pt" height="316.8pt" viewBox="0 0 734.4 316.8" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-08-27T09:27:54.778851</dc:date>
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
L 347.152029 51.694206 
L 37.726099 51.694206 
z
" style="fill: #ffffff"/>
   </g>
   <g clip-path="url(#paa7d9728fd)">
    <image xlink:href="data:image/png;base64,
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" id="imagea8a0905def" transform="scale(1 -1) translate(0 -229.68)" x="37.44" y="-51.84" width="309.6" height="229.68"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="m543fe8f9d4" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m543fe8f9d4" x="37.726099" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="89.297088" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="140.868076" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="192.439064" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="244.010052" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="295.581041" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="347.152029" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <path id="me46e9fde2c" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="51.694206" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- 0 -->
      <g transform="translate(24.363599 55.493034) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-13"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_9">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="87.648065" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- 10 -->
      <g transform="translate(18.001099 91.446893) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_10">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="123.601923" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- 20 -->
      <g transform="translate(18.001099 127.400751) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-15"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_11">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="159.555782" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_12">
      <!-- 30 -->
      <g transform="translate(18.001099 163.35461) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-16"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_12">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="195.50964" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_13">
      <!-- 40 -->
      <g transform="translate(18.001099 199.308468) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-17"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_13">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="231.463499" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_14">
      <!-- 50 -->
      <g transform="translate(18.001099 235.262327) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-18"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_14">
      <g>
       <use xlink:href="#me46e9fde2c" x="37.726099" y="267.417357" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_15">
      <!-- 60 -->
      <g transform="translate(18.001099 271.216185) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-19"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(63.625 0)"/>
      </g>
     </g>
    </g>
    <g id="text_16">
     <!-- dimension i -->
     <g transform="translate(11.358521 198.51335) rotate(-90) scale(0.11 -0.11)">
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
L 37.726099 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 347.152029 281.798901 
L 347.152029 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 37.726099 281.798901 
L 347.152029 281.798901 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 37.726099 51.694206 
L 347.152029 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_17">
    <!-- Full positional encoding -->
    <g transform="translate(37.726099 31.891062) scale(0.115 -0.115)">
     <defs>
      <path id="DejaVuSans-29" d="M 628 4666 
L 3309 4666 
L 3309 4134 
L 1259 4134 
L 1259 2759 
L 3109 2759 
L 3109 2228 
L 1259 2228 
L 1259 0 
L 628 0 
L 628 4666 
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
      <path id="DejaVuSans-4f" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
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
     <use xlink:href="#DejaVuSans-29"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(52.046875 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(115.421875 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(143.203125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(170.984375 0)"/>
     <use xlink:href="#DejaVuSans-53" transform="translate(202.765625 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(266.25 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(327.4375 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(379.53125 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(407.3125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(446.515625 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(474.296875 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(535.484375 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(598.859375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(660.140625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(687.921875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(719.703125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(781.234375 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(844.609375 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(899.59375 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(960.78125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1024.265625 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1052.046875 0)"/>
     <use xlink:href="#DejaVuSans-4a" transform="translate(1115.421875 0)"/>
    </g>
    <!-- (d_model=64, positions 0 to 59) -->
    <g transform="translate(37.726099 45.694206) scale(0.115 -0.115)">
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
     <use xlink:href="#DejaVuSans-57" transform="translate(1313.875 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1353.078125 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(1414.265625 0)"/>
     <use xlink:href="#DejaVuSans-18" transform="translate(1446.046875 0)"/>
     <use xlink:href="#DejaVuSans-1c" transform="translate(1509.671875 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1573.296875 0)"/>
    </g>
   </g>
  </g>
  <g id="axes_2">
   <g id="patch_7">
    <path d="M 359.515009 281.798901 
L 368.797787 281.798901 
L 368.797787 51.694206 
L 359.515009 51.694206 
z
" style="fill: #ffffff"/>
   </g>
   <image xlink:href="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAA0AAAE/CAYAAACdLkluAAABrUlEQVR4nO2aQW4EIQwEDUv+ky/l/2+YIZfZ3F0rlRxk7i3jotvsoB1f3z87kmuN+cpqRNFMK2ilNV5ANE8EMfPF2hGfVdJAaIatDkLsybJRdRBzDlIpL4IgBtie1pMIYhIQWk8IOY0GEZW2EQVBeiptI3p9ku0xEGmNGw0LRPFovMj2aotEEOtIEOD76UhHkLh7IIj3ekZ8Imp6zzJBoDsXJTevMX+FNYi3yOoJVRKj0SAeEdCIeaIggAg6okFERCygiTXDe98DIhSNBvGulNeY0YDnhCpJY7lBPAuCICI4I04EYd0aY6f/6QBBxL7zokFEqFI0CC4SQcRdmR4FkfdecRuZjvBAAO+1I56lzogrL9pksJBKEATq6QIiNGE9EJuAQD3BSp4jSE/MEaVBaMn1Bos4I2oPFnNYarfGviwQpJLXkwjiJoY9EQTq6bZ6qg5C62nd2rCEPVXOkwiCRaM6CPD1CUGgStZTHYyG1JN5fZJzEm0kOqKjEfEPoqFdnzC5Uty97UF65KHzYqK0hlViILTtHQoCiSR6FETlw60OIj8qDwXRjvgT5Sv9Aoc7ThB+Bcs6AAAAAElFTkSuQmCC" id="image087d2fa964" transform="scale(1 -1) translate(0 -229.68)" x="359.28" y="-51.84" width="9.36" height="229.68"/>
   <g id="matplotlib.axis_3"/>
   <g id="matplotlib.axis_4">
    <g id="ytick_8">
     <g id="line2d_15">
      <defs>
       <path id="m640d637390" d="M 0 0 
L 3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m640d637390" x="368.797787" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m640d637390" x="368.797787" y="253.035814" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_19">
      <!-- −0.75 -->
      <g transform="translate(375.797787 256.454759) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="224.272727" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_20">
      <!-- −0.50 -->
      <g transform="translate(375.797787 227.691672) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="195.50964" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_21">
      <!-- −0.25 -->
      <g transform="translate(375.797787 198.928586) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="166.746553" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_22">
      <!-- 0.00 -->
      <g transform="translate(375.797787 170.165499) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="137.983467" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_23">
      <!-- 0.25 -->
      <g transform="translate(375.797787 141.402412) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="109.22038" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_24">
      <!-- 0.50 -->
      <g transform="translate(375.797787 112.639325) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="80.457293" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_25">
      <!-- 0.75 -->
      <g transform="translate(375.797787 83.876238) scale(0.09 -0.09)">
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
       <use xlink:href="#m640d637390" x="368.797787" y="51.694206" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_26">
      <!-- 1.00 -->
      <g transform="translate(375.797787 55.113152) scale(0.09 -0.09)">
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
L 368.797787 51.694206 
L 364.156398 51.694206 
L 359.515009 51.694206 
L 359.515009 281.798901 
z
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
  </g>
  <g id="axes_3">
   <g id="patch_9">
    <path d="M 462.02522 281.798901 
L 725.03726 281.798901 
L 725.03726 51.694206 
L 462.02522 51.694206 
z
" style="fill: #ffffff"/>
   </g>
   <g id="matplotlib.axis_5">
    <g id="xtick_8">
     <g id="line2d_24">
      <g>
       <use xlink:href="#m543fe8f9d4" x="462.02522" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="505.86056" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="549.6959" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="593.53124" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="637.36658" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="681.20192" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#m543fe8f9d4" x="725.03726" y="281.798901" style="stroke: #000000; stroke-width: 0.8"/>
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="271.340621" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_35">
      <!-- −1.00 -->
      <g transform="translate(424.379907 275.139449) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="245.192232" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_36">
      <!-- −0.75 -->
      <g transform="translate(424.379907 248.99106) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="219.043843" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_37">
      <!-- −0.50 -->
      <g transform="translate(424.379907 222.842671) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="192.895454" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_38">
      <!-- −0.25 -->
      <g transform="translate(424.379907 196.694283) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="166.747066" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_39">
      <!-- 0.00 -->
      <g transform="translate(432.759595 170.545894) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="140.598677" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_40">
      <!-- 0.25 -->
      <g transform="translate(432.759595 144.397505) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="114.450288" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_41">
      <!-- 0.50 -->
      <g transform="translate(432.759595 118.249116) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="88.301899" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_42">
      <!-- 0.75 -->
      <g transform="translate(432.759595 92.100727) scale(0.1 -0.1)">
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
       <use xlink:href="#me46e9fde2c" x="462.02522" y="62.153511" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_43">
      <!-- 1.00 -->
      <g transform="translate(432.759595 65.952339) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-14"/>
       <use xlink:href="#DejaVuSans-11" transform="translate(63.625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(95.40625 0)"/>
       <use xlink:href="#DejaVuSans-13" transform="translate(159.03125 0)"/>
      </g>
     </g>
    </g>
    <g id="text_44">
     <!-- value -->
     <g transform="translate(417.737329 181.770147) rotate(-90) scale(0.11 -0.11)">
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
      </defs>
      <use xlink:href="#DejaVuSans-59"/>
      <use xlink:href="#DejaVuSans-44" transform="translate(59.1875 0)"/>
      <use xlink:href="#DejaVuSans-4f" transform="translate(120.46875 0)"/>
      <use xlink:href="#DejaVuSans-58" transform="translate(148.25 0)"/>
      <use xlink:href="#DejaVuSans-48" transform="translate(211.625 0)"/>
     </g>
    </g>
   </g>
   <g id="line2d_40">
    <path d="M 462.02522 166.747066 
L 466.408754 78.734624 
L 470.792288 71.640415 
L 475.175822 151.986822 
L 479.559356 245.903729 
L 483.94289 267.044365 
L 488.326424 195.972126 
L 492.709958 98.030502 
L 497.093492 63.266569 
L 501.477026 123.642128 
L 505.86056 223.648168 
L 510.244094 271.339596 
L 514.627628 222.869135 
L 519.011162 122.800302 
L 523.394696 63.135921 
L 527.77823 98.731149 
L 532.161764 196.859897 
L 536.545298 267.303047 
L 540.928832 245.295492 
L 545.312366 151.070875 
L 549.6959 71.258876 
L 554.079434 79.238278 
L 558.462968 167.672856 
L 562.846502 255.256266 
L 567.230036 261.464726 
L 571.61357 180.590206 
L 575.997104 86.988366 
L 580.380638 66.716307 
L 584.764172 138.412066 
L 589.147706 236.158893 
L 593.53124 270.088806 
L 597.914774 209.006799 
L 602.298308 109.071389 
L 606.681842 62.162729 
L 611.065376 111.408427 
L 615.44891 211.532213 
L 619.832444 270.480742 
L 624.215978 234.057007 
L 628.599512 135.748822 
L 632.983046 65.94028 
L 637.36658 88.813031 
L 641.750114 183.337974 
L 646.133648 262.609313 
L 650.517182 253.745343 
L 654.900716 164.895558 
L 659.28425 77.748041 
L 663.667784 72.425816 
L 668.051318 153.822113 
L 672.434852 247.101552 
L 676.818386 266.503446 
L 681.20192 194.189784 
L 685.585454 96.645413 
L 689.968988 63.552178 
L 694.352522 125.335847 
L 698.736056 225.192799 
L 703.11959 271.315013 
L 707.503124 221.297939 
L 711.886658 121.127043 
L 716.270192 62.898986 
L 720.653726 100.148374 
" clip-path="url(#pebbc57b9c7)" style="fill: none; stroke: #3b6ea5; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_41">
    <path d="M 462.02522 62.153511 
L 466.408754 110.234927 
L 470.792288 210.273343 
L 475.175822 270.2939 
L 479.559356 235.113976 
L 483.94289 137.077829 
L 488.326424 66.319442 
L 492.709958 87.893749 
L 497.093492 181.965431 
L 501.477026 262.045419 
L 505.86056 254.50854 
L 510.244094 166.284166 
L 514.627628 78.48538 
L 519.011162 71.833981 
L 523.394696 152.445234 
L 527.77823 246.205525 
L 532.161764 266.912075 
L 536.545298 195.527377 
L 540.928832 97.682194 
L 545.312366 63.334935 
L 549.6959 124.064312 
L 554.079434 224.036016 
L 558.462968 271.336523 
L 562.846502 222.477965 
L 567.230036 122.380675 
L 571.61357 63.07364 
L 575.997104 99.083474 
L 580.380638 197.302902 
L 584.764172 267.429435 
L 589.147706 244.989062 
L 593.53124 150.613358 
L 597.914774 71.070911 
L 602.298308 79.492679 
L 606.681842 168.135728 
L 611.065376 255.502047 
L 615.44891 261.267446 
L 619.832444 180.131243 
L 624.215978 86.689689 
L 628.599512 66.852518 
L 632.983046 138.857933 
L 637.36658 236.504488 
L 641.750114 270.016391 
L 646.133648 208.582952 
L 650.517182 108.685792 
L 654.900716 62.1699 
L 659.28425 111.801771 
L 663.667784 211.950093 
L 668.051318 270.53896 
L 672.434852 233.702038 
L 676.818386 135.307023 
L 681.20192 65.817838 
L 685.585454 89.12252 
L 689.968988 183.794851 
L 694.352522 262.793527 
L 698.736056 253.487529 
L 703.11959 164.43275 
L 707.503124 77.505741 
L 711.886658 72.626795 
L 716.270192 154.281592 
L 720.653726 247.397087 
" clip-path="url(#pebbc57b9c7)" style="fill: none; stroke: #e08a2c; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_42">
    <path d="M 462.02522 166.747066 
L 466.408754 142.175835 
L 470.792288 118.979884 
L 475.175822 98.457514 
L 479.559356 81.757386 
L 483.94289 69.814224 
L 488.326424 63.2965 
L 492.709958 62.569018 
L 497.093492 67.672496 
L 501.477026 78.321288 
L 505.86056 93.919367 
L 510.244094 113.593694 
L 514.627628 136.243073 
L 519.011162 160.599794 
L 523.394696 185.300585 
L 527.77823 208.962915 
L 532.161764 230.262377 
L 536.545298 248.006817 
L 540.928832 261.20306 
L 545.312366 269.112496 
L 549.6959 271.292426 
L 554.079434 267.620837 
L 558.462968 258.303232 
L 562.846502 243.861127 
L 567.230036 225.102863 
L 571.61357 203.078361 
L 575.997104 179.020356 
L 580.380638 154.275402 
L 584.764172 130.2285 
L 589.147706 108.225583 
L 593.53124 89.498177 
L 597.914774 75.094478 
L 602.298308 65.820674 
L 606.681842 62.195832 
L 611.065376 64.422838 
L 615.44891 72.377044 
L 619.832444 85.613244 
L 624.215978 103.390593 
L 628.599512 124.714074 
L 632.983046 148.390188 
L 637.36658 173.093757 
L 641.750114 197.442094 
L 646.133648 220.072397 
L 650.517182 239.718022 
L 654.900716 255.279382 
L 659.28425 265.885491 
L 663.667784 270.942713 
L 668.051318 270.167989 
L 672.434852 263.604683 
L 676.818386 251.620149 
L 681.20192 234.885174 
L 685.585454 214.336435 
L 689.968988 191.124066 
L 694.352522 166.547289 
L 698.736056 141.981695 
L 703.11959 118.802245 
L 707.503124 98.30632 
L 711.886658 81.641099 
L 716.270192 69.739352 
L 720.653726 63.267235 
" clip-path="url(#pebbc57b9c7)" style="fill: none; stroke: #4c9a6f; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_43">
    <path d="M 462.02522 62.153511 
L 466.408754 65.080619 
L 470.792288 73.69811 
L 475.175822 87.523654 
L 479.559356 105.78342 
L 483.94289 127.455389 
L 488.326424 151.326555 
L 492.709958 176.060825 
L 497.093492 200.273793 
L 501.477026 222.610234 
L 505.86056 241.819951 
L 510.244094 256.827756 
L 514.627628 266.793646 
L 519.011162 271.159818 
L 523.394696 269.681893 
L 527.77823 262.442592 
L 532.161764 249.847108 
L 536.545298 232.600422 
L 540.928832 211.667851 
L 545.312366 188.221014 
L 549.6959 163.572257 
L 554.079434 139.101197 
L 558.462968 116.177507 
L 562.846502 96.084251 
L 567.230036 79.946071 
L 571.61357 68.666238 
L 575.997104 62.876097 
L 580.380638 62.899729 
L 584.764172 68.735811 
L 589.147706 80.05769 
L 593.53124 96.23167 
L 597.914774 116.352474 
L 602.298308 139.293919 
L 606.681842 163.771947 
L 611.065376 188.416496 
L 615.44891 211.848182 
L 619.832444 232.75551 
L 624.215978 249.968272 
L 628.599512 262.523051 
L 632.983046 269.717143 
L 637.36658 271.147886 
L 641.750114 266.7352 
L 646.133648 256.726068 
L 650.517182 241.680711 
L 654.900716 222.441237 
L 659.28425 200.084497 
L 663.667784 175.861825 
L 668.051318 151.12899 
L 672.434852 127.270316 
L 676.818386 105.621199 
L 681.20192 87.393364 
L 685.585454 73.607044 
L 689.968988 65.033873 
L 694.352522 62.153701 
L 698.736056 65.127736 
L 703.11959 73.789517 
L 707.503124 87.654234 
L 711.886658 105.945864 
L 716.270192 127.640604 
L 720.653726 151.524177 
" clip-path="url(#pebbc57b9c7)" style="fill: none; stroke: #c0504d; stroke-width: 1.6; stroke-linecap: square"/>
   </g>
   <g id="line2d_44">
    <path d="M 462.02522 166.747066 
L 725.03726 166.747066 
" clip-path="url(#pebbc57b9c7)" style="fill: none; stroke: #cccccc; stroke-width: 0.8; stroke-linecap: square"/>
   </g>
   <g id="patch_10">
    <path d="M 462.02522 281.798901 
L 462.02522 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_11">
    <path d="M 725.03726 281.798901 
L 725.03726 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_12">
    <path d="M 462.02522 281.798901 
L 725.03726 281.798901 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_13">
    <path d="M 462.02522 51.694206 
L 725.03726 51.694206 
" style="fill: none; stroke: #333333; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="text_45">
    <!-- A few individual dimensions -->
    <g transform="translate(462.02522 31.891062) scale(0.115 -0.115)">
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
      <path id="DejaVuSans-5a" d="M 269 3500 
L 844 3500 
L 1563 769 
L 2278 3500 
L 2956 3500 
L 3675 769 
L 4391 3500 
L 4966 3500 
L 4050 0 
L 3372 0 
L 2619 2869 
L 1863 0 
L 1184 0 
L 269 3500 
z
" transform="scale(0.015625)"/>
     </defs>
     <use xlink:href="#DejaVuSans-24"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(68.40625 0)"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(100.1875 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(135.390625 0)"/>
     <use xlink:href="#DejaVuSans-5a" transform="translate(196.921875 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(278.703125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(310.484375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(338.265625 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(401.640625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(465.125 0)"/>
     <use xlink:href="#DejaVuSans-59" transform="translate(492.90625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(552.09375 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(579.875 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(643.359375 0)"/>
     <use xlink:href="#DejaVuSans-44" transform="translate(706.734375 0)"/>
     <use xlink:href="#DejaVuSans-4f" transform="translate(768.015625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(795.796875 0)"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(827.578125 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(891.0625 0)"/>
     <use xlink:href="#DejaVuSans-50" transform="translate(918.84375 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(1016.25 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1077.78125 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1141.15625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(1193.25 0)"/>
     <use xlink:href="#DejaVuSans-52" transform="translate(1221.03125 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(1282.21875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1345.59375 0)"/>
    </g>
    <!-- (different frequencies) -->
    <g transform="translate(462.02522 45.694206) scale(0.115 -0.115)">
     <defs>
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
     <use xlink:href="#DejaVuSans-b"/>
     <use xlink:href="#DejaVuSans-47" transform="translate(39.015625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(102.5 0)"/>
     <use xlink:href="#DejaVuSans-13ae" transform="translate(130.28125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(199.171875 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(260.703125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(299.609375 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(361.140625 0)"/>
     <use xlink:href="#DejaVuSans-57" transform="translate(424.515625 0)"/>
     <use xlink:href="#DejaVuSans-3" transform="translate(463.71875 0)"/>
     <use xlink:href="#DejaVuSans-49" transform="translate(495.5 0)"/>
     <use xlink:href="#DejaVuSans-55" transform="translate(530.703125 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(569.609375 0)"/>
     <use xlink:href="#DejaVuSans-54" transform="translate(631.140625 0)"/>
     <use xlink:href="#DejaVuSans-58" transform="translate(694.625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(758 0)"/>
     <use xlink:href="#DejaVuSans-51" transform="translate(819.53125 0)"/>
     <use xlink:href="#DejaVuSans-46" transform="translate(882.90625 0)"/>
     <use xlink:href="#DejaVuSans-4c" transform="translate(937.890625 0)"/>
     <use xlink:href="#DejaVuSans-48" transform="translate(965.671875 0)"/>
     <use xlink:href="#DejaVuSans-56" transform="translate(1027.203125 0)"/>
     <use xlink:href="#DejaVuSans-c" transform="translate(1079.296875 0)"/>
    </g>
   </g>
   <g id="legend_1">
    <g id="patch_14">
     <path d="M 658.643979 112.897019 
L 718.73726 112.897019 
Q 720.53726 112.897019 720.53726 111.097019 
L 720.53726 57.994206 
Q 720.53726 56.194206 718.73726 56.194206 
L 658.643979 56.194206 
Q 656.843979 56.194206 656.843979 57.994206 
L 656.843979 111.097019 
Q 656.843979 112.897019 658.643979 112.897019 
z
" style="fill: #ffffff; opacity: 0.85; stroke: #cccccc; stroke-linejoin: miter"/>
    </g>
    <g id="line2d_45">
     <path d="M 660.443979 63.4828 
L 669.443979 63.4828 
L 678.443979 63.4828 
" style="fill: none; stroke: #3b6ea5; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_46">
     <!-- dim 0 -->
     <g transform="translate(685.643979 66.6328) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-13" transform="translate(220.453125 0)"/>
     </g>
    </g>
    <g id="line2d_46">
     <path d="M 660.443979 76.983503 
L 669.443979 76.983503 
L 678.443979 76.983503 
" style="fill: none; stroke: #e08a2c; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_47">
     <!-- dim 1 -->
     <g transform="translate(685.643979 80.133503) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(220.453125 0)"/>
     </g>
    </g>
    <g id="line2d_47">
     <path d="M 660.443979 90.484206 
L 669.443979 90.484206 
L 678.443979 90.484206 
" style="fill: none; stroke: #4c9a6f; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_48">
     <!-- dim 10 -->
     <g transform="translate(685.643979 93.634206) scale(0.09 -0.09)">
      <use xlink:href="#DejaVuSans-47"/>
      <use xlink:href="#DejaVuSans-4c" transform="translate(63.484375 0)"/>
      <use xlink:href="#DejaVuSans-50" transform="translate(91.265625 0)"/>
      <use xlink:href="#DejaVuSans-3" transform="translate(188.671875 0)"/>
      <use xlink:href="#DejaVuSans-14" transform="translate(220.453125 0)"/>
      <use xlink:href="#DejaVuSans-13" transform="translate(284.078125 0)"/>
     </g>
    </g>
    <g id="line2d_48">
     <path d="M 660.443979 103.984909 
L 669.443979 103.984909 
L 678.443979 103.984909 
" style="fill: none; stroke: #c0504d; stroke-width: 1.6; stroke-linecap: square"/>
    </g>
    <g id="text_49">
     <!-- dim 11 -->
     <g transform="translate(685.643979 107.134909) scale(0.09 -0.09)">
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
   <!-- Positional Encoding: each dimension oscillates at its own frequency -->
   <g transform="translate(147.526406 12.878209) scale(0.13 -0.13)">
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
    <use xlink:href="#DejaVuSans-1d" transform="translate(971.046875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1004.734375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1036.515625 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(1098.046875 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1159.328125 0)"/>
    <use xlink:href="#DejaVuSans-4b" transform="translate(1214.3125 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1277.6875 0)"/>
    <use xlink:href="#DejaVuSans-47" transform="translate(1309.46875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1372.953125 0)"/>
    <use xlink:href="#DejaVuSans-50" transform="translate(1400.734375 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(1498.140625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1559.671875 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1623.046875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(1675.140625 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1702.921875 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(1764.109375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(1827.484375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(1859.265625 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(1920.453125 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(1972.546875 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2027.53125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2055.3125 0)"/>
    <use xlink:href="#DejaVuSans-4f" transform="translate(2083.09375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2110.875 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2172.15625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2211.359375 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2272.890625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2324.984375 0)"/>
    <use xlink:href="#DejaVuSans-44" transform="translate(2356.765625 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2418.046875 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2457.25 0)"/>
    <use xlink:href="#DejaVuSans-4c" transform="translate(2489.03125 0)"/>
    <use xlink:href="#DejaVuSans-57" transform="translate(2516.8125 0)"/>
    <use xlink:href="#DejaVuSans-56" transform="translate(2556.015625 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2608.109375 0)"/>
    <use xlink:href="#DejaVuSans-52" transform="translate(2639.890625 0)"/>
    <use xlink:href="#DejaVuSans-5a" transform="translate(2701.078125 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(2782.859375 0)"/>
    <use xlink:href="#DejaVuSans-3" transform="translate(2846.234375 0)"/>
    <use xlink:href="#DejaVuSans-49" transform="translate(2878.015625 0)"/>
    <use xlink:href="#DejaVuSans-55" transform="translate(2913.21875 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(2952.125 0)"/>
    <use xlink:href="#DejaVuSans-54" transform="translate(3013.65625 0)"/>
    <use xlink:href="#DejaVuSans-58" transform="translate(3077.140625 0)"/>
    <use xlink:href="#DejaVuSans-48" transform="translate(3140.515625 0)"/>
    <use xlink:href="#DejaVuSans-51" transform="translate(3202.046875 0)"/>
    <use xlink:href="#DejaVuSans-46" transform="translate(3265.421875 0)"/>
    <use xlink:href="#DejaVuSans-5c" transform="translate(3320.40625 0)"/>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="paa7d9728fd">
   <rect x="37.726099" y="51.694206" width="309.42593" height="230.104694"/>
  </clipPath>
  <clipPath id="pebbc57b9c7">
   <rect x="462.02522" y="51.694206" width="263.01204" height="230.104694"/>
  </clipPath>
 </defs>
</svg>
</div>

## Why attention over recurrence or convolution?

The paper compares the three approaches on three criteria: complexity per layer, sequential operations required, and maximum path length between two positions.

| Layer type | Complexity | Sequential ops | Maximum path |
|---|---|---|---|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(logₖ n) |

Self-attention connects **every** position in a single operation, versus *n* sequential operations for an RNN, and several stacked layers for a CNN.

One caveat though: this isn't free. Self-attention's per-layer complexity is **quadratic** in *n* (because of the *n×n* matrix), versus linear for recurrence. For a sequence of 10,000 tokens with d=512, self-attention ends up costing roughly **20 times more** per layer than an RNN. This is precisely why the paper mentions, as a future direction, a "restricted self-attention" (limiting each position to a local neighborhood), the conceptual ancestor of a whole family of efficient attention methods that would come later (sparse attention, linear attention...).

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
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="203.147454" y="312.444562" transform="rotate(-0 203.147454 312.444562)">sequence length n</text>
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
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="14.798438" y="174.728" transform="rotate(-90 14.798438 174.728)">maximum path length</text>
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
<text style="font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="203.147454" y="59.688" transform="rotate(-0 203.147454 59.688)">Linear scale</text>
</g>
<g id="legend_1">
<g id="patch_7">
<path d="M 194.030298 109.917219 L 359.741783 109.917219 Q 361.441783 109.917219 361.441783 108.217219 L 361.441783 71.638 Q 361.441783 69.938 359.741783 69.938 L 194.030298 69.938 Q 192.330298 69.938 192.330298 71.638 L 192.330298 108.217219 Q 192.330298 109.917219 194.030298 109.917219 z " style="fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
</g>
<g id="line2d_20">
<path d="M 195.730298 76.821672 L 204.230298 76.821672 L 212.730298 76.821672 " style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_20">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="79.796672" transform="rotate(-0 219.530298 79.796672)">Recurrent: O(n)</text>
</g>
<g id="line2d_21">
<path d="M 195.730298 89.298078 L 204.230298 89.298078 L 212.730298 89.298078 " style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_21">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="92.273078" transform="rotate(-0 219.530298 92.273078)">Convolutional: O(logₖ n), k=3</text>
</g>
<g id="line2d_22">
<path d="M 195.730298 101.774484 L 204.230298 101.774484 L 212.730298 101.774484 " style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="text_22">
<text style="font-size: 8.5px; font-family: 'DejaVu Sans'; text-anchor: start" x="219.530298" y="104.749484" transform="rotate(-0 219.530298 104.749484)">Self-Attention: O(1)</text>
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
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="578.688796" y="312.444562" transform="rotate(-0 578.688796 312.444562)">sequence length n</text>
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
<text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="385.46478" y="174.728" transform="rotate(-90 385.46478 174.728)">maximum path length (log scale)</text>
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
<text style="font-size: 10.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="578.688796" y="59.688" transform="rotate(-0 578.688796 59.688)">Logarithmic scale</text>
</g>
<g id="legend_2">
<g id="patch_13">
<path d="M 552.609375 279.768 L 735.633125 279.768 Q 737.233125 279.768 737.233125 278.168 L 737.233125 231.998 Q 737.233125 230.398 735.633125 230.398 L 552.609375 230.398 Q 551.009375 230.398 551.009375 231.998 L 551.009375 278.168 Q 551.009375 279.768 552.609375 279.768 z " style="fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
</g>
<g id="line2d_64">
<path d="M 554.209375 236.87675 L 562.209375 236.87675 L 570.209375 236.87675 " style="fill: none; stroke: #c0504d; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_38">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="239.67675" transform="rotate(-0 576.609375 239.67675)">Recurrent: O(n)</text>
</g>
<g id="line2d_65">
<path d="M 554.209375 248.61925 L 562.209375 248.61925 L 570.209375 248.61925 " style="fill: none; stroke: #4c9a6f; stroke-width: 2; stroke-linecap: square"/>
</g>
<g id="text_39">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="251.41925" transform="rotate(-0 576.609375 251.41925)">Convolutional: O(logₖ n)</text>
</g>
<g id="line2d_66">
<path d="M 554.209375 260.36175 L 562.209375 260.36175 L 570.209375 260.36175 " style="fill: none; stroke: #3b6ea5; stroke-width: 2.4; stroke-linecap: square"/>
</g>
<g id="text_40">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="263.16175" transform="rotate(-0 576.609375 263.16175)">Self-Attention: O(1)</text>
</g>
<g id="line2d_67">
<path d="M 554.209375 272.10425 L 562.209375 272.10425 L 570.209375 272.10425 " style="fill: none; stroke-dasharray: 6.66,2.88; stroke-dashoffset: 0; stroke: #e08a2c; stroke-width: 1.8"/>
</g>
<g id="text_41">
<text style="font-size: 8px; font-family: 'DejaVu Sans'; text-anchor: start" x="576.609375" y="274.90425" transform="rotate(-0 576.609375 274.90425)">Restricted Self-Attention: O(n/r), r=10</text>
</g>
</g>
</g>
<g id="text_42">
<text style="font-size: 11.5px; font-family: 'DejaVu Sans'; text-anchor: middle" x="374.033125" y="15.938203" transform="rotate(-0 374.033125 15.938203)">Maximum path length between two positions, by layer type (Table 1)</text>
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

## The results, in brief

On WMT 2014 English-German, the "big" model reaches **28.4 BLEU**, more than 2 points above the best previous result, including ensembles. On English-French, **41.0 BLEU**, at under a quarter of the training cost of the previous state of the art. Even the "base" model (12 hours on 8 GPUs) beats everything that came before it.

Two training details worth a closer look:

- **Learning rate warmup**: a linear increase over 4000 steps, then decay proportional to 1/√step. With such a deep network (6+6 layers, full of residual connections and layer norms), a high learning rate from the start would produce unstable updates that cascade and amplify across layers.
- **Label smoothing** (ε=0.1): deliberately blunts the training targets. This hurts perplexity (the model becomes "less confident") but improves accuracy and BLEU, a deliberate trade-off against overfitting.

## Key takeaways

- The RNN problem isn't depth (everyone has that): it's the **horizontal sequential dependency** that prevents parallelization within a sequence.
- Attention is a weighted average: a query interrogates keys, softmax turns the scores into weights, and the values are combined accordingly.
- Scaling by √dₖ keeps the softmax from saturating as the dimension grows.
- Multi-head means several parallel "viewpoints" on the same sequence, with a genuine trade-off between number of heads and capacity per head.
- Causal masking keeps the decoder honest during parallelized training.
- All of this comes at a price: quadratic complexity in sequence length, still an active research topic today.

**Going further:** I implemented the full Transformer from scratch in PyTorch (encoder-decoder, multi-head attention, positional encoding, causal masking) to check that I truly understand every step, not just the formula. The code is available on [GitHub](https://github.com/LaryConseiga/Architecture-Transformer-from-Scratch){:target="_blank"}.
