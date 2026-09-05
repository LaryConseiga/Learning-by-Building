---
title: "Scaling Laws: What This Means Going Forward (Part 4/4)"
date: 2026-09-05 10:00:01 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, week-4]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/les-scaling-laws-ce-que-ca-change-pour-la-suite-partie-4-4/' | relative_url }})*

Part 4, the last of this series on Scaling Laws. In [Part 3]({{ '/posts/scaling-laws-how-to-spend-your-compute-budget-part-3-4/' | relative_url }}), we saw how to split a fixed compute budget. Today we close things out with the paper's discussion.

## Closing the paper, not the topic

We've spent three articles taking apart the power laws from Kaplan et al. In this last part, we step back a bit. What do the authors themselves think of their results, where are the limits, and above all, what did this change for the research that followed?

## A law with no theory behind it

Here's a rather honest admission from the authors. They compare their scaling laws to the ideal gas law in physics. It's a universal macroscopic law that describes the behavior of a gas very well without depending on the precise microscopic details of each molecule.

The problem is that when the ideal gas law was discovered, there wasn't yet an underlying theory to explain it from first principles. That theory, statistical mechanics, came later.

That's exactly the position the authors of this paper find themselves in. They observe a solid, reproducible phenomenon, measured across seven orders of magnitude. But they don't have a fundamental explanation for why it works this way. We're observing the thermodynamics of deep learning, without yet having its statistical mechanics.

## The paper's most visionary idea: "more is different"

This is probably the most important passage in the whole discussion, and yet it fits in a few sentences.

The authors note that a smooth, continuous improvement in loss can mask qualitative changes in capability. They use the example of global economic growth. Seen from a distance, it looks smooth and continuous. But that smooth curve reveals nothing about the specific technological breakthroughs actually driving it underneath, like the arrival of the internet or of electricity.

Apply the same idea to a language model. Its loss drops in a steady, predictable way as it grows. But that continuous drop can very well hide the sudden appearance of entirely new capabilities that simply didn't exist in smaller models.

This intuition would become central a few years later under the name emergent capabilities, a concept popularized in particular by work studying at what size certain models suddenly become able to solve reasoning problems, do reliable arithmetic, or follow complex instructions, while slightly smaller models were unable to.

This is an excellent candidate for a future reading session, where we could connect this 2020 paper to more recent work on emergent capabilities, as well as to Chinchilla, which refined the optimal relationship between model size and dataset size that we saw in Part 2.

## Big models, big deal

The authors' most direct practical conclusion can be summed up in one sentence: large models turn out to be far more sample efficient than previously thought.

With the hindsight we have today, this result was an extremely strong signal. It directly motivated the direction taken afterward with models like GPT-3, and then the whole generation of large language models we know now. The idea that "bigger is better, and predictably so" served as solid empirical justification for massive investment in ever larger models.

## A contradiction at very large scale

A more technical but interesting point: the authors themselves note that their equations, pushed far beyond the scales they tested, eventually contradict each other.

![Intersection between L(C_min) and L(D(C)) at very large scale](/assets/img/posts/scaling-laws-extrapolation-limit.png)
_Figure 15, page 17. The intersection between the two curves marks the point where the paper's predictions start to contradict each other._

Basically, the amount of data needed for compute-optimal training grows too slowly compared to what would be needed to avoid overfitting at very large scale. The two trends eventually cross at absolutely gigantic scales, around $10^{12}$ parameters and $10^{12}$ tokens.

The authors cautiously interpret this crossing point as a possible estimate of the ultimate performance limit of a language model on natural text, perhaps linked to the intrinsic entropy of language itself. But they stay honest about the fact that this extrapolation is highly uncertain, and that their scaling laws probably have to bend well before reaching that point.

## The limits the authors acknowledge themselves

The paper ends with a fairly transparent list of caveats. There's no solid underlying theory to explain why these power laws appear. The regime of very small datasets hasn't been well explored. The behavior of the critical batch size remains uncertain far from the tested range. And above all, none of these relationships have been verified at scales much larger than those studied in the paper, which leaves the door open to surprises.

This scientific honesty is part of what makes this paper solid: the authors never claim to have found a universal, definitive law, only a very robust empirical trend within a given range.

## Key takeaways from this whole series

We saw that a language model's performance depends mainly on three scale factors, model size, data size, and compute, and that this dependence follows remarkably regular power laws.

We saw that the model's precise shape matters little, but that the type of architecture matters a lot, especially for handling long contexts.

We saw that the amount of data needed grows more slowly than the model size, which has concrete implications for avoiding overfitting.

We saw that, facing a fixed compute budget, the best strategy is to heavily favor model size over training time, which justifies the "big model, early stop" strategy over "small model, full training."

And finally, we saw that seemingly smooth, predictable progress can hide qualitative jumps in capability, an idea that opened the door to an entire line of research on the emergent capabilities of large models.
