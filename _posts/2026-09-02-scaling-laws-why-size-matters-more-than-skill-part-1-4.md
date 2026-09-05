---
title: "Scaling Laws: Why Size Matters More Than Skill (Part 1/4)"
date: 2026-09-02 10:00:01 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, week-4]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/les-scaling-laws-pourquoi-la-taille-compte-plus-que-le-talent-partie-1-4/' | relative_url }})*

## The paper that (quietly) changed everything

If you had to point to ONE paper that explains why we went from "small, carefully tuned models" to "monsters with billions of parameters," it would probably be this one: **Scaling Laws for Neural Language Models**, published by a team at OpenAI (Kaplan et al.) in January 2020.

No revolutionary new architecture in it. No magic training trick. Just... curves. Lots of curves. And a conclusion that ended up justifying the race toward giant models we see today with LLMs.

In this 4-part series, we're going to take this paper apart together, without unnecessary jargon. Today: the three basic laws, and why they're more surprising than they look.

![Three power laws: loss as a function of compute, dataset size, and number of parameters](/assets/img/posts/scaling-laws-three-power-laws.png)
_Figure 1, page 3. The one chart to remember from the whole paper. Three curves, three power laws, one message._

## The setup: three ingredients, one recipe

To train a language model, you have three main levers:

- **N** : the size of the model (the number of parameters)
- **D** : the amount of data you train it on
- **C** : the amount of compute you put into training

Natural question: if I increase one of these three things, what happens to the model's performance?

The paper's answer, in one sentence: **it follows a power law, with remarkable regularity**, and this holds across almost **seven orders of magnitude** (we're talking about models ranging from 768 parameters to 1.5 billion).

Concretely:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C_{min}) = \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_C^{min}}, \quad \alpha_C^{min} \approx 0.050$$

Don't panic at the formulas. The message behind them is simple: **the more you increase N, D, or C, the more the loss (the model's error) drops, and it drops in a perfectly predictable way**, like a well behaved curve that never throws a tantrum.

## The analogy that helps: the cake recipe that never fails

Imagine a cake recipe where, no matter how much batter you prepare, the result is **always proportionally just as good**. You can predict in advance, with a simple mathematical rule, exactly how good your cake will turn out based on how much flour you use.

That's roughly what this paper says about neural networks: performance isn't some artisanal mystery where you need "the touch." It's predictable. **You can draw the curve before you're even done baking.**

## The real scoop: the model's shape barely matters

Here's the result that breaks the most common intuitions: for an equal number of parameters, **how you organize your model** (deep and thin? wide and short?) **barely changes performance**.

![Loss variation with the depth/width ratio, the feed-forward ratio, and the attention head dimension](/assets/img/posts/scaling-laws-depth-width-ratio.png)
_Figure 5, page 8. The depth/width ratio can vary by a factor of 40 while changing the loss by only a few percent._

In other words: an architect who spends weeks fine-tuning the number of layers, the exact width, the number of attention heads... is probably wasting their time, compared to someone who simply increases the model's total size. **What matters is scale. Not fine-grained design.**

It's a bit like discovering that no matter the shape of your water tank (square, round, tall, wide), what determines how much water it holds is its total volume. Shape is almost a cosmetic detail.

## Transformer vs LSTM: where architecture *does* matter

One important nuance: saying that the "internal shape" of a Transformer barely matters doesn't mean the *type* of architecture doesn't matter at all.

![Loss comparison between LSTM and Transformer by model size and position in context](/assets/img/posts/scaling-laws-lstm-vs-transformer.png)
_Figure 7, page 9. A direct comparison between LSTM and Transformer._

The paper shows that LSTMs (the ancestor of Transformers for text) keep up with Transformers pretty well... on the first tokens of a text. But as soon as the context gets longer, LSTMs fall behind, while Transformers keep improving.

**Why?** An LSTM has to carry information through a long chain of steps, a bit like a game of telephone where the message gets diluted at each relay. A Transformer, on the other hand, can "look" directly at any word in the text, no matter how far away, with no relay and no dilution.

This is, among other things, what explains why the Transformer architecture became the undisputed standard for modern NLP.

## Bonus: generalization comes "for free"

One last nice point from this section: the authors tested their models (trained only on a dataset called WebText2) on other types of text, like Wikipedia, books, or Common Crawl.

![Loss generalization across other text distributions (Wikipedia, books, Common Crawl)](/assets/img/posts/scaling-laws-generalization.png)
_Figure 8, page 10. Performance on other text distributions tracks performance on the training data, with a constant offset._

Result: the model improves on these other texts **at the same rate** as it improves on its own training data, with just a small fixed offset. In other words, improving a model on its training data also improves it everywhere else, with no extra effort.

## Key takeaways from this part

- Three levers (N, D, C), three power laws, predictable across enormous orders of magnitude
- The model's precise shape barely matters; scale is what dominates
- The *type* of architecture (attention vs. recurrence), on the other hand, matters a lot, especially on long contexts
- A model that improves on its training data generalizes "for free" elsewhere

In [**Part 2**]({{ '/posts/scaling-laws-the-right-balance-between-size-and-data-part-2-4/' | relative_url }}), we'll look at what happens when you vary N and D *at the same time*, and why the relationship between the two isn't the one you'd naturally expect (spoiler: doubling your model doesn't require doubling your data).
