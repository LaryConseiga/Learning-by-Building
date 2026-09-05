---
title: "Scaling Laws: The Right Balance Between Size and Data (Part 2/4)"
date: 2026-09-03 10:00:01 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, week-4]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/les-scaling-laws-le-juste-equilibre-entre-taille-et-donnees-partie-2-4/' | relative_url }})*

Part 2 of this series on Scaling Laws. In [Part 1]({{ '/posts/scaling-laws-why-size-matters-more-than-skill-part-1-4/' | relative_url }}), we saw the three basic laws and why the model's shape matters less than its scale. Today, we look at what happens when two of these factors move at the same time.

## The problem we haven't solved yet

In Part 1, we saw three separate laws: L(N), L(D), and L(C). Each one assumes the other two factors are "infinite," meaning they aren't a bottleneck.

But in real life, you never have an infinite data budget while you grow your model. So the real question becomes: **what happens when N and D move at the same time?**

That's the subject of this second part, and it's where we run into one of the most practically useful results in the paper.

## The classic trap: overfitting

You've probably already heard of overfitting. A model too big for the amount of data it sees ends up memorizing that data by heart, rather than learning general rules. The result: it's excellent on its training data, but terrible on anything it has never seen.

It's a bit like a student who memorizes the exact answers to one specific practice exam, without understanding the reasoning behind them. On the day of the real exam, with slightly different questions, they're lost.

## The equation that unifies everything

The authors propose a formula that captures how the loss behaves when N and D vary together:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

![Loss as a function of N for different values of D, and the extent of overfitting as a function of the D/N ratio](/assets/img/posts/scaling-laws-overfitting-regimes.png)
_Figure 9, page 11. You can clearly see the two regimes: on the left, for a fixed D, performance stalls as soon as N gets too large (a sign of overfitting). On the right, the extent of overfitting follows a single, predictable curve._

This formula didn't come out of nowhere. It satisfies three common-sense rules.

First, if you change how you split your text into tokens, the loss should just shift globally, without the shape of the equation changing.

Second, if you have infinite data (D tends to infinity), the equation should reduce to L(N) alone. And if you have an infinitely large model (N tends to infinity), it should reduce to L(D) alone. That matches what we saw in Part 1.

Third, the equation has to remain well behaved mathematically even in extreme cases, which is a more technical principle but one that guarantees the model's consistency.

## The practical rule to remember

From this equation follows a handy rule for figuring out how much data you need for a given model, if you want to avoid overfitting:

$$D \gtrsim (5 \times 10^3) \, N^{0.74}$$

Here's the counterintuitive part. You might think that if you multiply your model's size by 8, you also need 8 times more data to stay safe. That's not the case. You only need about **5 times more data**, not 8 times more.

The data requirement grows **more slowly** than the model size.

## Why this relationship is sublinear

Here's the key intuition. N and D don't bring the same kind of value to the model.

Part of a large model's extra capacity goes toward better **generalizing** the regularities already present in the existing data. That part doesn't need new data to be useful.

Another part of that extra capacity does go toward actually absorbing new information, and that's the part that would need more data so it doesn't end up memorized rather than generalized.

Result: doubling the model's size doesn't double the risk of pure memorization. Part of that extra size gets absorbed by better compression of the patterns already present in the data. That's why the relationship between N and D is sublinear.

Picture two students studying for the same course. The first has an average memory and has to retain everything word for word. The second has excellent analytical skills and manages to extract the big principles of the course, which lets them answer correctly even on questions they've never seen in that exact form. The second student "generalizes" better with the same amount of material studied. That's roughly what happens when you increase N: you're giving the model better generalization ability, not just more raw memory.

## What this changes in practice

This relationship offers a very concrete guide. If you know how many parameters your model will have, you can directly estimate the minimum number of tokens you need so you don't waste its capacity on useless memorization.

This is a result that, a few years later, directly inspired work like Chinchilla (DeepMind, 2022), which further refined this relationship between model size and dataset size. We'll probably come back to that in a future reading session.

## Key takeaways from this part

Overfitting depends on a precise combination of N and D, not on either one in isolation.

The amount of data needed grows more slowly than the model size, in an approximate ratio of $N^{0.74}$.

This sublinearity comes from the fact that part of a large model's capacity goes toward better generalizing existing data, not just absorbing more of it.

In [**Part 3**]({{ '/posts/scaling-laws-how-to-spend-your-compute-budget-part-3-4/' | relative_url }}), we tackle the most concrete and most cited part of the paper: how to split a fixed compute budget between model size, batch size, and number of training steps. It's this result that historically justified the race toward ever bigger models.

## Section 5: training time also enters the equation

Before we get to compute allocation in Part 3, there's one more ingredient to lay down: training time, or more precisely the number of training steps (S).

### The batch size problem

To properly compare models trained at different speeds, you first need to settle a technical detail. Most of the models studied in this paper weren't trained with an optimal batch size. There's a so called "critical" batch size, $B_{crit}$, below which increasing the batch lets you go faster with almost no loss of efficiency, and above which the speed gains level off.

$$B_{crit}(L) \approx \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

Interesting point: $B_{crit}$ only depends on the current loss, not on the model size. The further along the model is (lower loss), the bigger a batch you can afford.

![Critical batch size as a function of the loss](/assets/img/posts/scaling-laws-critical-batch-size.png)
_Figure 10, page 12. The critical batch size follows a very clean power law as a function of the loss._

### A universal measure of time

To compare runs trained with different batch sizes on equal footing, the authors define $S_{min}$, a kind of "equivalent step count" as if the model had been trained at the best theoretically possible speed.

The combined equation then becomes:

$$L(N, S_{min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{min}}\right)^{\alpha_S}$$

![Learning curves of all the models, superimposed once adjusted with S_min](/assets/img/posts/scaling-laws-loss-vs-smin.png)
_Figure 4 (right), page 5. The learning curves of all the models, once adjusted with $S_{min}$, line up according to this same formula._

Interesting remark here: unlike the L(N, D) equation seen above, this is a simple **addition** of the two terms, not a single power enveloping both of them. That reflects the fact that model size and training time play fairly independent roles. Having a small model and having taken few steps are two separate flaws that add up, with no multiplying effect between them, unlike the N and D pair, where a small D can completely cancel out the benefit of a large N.

This piece will be essential for Part 3, where we'll finally answer the question that kicked off the whole race toward bigger models: if I have a fixed compute budget, how do I best split it up?
