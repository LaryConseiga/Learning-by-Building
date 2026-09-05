---
title: "Scaling Laws: How to Spend Your Compute Budget (Part 3/4)"
date: 2026-09-04 10:00:01 +0000
categories: [Deep Learning, Scaling Laws]
tags: [scaling-laws, llm, transformers, deep-learning, week-4]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/les-scaling-laws-comment-depenser-son-budget-de-calcul-partie-3-4/' | relative_url }})*

Part 3 of this series on Scaling Laws. In [Part 2]({{ '/posts/scaling-laws-the-right-balance-between-size-and-data-part-2-4/' | relative_url }}), we saw how N and D interact, and introduced the notion of $S_{min}$. Today, we answer the question that kicked off the race toward bigger models.

## The million dollar question (literally)

We finally get there. This is the part of the paper that has most influenced how the industry has trained its models since 2020.

Imagine you have a fixed compute budget. Say, the equivalent of X GPU hours. How do you best split it between three things: your model's size, your batch size, and the number of training steps?

Before this paper, the dominant intuition was fairly simple: you pick a reasonably sized model, and you train it to convergence, meaning until the loss stops dropping. This paper shows that intuition is actually pretty bad.

## The central result

By combining everything we saw in the previous parts, in particular the L(N, $S_{min}$) equation from the end of Part 2, the authors arrive at the following formula for optimal compute:

$$L(C_{min}) = \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_C^{min}}, \quad \alpha_C^{min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.050$$

This formula for $\alpha_C^{min}$ is rather elegant. It's a kind of average that combines how model size, batch size, and number of steps each contribute to using the available compute.

![Loss as a function of optimal compute, adjusted for ideal training speed](/assets/img/posts/scaling-laws-optimal-compute-fit.png)
_Figure 13, page 15. Once adjusted for training at the optimal speed, the performance-versus-compute curve becomes even cleaner than in Figure 1 from Part 1._

## The real scoop: where your extra budget should go

Here's the part that truly changed things in the industry. When your compute budget grows, how should that growth be split?

$$N \propto C^{0.73}, \qquad B \propto C^{0.24}, \qquad S \propto C^{0.03}$$

![Growth of the optimal model size and number of steps as a function of compute](/assets/img/posts/scaling-laws-optimal-allocation-growth.png)
_Figure 14, page 16. The optimal model size grows very quickly with available compute, while the number of steps stays almost constant._

To put it concretely: if your compute budget is multiplied by a **billion**, your model's size should grow by roughly **a million times**. Batch size should grow by roughly 100 times. And the number of sequential training steps? Barely under 10 times.

![Split of a compute increase between model size, batch size, and sequential steps](/assets/img/posts/scaling-laws-compute-allocation.png)
_Figure 3, page 4. This figure sums it all up in one image: over a billion-fold increase in compute, almost all of it should go into growing the model, a small part into increasing the batch, and almost none into increasing the number of sequential steps._

## Why this is excellent news in practice

There's a crucial distinction between the compute available and the time actually needed to use it.

The number of sequential steps is the number of iterations that must happen one after another, in order. You can't compute step 501 before finishing step 500, since each step updates the weights that will be used in the next one.

Batch size, on the other hand, parallelizes easily. If you have a thousand machines available, you can hand each of them a different slice of the same batch and compute everything at once.

Let's go back to the moving-day image. If the load to move increases, you have two options. You can make more trips one after another, which takes more time. Or you can bring more trucks at once, which takes more resources but not more time.

This paper's result says the best strategy is almost exclusively the second option. That means you can train gigantic models without training time exploding unmanageably, as long as you have enough machines available in parallel.

## Inefficient convergence: the paper's most counterintuitive conclusion

Here's where it all comes together. Since N has to grow extremely fast with compute, and S (the number of steps) has to grow very slowly, the direct consequence is that a large model with a given compute budget will necessarily take far fewer steps than a small model. So it will stop well short of full convergence.

![Performance comparison between large and small models by number of tokens processed](/assets/img/posts/scaling-laws-sample-efficiency.png)
_Figure 2, page 4. Here we see that large models reach an excellent level of performance while processing far fewer tokens than small models._

In other words, the best strategy isn't to train a small model until it has given everything it can. It's to train a very large model, and deliberately stop it well before it finishes converging.

Why does this work? Because a large model is far more efficient per sample seen. Each training step gains it more performance than the same step would gain a small model. Even with just a few thousand steps, a large model comfortably beats a small model pushed to its absolute limit, because the small one plateaus due to its limited capacity, no matter how long you let it run.

It's a bit like comparing two athletes. The first has limited physical potential and very quickly hits a performance ceiling, no matter how much they train. The second has enormous potential and keeps improving rapidly with every session, even though they haven't logged as many training hours as the first. Betting on the second, even with fewer sessions, gives better results.

## Key takeaways from this part

Facing a fixed compute budget, the best strategy is to grow the model massively, increase batch size moderately, and barely increase the number of sequential steps.

This split is excellent news in practice, since it lets you absorb most of the growth in compute through parallelization, rather than through training time that would otherwise explode.

The optimal strategy is to train a very large model and deliberately stop it well before full convergence, rather than training a small model all the way through.

In [**Part 4**]({{ '/posts/scaling-laws-what-this-means-going-forward-part-4-4/' | relative_url }}), the last of this series, we'll step back with the paper's discussion. Among other things, we'll talk about an idea that would later become central to LLM research: the fact that smooth, predictable improvements in performance can hide sudden qualitative jumps in capability. This is the famous concept of emergent capabilities.
