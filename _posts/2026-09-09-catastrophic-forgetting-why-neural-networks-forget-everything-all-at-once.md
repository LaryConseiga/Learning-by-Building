---
title: "Catastrophic Forgetting: Why Neural Networks Forget Everything All at Once"
date: 2026-09-09 10:00:01 +0000
categories: [Deep Learning, Continual Learning]
tags: [catastrophic-forgetting, connectionism, neural-networks, memory]
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/le-catastrophic-forgetting-pourquoi-les-reseaux-de-neurones-oublient-tout-dun-coup/' | relative_url }})*

## Introduction: a thirty year old problem, still unsolved

Before getting into the latest continual learning papers, it's worth going back to the source. In 1999, Robert French published a review in *Trends in Cognitive Sciences* that takes stock of ten years of research on a problem that seems trivial at first glance, but turns out to be one of the most stubborn in the field: **catastrophic forgetting**.

The idea was first highlighted in the late 1980s by McCloskey and Cohen, then by Ratcliff. The finding was simple and a little unsettling: when a backpropagation neural network learns a new set of information, it can **erase, almost entirely, all at once**, everything it knew before. Not a gradual forgetting like the kind we experience ourselves (slowly forgetting the first name of an old classmate), but a genuine collapse.

It's that word "catastrophic" that sets the tone. And understanding why it happens means understanding a fundamental tension that still runs through the entire field of continual learning today.

## The real problem: stability versus plasticity

French connects this phenomenon to a more general problem, long known in cognitive science: the **stability-plasticity problem**. A memory system, to be useful, has to pull off a difficult balancing act:

- be **plastic**, meaning able to learn new things,
- be **stable**, meaning able to not lose what it already knew.

The problem is that in a classic neural network, these two requirements pull in opposite directions, and here's why.

### The story of shared weights

A distributed neural network (like a simple backpropagation network) doesn't have a separate memory drawer for each piece of information it learns. It has one set of weights, shared across every pattern it learns. That's what makes it efficient: no need to store each example individually, and above all, that's what gives it its ability to **generalize**, meaning to react sensibly to an input it has never seen exactly, as long as it resembles something familiar.

The catch is that this same weight serves several different patterns. Picture a single control knob used to tune two different musical instruments. If instrument A needs the knob set to 7, but instrument B needs it set to 3, every time you adjust the knob for B, you throw A out of tune.

That's exactly what happens during learning. When the network learns a new pattern B, backpropagation adjusts the weights so that B is learned well, without caring whether those same weights were also serving A. Result: A degrades.

### Why "catastrophic" and not just "a bit annoying"

Here's where it gets interesting: this phenomenon could very well be gradual and manageable, like the interference observed in humans (French cites an old experiment by Barnes and Underwood, where human subjects learning a new list of words gradually forget the old one, without ever losing it all at once).

But that's not what happens with neural networks. French reports an experiment by Kolen and Pollack (1990), run on a tiny network (two input neurons, two hidden, one output) learning the XOR logic function, one of the simplest examples that exist in machine learning. Result: even on this tiny toy, they show that the space of possible weights contains regions that could be called "weight cliffs." A tiny shift in that space, almost imperceptible, can radically change the network's behavior.

Picture yourself walking a mountain trail in the fog. Most of the time, a small misstep changes nothing. But at certain specific spots, that same small step sends you off a cliff. It's this unpredictability of the terrain that turns ordinary interference into sudden collapse. If the terrain were smooth and predictable everywhere, you'd only get gradual interference, as in humans. It's these hidden cliffs that make the problem catastrophic.

## How forgetting is measured (and why it's subtler than it looks)

To study this phenomenon, it first had to be measured. The original method used by McCloskey and Cohen and by Ratcliff is called **exact recognition**: you take an old pattern, run it back through the network, and check whether every output neuron stays within 0.5 of its original value. If even one neuron crosses that threshold, the whole pattern is declared "forgotten."

The problem with this kind of all or nothing criterion is that it flattens all nuance. Picture a pattern where just one output neuron out of ten slightly crosses the threshold, while the other nine are perfect: it gets labeled "completely forgotten," which overstates the actual damage. Conversely, a pattern can drift substantially across several neurons (say, from 0.9 down to 0.6) without ever crossing the 0.5 threshold, and it gets labeled "not forgotten at all," which understates the actual degradation.

A finer, complementary measure is to look at **relearning time**: how many training cycles does the network need to get back to its original performance level on the old pattern? If the information was truly destroyed, relearning will take roughly as long as learning from scratch. If a trace survives in the weights, relearning will be fast, almost as if the network "remembered" a little. It's a continuous measure, one that captures the degree of forgetting rather than a simple binary verdict.

## The early solutions: reducing overlap

Once the problem was well identified, several researchers in the 90s proposed solutions that, on the surface, look unrelated, but that actually all pursue the same underlying idea.

- **Kortge (1990)** modifies the learning rule with "novelty vectors": instead of correcting every active unit when the network makes a mistake, only the units actually responsible for the error get corrected.
- **French himself (1991, 1992)** proposes **activation sharpening**: a mechanism that boosts the activity of the hidden neurons already most active for a given pattern, and reduces that of the others, to create more "sparse" representations.
- **McRae and Hetherington (1993)** show that if you pretrain the network on a random sample from a structured domain (like language), catastrophic forgetting nearly disappears for the rest of learning.

The common thread across these three approaches is **reducing the overlap of internal representations** between different patterns. If two patterns activate different hidden neurons, they draw on different weights, and learning one no longer overwrites the other. It's as if, instead of everyone fighting over the same two or three control knobs, each group of patterns had its own dedicated knobs, with just enough overlap left where it's useful for generalization.

French calls these **"semi-distributed"** representations: neither fully local (a single neuron per pattern, no generalization possible), nor fully distributed (total overlap, catastrophic risk), but a compromise between the two.

### The extreme case: going fully local

Some models, like **CALM** or **ALCOVE**, push this logic all the way. In ALCOVE, for instance, a hidden neuron's activation depends on its distance to the input: each neuron "covers" a limited region of the possible input space, a bit like a spotlight that only lights up a small corner of the stage. If you tune that spotlight to be very narrow, the network becomes essentially local, and catastrophic forgetting almost entirely disappears.

But nothing comes for free. What made distributed networks powerful in the first place is precisely their ability to generalize, to react intelligently to an input never seen before but close to something familiar, or to a noisy or incomplete input. Push too far into "fully local," and each pattern becomes an isolated island: no more overlap, so no more forgetting, but also no real generalization left. The network no longer knows what to do with an input it hasn't seen in exactly that form.

And this raises a problem that's deeper than a technical one. If the goal is to faithfully model how a human brain learns and retains information, a "fully local" system isn't satisfying, because humans generalize constantly: we recognize a half hidden face, we understand a sentence we've never heard in that exact wording before. By sacrificing generalization to avoid forgetting, you move further away from the very behavior you're trying to reproduce.

## Rehearsal: replaying old memories

A second family of solutions doesn't touch the internal representation, but rather how learning unfolds. The idea behind **rehearsal**: instead of training the network only on new patterns, you mix the new patterns with old patterns already learned, a bit like reviewing your old coursework at the same time as you learn new material.

The practical problem becomes obvious as soon as you step out of the lab: this method assumes you kept all the old patterns somewhere. In a realistic continual learning scenario, that's almost never the case. You haven't necessarily stored every example you learned months ago, and even if you could, keeping everything indefinitely would be costly.

### The clever idea behind pseudopatterns

That's where an innovation proposed by Robins in 1995 comes in: **pseudopatterns**. The idea is to completely bypass the need to keep the real old data. Here's how it works:

1. Take the already trained network, which has learned a certain function (a certain way of mapping inputs to outputs).
2. Generate a **completely random** input.
3. Run it through the network, which produces an output.
4. That pair (random input, produced output) forms a **pseudopattern**.

Why does this work? Because the output produced for a random input isn't random itself: it faithfully reflects the function the network has internalized in its weights, meaning its way of generalizing, shaped by everything it has learned so far. The pseudopattern therefore captures a kind of imprint of the network's knowledge, without ever needing to store a real original example.

These pseudopatterns are then mixed with the new patterns during training. The network keeps "seeing" a condensed summary of what it knew before, which keeps its weights from drifting too far from the original solution.

### When it backfires: catastrophic remembering

French points out an amusing and instructive limitation of this technique, which he calls **catastrophic remembering** (the irony of the name is intentional). Picture an autoassociative network, whose task is to reproduce at the output exactly what it's given at the input. The network "knows" it has already seen a pattern if, when given that pattern again, the output closely resembles the input.

If you generate and replay pseudopatterns in a loop, over many cycles, the network gets better and better at generalizing, to the point of faithfully reproducing any input, even pure noise it has never seen. The problem is that, at that point, the network loses its ability to **discriminate**: it can no longer tell a pattern it actually learned apart from a random input that just happens, by accident, to resemble something familiar. Everything ends up feeling familiar to it.

This is a new face of the same stability-plasticity dilemma, but this time along the axis of memorization versus discrimination rather than stability versus generalization.

## Splitting into two systems: taking inspiration from the brain

The last major direction French develops, together with Ans and Rousset on their own side, is architectural: rather than a single network that has to do everything, you use **two separate networks**. One for fast processing of recent information, the other for long term storage of already consolidated regularities.

This idea is picked up and given a biological grounding by McClelland, McNaughton, and O'Reilly (1995), who anchor it in a well known neuroscience distinction: the **hippocampus** (fast learning of new, specific information) and the **neocortex** (slow, gradual discovery of general structure, long term storage). Their argument is that these two functions are fundamentally incompatible within a single system: learning something new and specific quickly, and slowly discovering general regularities, pull in opposite directions, a bit like trying to sprint and walk slowly at the same time with the same legs.

The link to pseudopatterns then becomes obvious. If the "long term" network keeps running in parallel and generates its own pseudopatterns to consolidate itself, while the "recent" network learns the new patterns, you completely avoid the mixing of roles seen in catastrophic remembering. Each network has a clear job: one encodes the new, the other digests and consolidates the old, without the two information streams colliding in the same weight space.

## What was still open in 1999

French closes his article with a list of questions that were unresolved at the time. One of the most interesting concerns pseudopatterns themselves: does this mechanism, purely computational to begin with, have a genuine biological equivalent?

In other words, does the brain actually generate something like "internal random inputs" that it replays through its own circuits to consolidate itself? And if so, when? French offers a hypothesis: perhaps during **REM sleep**, that period when the brain stays active but cut off from external stimulation, which would fit reasonably well with the idea of generating inputs that don't come from the real world. He adds an important nuance: if this mechanism really exists, there's no reason to assume it would be purely random the way it is in computational models. The brain may have evolved a smarter way of doing this internal rehearsal, preferentially replaying what matters most rather than pure noise.

So, do you think your dreams really replay your memories at random, or do they already sort out what deserves to be kept?
