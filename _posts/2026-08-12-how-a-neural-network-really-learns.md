---
title: "How a Neural Network Really Learns: From Raw Pixels to Backpropagation"
date: 2026-08-12 09:00:01 +0000
categories: [Deep Learning, Fundamentals]
tags: [deep-learning, backpropagation, supervised-learning, neural-networks, mlp]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/comment-un-reseau-de-neurones-apprend-vraiment/' | relative_url }})*

## Why this first article

This week I'm starting a four-week, self-directed deep learning curriculum: perceptrons and backpropagation, then CNNs/RNNs, then Transformers, then LLMs and scaling laws. Each week comes with a code project and an article: this is the first one.

The throughline for week 1 is a foundational text: *Deep Learning*, published by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton in *Nature* in 2015[^lecun2015]. It isn't a narrow technical research paper, but a synthesis written by three of the researchers who did the most to make deep learning what it is today, aimed at a broad scientific audience. That makes it an excellent entry point.

This article covers three things: **why** deep learning changed the game compared to classical machine learning, **how** a neural network actually learns (supervised learning), and **by what mathematical mechanism** this learning propagates through all the layers of a network (backpropagation). The accompanying code (a from-scratch multilayer perceptron implementation) is available in [this GitHub repository](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/mlp-from-scratch){:target="_blank"}.

## The problem deep learning came to solve

Before understanding what deep learning brings, it helps to understand what the problem was before it.

### Classical machine learning needed a human translator

For decades, building a pattern-recognition system required considerable manual engineering work. An expert had to design a **feature extractor**: a hand-written function that transformed raw data (for example, the pixel values of an image) into a more usable representation, from which a simple classification algorithm could then do its job.

As the authors put it:

> Conventional machine-learning techniques were limited in their ability to process natural data in their raw form. For decades, constructing a pattern-recognition or machine-learning system required careful engineering and considerable domain expertise to design a feature extractor that transformed the raw data […] into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input.[^lecun2015]

Concretely, for a 2005-era face-recognition system, an engineer had to spend time deciding: which edges to look for, which textures, which facial proportion ratios were relevant. That work was slow, expensive, task-specific, and didn't generalize well to a neighboring problem.

### The central idea: learn representations, don't design them

Deep learning flips this paradigm. Instead of a human designing the relevant features, the network **discovers them itself**, from data, through a general-purpose learning procedure.

> The key aspect of deep learning is that these layers of features are not designed by human engineers: they are learned from data using a general-purpose learning procedure.[^lecun2015]

What makes this possible is the idea of a **hierarchy of representations**. A deep network is made of several stacked layers, and each layer transforms the representation it receives from the previous layer into a slightly more abstract representation. For an image, this hierarchy typically looks like this:

1. **Layer 1**: detects the presence or absence of edges, at particular orientations and positions
2. **Layer 2**: detects motifs, by spotting particular arrangements of layer-1 edges
3. **Layer 3**: assembles these motifs into larger combinations corresponding to parts of familiar objects
4. **Later layers**: detect whole objects, as combinations of those parts

One essential point to grasp: **each layer never "sees" the raw data**, except the very first one. Layer 3 never looks at the image's pixels directly; it only receives the already-transformed output of layer 2. It's a strictly sequential process, where abstraction accumulates stage after stage, a bit like an assembly line where each workstation only sees the previous station's output, never the original raw material.

And crucially: **none of this is programmed in advance**. Nobody explicitly tells layer 2 to "look for closed loops" or "look for cross-shaped intersections." It's an **emergent** phenomenon: it's the overall optimization process (detailed further below), seeking to minimize classification error, that pushes each layer to build the representations most useful for the task, and these representations turn out, after the fact, to correspond to concepts a human would recognize (edges, motifs, object parts).

An interesting consequence of this hierarchy: a network's very first layers (the edge detectors) are nearly **identical** from one task to another, whether the network is trained to recognize cats, faces, or road signs, because these are universal statistical structures of natural images. Deeper layers, on the other hand, diverge strongly from task to task, because they encode whatever is specifically discriminative for that particular problem. This is, in fact, the principle behind *transfer learning*, which we'll come across later in this program.

### Why this approach has scaled so well

One last remark from the authors, which explains a good part of the field's dynamics since 2012:

> We think that deep learning will have many more successes in the near future because it requires very little engineering by hand, so it can easily take advantage of increases in the amount of available computation and data.[^lecun2015]

Unlike an approach based on hand-designed features (which plateaus fairly quickly even given more data or compute, because a human can't design 100 times more relevant features just because there's 100 times more data), deep learning **directly absorbs** both of these resources. More data and more compute translate almost mechanically into better learned representations. It's this scalability property that explains the current race toward ever-larger models, trained on ever-larger datasets.

## Supervised learning: how a network learns, concretely

Let's move from "why" to "how." The most common form of learning, deep or not, is **supervised learning**.

### The five-step mechanism

Imagine we want to build a system that classifies images into four categories: house, car, person, pet. We first collect a large dataset of images, each labeled with its correct category.

**1. The forward pass.** We show the network an image. It outputs a vector of scores, one per category. We'd like the correct category to have the highest score, but before any training, with randomly initialized weights, that's almost never the case.

**2. The objective function (loss).** We compute a function that measures the gap (the error, or distance) between the scores produced by the network and the desired pattern of scores (typically, a score of 1 for the correct category, 0 for the others).

**3. The weights.** The network then adjusts its internal, tunable parameters to reduce this error. These parameters, called weights, are real numbers: genuine "knobs" that define the network's input-output function. In a typical deep learning system, there can be hundreds of millions of them.

**4. The gradient.** To correctly adjust the weight vector, the learning algorithm computes a **gradient**: for each weight, it indicates by how much the error would increase or decrease if that weight were increased by an infinitesimal amount.

**5. The update.** The weight vector is then adjusted **in the direction opposite** to the gradient:

$$
w \leftarrow w - \eta \cdot \frac{\partial E}{\partial w}
$$

where $\frac{\partial E}{\partial w}$ is the gradient of the error $E$ with respect to weight $w$, and $\eta$ (eta) is the **learning rate**, a small positive number that controls the size of the update step. If the gradient is positive (increasing $w$ would increase the error), we decrease $w$. If it's negative, we increase it. The minus sign in front of the learning rate guarantees we always move in the direction that reduces the error.

### The hilly-landscape picture

The authors offer a useful analogy to visualize this process:

> [The objective function, averaged over all the training examples,] can be seen as a kind of hilly landscape in the high-dimensional space of weight values. The negative gradient vector indicates the direction of steepest descent in this landscape, taking it closer to a minimum, where the output error is low on average.[^lecun2015]

Picture a hilly landscape, but in several million dimensions (one dimension per weight, impossible to picture mentally beyond three, but the 2D or 3D intuition helps). We're looking for the lowest point of this landscape, i.e., the minimal error. The gradient gives the local slope at wherever we currently stand, and we move forward by descending that slope, one small step at a time, hence the name **gradient descent**.

### In practice: stochastic gradient descent

Computing the exact gradient would, in theory, require averaging over the entire dataset at every update step. With hundreds of millions of examples, that would be far too costly in time and computation.

In practice, we therefore use **stochastic gradient descent** (SGD):

> This consists of showing the input vector for a few examples, computing the outputs and the errors, computing the average gradient for those examples, and adjusting the weights accordingly. The process is repeated for many small sets of examples from the training set until the average of the objective function stops decreasing.[^lecun2015]

We take a small subset of examples (a *mini-batch*, say 32 or 128 images), compute the gradient on just that subset, update the weights, then repeat with a new subset. It's "stochastic" because each small batch only gives a noisy estimate of the exact gradient computed over the whole dataset, not the perfect value.

One thing that surprised many practitioners at the time: this simple method, despite being approximate, generally finds a good set of weights surprisingly fast, compared to far more sophisticated optimization techniques.

A useful clarification: SGD's randomness concerns the **order** in which examples are presented, not their omission. In practice, the dataset is shuffled and then split into successive mini-batches that cover every example: one full pass is called an *epoch*. Each example is therefore seen exactly once per epoch, with no systematic bias.

### Why linear classifiers aren't enough

A good chunk of practical machine learning applications use linear classifiers applied to hand-designed features. A two-class linear classifier computes a weighted sum of the feature vector's components; if this sum exceeds a threshold, the input is classified into a given category.

The problem, known for a long time:

> Since the 1960s we have known that linear classifiers can only carve their input space into very simple regions, namely half-spaces separated by a hyperplane.[^lecun2015]

A linear classifier can only draw a straight (or flat, in high dimensions) boundary between two classes. If the data isn't separable by a straight line, it fails, no matter how its weights are tuned.

The example the authors give illustrates well why this is a serious problem in computer vision: two photos of the same Samoyed dog, taken in different poses and different environments, can look very different pixel by pixel, whereas a photo of a Samoyed and a photo of a white wolf, taken in a similar pose and environment, can look very close pixel by pixel. A linear classifier operating directly on raw pixels has no reliable way to tell the difference.

This problem has a precise name in the paper: the **selectivity–invariance dilemma**.

> This is why shallow classifiers require a good feature extractor that solves the selectivity–invariance dilemma — one that produces representations that are selective to the aspects of the image that are important for discrimination, but that are invariant to irrelevant aspects such as the pose of the animal.[^lecun2015]

- **Selectivity**: staying sensitive to details that truly matter for discrimination (the shape of the snout, the skull structure)
- **Invariance**: staying insensitive to details that don't matter (pose, lighting, background)

This is exactly what the hierarchy of representations, described above, solves: each layer progressively becomes more invariant to irrelevant details, while remaining selective about what actually matters for the task.

## Backpropagation: how error travels back to the earliest layers

We now know, in theory, how a network adjusts its weights. One crucial question remains: in a multi-layer network, a weight sitting deep near the input influences the output only **indirectly**: it influences the next layer, which influences the layer after that, and so on until the output, where the error is computed. How do we compute the effect of such a distant weight on an error measured so far downstream?

This is exactly what the **backpropagation** algorithm solves.

### The forward pass, formally

Before looking at the backward mechanism, let's formalize the forward trip. For a hidden unit $j$, receiving outputs $x_i$ from the previous layer:

$$
z_j = \sum_{i} w_{ij}\, x_i \qquad\qquad y_j = f(z_j)
$$

$z_j$ is the weighted sum of the inputs, and $y_j$ is the unit's output, obtained by applying a non-linear function $f$ to $z_j$. Without this non-linearity, stacking layers would be pointless: a weighted sum of weighted sums is still a weighted sum, unable to represent complex functions. This non-linearity is what gives the network its ability to learn non-linear decision boundaries, exactly what the linear classifier above was missing.

Among the non-linear functions in use, the most common today is **ReLU** (*Rectified Linear Unit*): $f(z) = \max(0, z)$. We'll come back to why it supplanted the more classical sigmoids.

### The key insight of backpropagation

The core intuition, as stated by the authors:

> The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module.[^lecun2015]

In other words: to know a layer's effect on the final error, you only need to know the effect of the **next** layer, no need to recompute everything from scratch each time. This principle relies on the **chain rule** of derivatives: if $x$ influences $y$, and $y$ influences $z$, then the effect of $x$ on $z$ is the product of the two intermediate effects:

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

### The mechanism, layer by layer

Starting from the output and working back toward the input:

**Step 1: the output error**, directly computable since we know both the prediction $y_l$ and the target $t_l$:

$$
\frac{\partial E}{\partial y_l} = y_l - t_l
$$

**Step 2: conversion through the non-linearity**, applying the chain rule:

$$
\frac{\partial E}{\partial z_l} = \frac{\partial E}{\partial y_l} \cdot \frac{\partial y_l}{\partial z_l}
$$

**Step 3: propagation to the layer below.** This is the most important step to really understand:

$$
\frac{\partial E}{\partial y_j} = \sum_{l} w_{jl}\, \frac{\partial E}{\partial z_l}
$$

Concretely, this formula says: the error "inherited" by a unit $j$ is a **weighted sum** of the errors of every unit in the next layer that $j$ is connected to, weighted by the **same weights** $w_{jl}$ used in the forward pass, but now used in reverse.

Why precisely that weight? Because $w_{jl}$ measures, in the forward pass, how much $j$ influences $l$. If $w_{jl}$ is large, a small change in $y_j$ has a big effect on $z_l$, and therefore on the final error: $j$ then inherits a large share of responsibility. If $w_{jl}$ is close to zero, $j$ has almost no influence on the output through that connection, and inherits almost no error along that path. If $j$ is connected to several downstream units, it accumulates its share of responsibility from **each** of them, proportional to the importance of each connection.

Steps 2 and 3 are then repeated layer after layer, moving downward: the error propagates backward from the output toward the input. That's where the name "back-propagation" comes from.

**Step 4: each weight's gradient**, once we have $\partial E/\partial z_k$ for a layer:

$$
\frac{\partial E}{\partial w_{jk}} = y_j \cdot \frac{\partial E}{\partial z_k}
$$

This gradient is then plugged directly into the update formula seen above.

### Why this detour saves an enormous amount of computation

Without this mechanism, computing each weight's gradient independently would require, for every weight in a deep layer, retracing the entire chain of influence all the way to the output, a cost that would explode with the number of layers and weights. Thanks to backpropagation, a layer's error is computed **only once**, derived algebraically from the layer just above it, and every weight gradient in that layer then follows cheaply.

The result: the total cost of backpropagation stays on the same order of magnitude as a single forward pass. This is precisely what makes it practically possible to train networks with hundreds of millions of weights.

### The myth of local minima

In the 1990s, the machine learning community believed that simple gradient descent would get trapped in bad **local minima**, weight configurations where no small change would reduce the error, but which would be far from optimal.

Go back to the hilly-landscape picture: a local minimum is a small valley where the terrain rises in every direction around you, even though a much deeper valley might exist elsewhere; but reaching it would require climbing back up first, which gradient descent never does.

Experience has shown this fear was largely unfounded:

> In practice, poor local minima are rarely a problem with large networks. Regardless of the initial conditions, the system nearly always reaches solutions of very similar quality.[^lecun2015]

The real phenomenon at play is different: the error landscape, in very high dimensions, contains a combinatorially large number of **saddle points**, points where the gradient is zero, but where the terrain rises in some directions and falls in others, like the seat of a horse saddle. These points don't durably block the algorithm, because there's almost always at least one available descent direction, and empirically, nearly all of these saddle points have fairly similar error values, so it hardly matters which one you happen to cross.

### ReLU versus the vanishing gradient problem

One last important technical point, which explains why ReLU became dominant:

> The ReLU typically learns much faster in networks with many layers, allowing training of a deep supervised network without unsupervised pre-training.[^lecun2015]

Classical sigmoid functions (like the hyperbolic tangent) have a derivative that's nearly zero far from zero. Now, in the backpropagation formula seen above, we **multiply** the error by this derivative at every layer traversed. If this derivative is close to zero at each stage, and we multiply this factor ten or twenty times in a row (once per layer), the gradient becomes exponentially small as it moves back toward the first layers: this is the **vanishing gradient** problem. ReLU, whose derivative is either 0 or exactly 1, never gets progressively crushed this way, which makes it possible to train deep networks directly, without a preliminary unsupervised pre-training step.

## Key takeaways

- Deep learning replaces manual feature engineering with an **automatically learned hierarchy of representations**, each layer building its own features from the previous layer's output.
- **Supervised learning** adjusts a network's weights by minimizing an error function via gradient descent: in practice, a stochastic version (SGD) using small mini-batches rather than the full dataset.
- Classical linear classifiers fail on problems like vision, because they can't solve the **selectivity–invariance dilemma**.
- **Backpropagation** efficiently computes each weight's gradient, across every layer, by reusing the forward pass's weights to propagate the error in reverse: an explicit, layer-by-layer computation, never some automatic phenomenon.
- Historical fears about local minima turned out to be largely unfounded; the real challenge (saddle points) doesn't durably block learning in high dimensions.

Next week, we move from generic multilayer perceptrons to an architecture specialized for image processing: **convolutional neural networks** (CNNs), which directly exploit the spatial structure of data to make training practical on large images.

---

*The code accompanying this article, a from-scratch implementation of a multilayer perceptron and backpropagation, with no deep learning framework, is available on [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/mlp-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
