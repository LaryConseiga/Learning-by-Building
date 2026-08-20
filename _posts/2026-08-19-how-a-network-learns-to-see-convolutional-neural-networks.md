---
title: "How a Network Learns to See: Convolutional Neural Networks"
date: 2026-08-19 09:00:01 +0000
categories: [Deep Learning, Computer Vision]
tags: [deep-learning, cnn, convnets, computer-vision, pooling]
math: true
---

*🇫🇷 [Version française]({{ '/posts/comment-un-reseau-apprend-a-voir-les-reseaux-de-neurones-convolutifs/' | relative_url }})*

## Where we left off

In the [first article]({{ '/posts/how-a-neural-network-really-learns/' | relative_url }}) of this series, we saw how a neural network learns: the forward pass, gradient descent, and backpropagation, the mechanism that computes each weight's effect on the final error, across every layer of a network.

What we hadn't covered yet is the **architecture** itself. A classical multilayer perceptron, so-called *fully-connected*, applies the same generic scheme regardless of the nature of the data. But for one specific problem (computer vision), this genericity gets expensive, and a specialized architecture becomes necessary: **convolutional neural networks**, or **ConvNets** (*Convolutional Neural Networks*, CNN). That's the subject of this article, still drawing from the foundational text by LeCun, Bengio, and Hinton[^lecun2015].

## The problem ConvNets solve: the explosion in the number of weights

### Reminder: how weights are counted in a fully-connected layer

In a fully-connected layer, every unit in the hidden layer is connected to **all** the units of the previous layer, with an independent weight per connection. The total number of weights is computed simply:

$$
\text{number of weights} = (\text{number of inputs}) \times (\text{number of units})
$$

For a small 100-pixel image and a hidden layer of 5 units, that gives $100 \times 5 = 500$ weights, still manageable.

### The problem at real-world scale

But real images aren't 100 pixels. Take a modestly sized image, 1000×1000 pixels (that is, 1,000,000 pixels), and a first hidden layer of just 1000 units, which is still modest for a vision network:

$$
1{,}000{,}000 \text{ pixels} \times 1000 \text{ units} = 1{,}000{,}000{,}000 \text{ weights}
$$

**A billion weights**, for the first layer alone. This number creates three concrete problems:

1. **Compute and memory.** Storing and updating a billion weights, again and again at every gradient descent step, demands considerable computing power and memory.
2. **Data requirements.** The more weights there are to tune, the more labeled examples are needed to adjust them correctly without falling into overfitting.
3. **No spatial structure exploited.** Each weight is tied to a pixel at a specific position. If the network learns to detect a motif in the top-left corner of an image, that motif won't be recognized if it appears elsewhere: the weights that would detect it in the bottom-right were never trained for that.

This last point connects directly to the **selectivity-invariance dilemma** raised in the first article: we want a network sensitive to details that matter (the shape of a snout), but insensitive to details that don't (its exact position in the image). A fully-connected layer offers no structural way to achieve this invariance.

## The four key ideas behind ConvNets

The text identifies precisely the four principles that address this problem:

> There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.[^lecun2015]

### 1. Local connections

Instead of a unit looking at the entire image, it only looks at a small **local patch**, for example a 3×3 or 5×5 pixel window. On a 6×6 pixel image with a 3×3 patch, a unit only needs 9 weights, compared to 36 for a fully-connected unit looking at the whole image.

This window doesn't move randomly: it **systematically** sweeps across the entire image, position after position, left to right then top to bottom, so that no zone is skipped.

### 2. Shared weights: the central idea

This is the most important mechanism, the one that truly distinguishes ConvNets from a simple layer of local connections:

> All units in a feature map share the same filter bank. […] the local statistics of images and other signals are invariant to location. In other words, if a motif can appear in one part of the image, it could appear anywhere, hence the idea of units at different locations sharing the same weights and detecting the same pattern in different parts of the array.[^lecun2015]

The same small set of weights (called the **filter bank**, or simply a filter) is reused at **every** position the patch sweeps over. This isn't nine different units with nine different sets of weights; it's **a single** unit, whose filter slides across the entire image.

The diagram below illustrates this mechanism: at position 1 and at position 2, it's exactly the **same** 9-weight filter being applied; only its position on the image changes.

![Weight-sharing mechanism in a convolutional layer](/assets/img/posts/convnet-shared-weights.svg)
_The same filter slides across the whole image; its weights never change, only its position does. The result forms a feature map, later reduced by pooling._

**The effect on the number of weights is dramatic.** Take the example of a 100×100 pixel image, with a 5×5 filter:

| Approach | Weights to learn |
|---|---|
| Fully-connected (1 unit looking at the whole image) | 10,000 |
| Local connections, no weight sharing | ≈ 230,400 |
| **Local connections + shared weights** | **25** |

It doesn't matter how many positions the filter sweeps over, ten or ten thousand: a single set of 25 weights is enough, since it's reused everywhere. This is mathematically a **convolution** operation, hence the architecture's name:

> Mathematically, the filtering operation performed by a feature map is a discrete convolution, hence the name.[^lecun2015]

**And this sharing also solves the invariance problem.** Since the same motif detector (say, a vertical-edge detector) is applied at every position in the image, it doesn't need to have been trained separately for every place that motif might appear.

A layer typically has several filters running in parallel, each independent, to detect different types of motifs:

> Different feature maps in a layer use different filter banks.[^lecun2015]

A filter in a deep layer actually looks at a patch across **all** the feature maps produced by the previous layer, not just one. A 3×3 filter receiving 10 input feature maps therefore has $3 \times 3 \times 10 = 90$ weights.

### 3. Pooling: reducing without learning

Pooling addresses a subtler problem that weight sharing alone doesn't fully solve: even with a shared filter, a detector's output still shifts slightly if the motif moves by just a few pixels.

> A typical pooling unit computes the maximum of a local patch of units in one feature map […] thereby reducing the dimension of the representation and creating an invariance to small shifts and distortions.[^lecun2015]

The mechanism, most often in the form of **max pooling**, is simple: the feature map is cut into small blocks (typically 2×2), and only the **maximum** value of each block is kept.

**Let's take a concrete example.** If a feature map contains these four values in a 2×2 block:

```
[ 0.2  0.8 ]
[ 0.1  0.3 ]
```

Pooling keeps only **0.8**, the maximum. It doesn't matter whether the detected motif was precisely in the top-right corner of the block or slightly shifted: as long as it's detected somewhere in this small zone, the strong signal is preserved.

This mechanism has two complementary effects: it **reduces the size** of the representation (a 2×2 pooling divides the number of values by four), and it **adds invariance** to small positional shifts.

**One important point not to confuse: pooling has no learned weights.** It isn't a layer trained by backpropagation like convolution; it's a fixed, purely mechanical operation. Weight sharing reduces the **number of parameters** to learn in the convolution; pooling reduces the **spatial size** of the output, with no parameters at all. These are two different reductions, at two different stages of processing.

### 4. Stacking multiple layers

A typical ConvNet architecture chains these building blocks several times in a row:

> Two or three stages of convolution, non-linearity and pooling are stacked, followed by more convolutional and fully-connected layers.[^lecun2015]

The precise order inside each stage is never arbitrary:

$$
\text{Convolution} \rightarrow \text{ReLU} \rightarrow \text{Pooling}
$$

The convolution first computes the local weighted sum ($z$) using the learned weights. The ReLU non-linearity ($f(z) = \max(0, z)$) is then applied to each value, without changing the grid's size. Pooling comes **last**, once the layer is activated, to reduce the spatial size before moving to the next stage.

This cycle repeats several times, not just once at the very end of the network. A 100×100 pixel image, after three successive 2×2 pooling stages, ends up around 12×12, while generally gaining more feature maps at each stage. This progression connects directly to the hierarchy of representations described in the first article: each convolution-ReLU-pooling stage roughly corresponds to a level of that hierarchy: edges, then motifs, then object parts.

**What's fixed in advance, and what's learned.** Filter size, the pooling window size, and the number of feature maps per layer are design choices, made before training, and nothing requires keeping them identical from one layer to the next. What's learned through backpropagation is only the **numerical values** inside each filter.

## The 2012 turning point: ImageNet

These principles already existed in the 1990s, with real practical successes: automated bank check reading, for example, processed more than 10% of checks in the United States during the 1990s. But ConvNets remained largely shunned by the computer vision community, which favored approaches based on hand-designed features, until one precise moment:

> Despite these successes, ConvNets were largely forsaken by the mainstream computer-vision and machine-learning communities until the ImageNet competition in 2012. When deep convolutional networks were applied to a data set of about a million images from the web that contained 1,000 different classes, they achieved spectacular results, almost halving the error rates of the best competing approaches.[^lecun2015]

Cutting the error rate of the best competing approaches in half, on a million images and 1000 categories: this result is what triggered the massive adoption of deep learning in computer vision. The text identifies precisely the ingredients of this success:

> This success came from the efficient use of GPUs, ReLUs, a new regularization technique called dropout, and techniques to generate more training examples by deforming the existing ones.[^lecun2015]

The role of GPUs and ReLU was already discussed in the first article. Two new elements deserve an explanation.

**Dropout** is a regularization technique that randomly disables a portion of the network's units during training, forcing the network not to over-rely on any particular combination of units, which reduces overfitting.

**Data augmentation**, meanwhile, involves artificially creating new training examples by slightly distorting existing images: rotation, cropping, mirroring, brightness variation. This isn't equivalent to simply collecting more real photos: this technique **specifically targets** the variations we know, a priori, shouldn't change the classification. It directly teaches the network the invariance it needs, complementing what the architecture already provides (shared weights and pooling).

## Key takeaways

- A fully-connected layer applied to real images produces an unmanageable number of weights, without exploiting the spatial structure of the data.
- ConvNets address this problem with four ideas: **local connections** (a small patch, not the whole image), **shared weights** (the same filter reused at every position, hence the term convolution), **pooling** (size reduction and invariance to small shifts, with no learned weights), and **layer stacking** (repeating the convolution-ReLU-pooling cycle).
- Weight sharing reduces the number of parameters to learn; pooling reduces the spatial size of the output: two distinct mechanisms, not to be confused.
- The 2012 ImageNet turning point demonstrated this architecture's practical superiority, driven by GPUs, ReLU, dropout, and data augmentation.

Next article: we leave the territory of still images for that of sequences (text and speech), with recurrent neural networks and distributed word representations.

---

*The code accompanying this article, a from-scratch implementation of a small ConvNet, with convolution, ReLU, and max pooling coded by hand, is available on [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/convnet-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
