---
title: "How a Network Reads a Sentence: Distributed Representations, RNNs, and LSTMs"
date: 2026-08-19 09:30:01 +0000
categories: [Deep Learning, Natural Language Processing]
tags: [deep-learning, rnn, lstm, nlp, word-embeddings, attention]
math: true
hidden: true
sitemap: false
---

*🇫🇷 [Version française]({{ '/posts/comment-un-reseau-lit-une-phrase-representations-distribuees-rnn-et-lstm/' | relative_url }})*

## Where we left off

The [two previous articles]({{ '/posts/how-a-neural-network-really-learns/' | relative_url }}) in this series covered supervised learning, backpropagation, and then convolutional networks (ConvNets), specialized for processing images. This article closes out our reading of the foundational text by LeCun, Bengio, and Hinton[^lecun2015] by shifting to new terrain: **language**.

An image is spatial data, of fixed size, where every position matters roughly equally. Text is fundamentally different: it's a **sequence**, of variable length, where the order of elements carries meaning. Handling this kind of data calls for a different approach; that's the subject of this article.

## Representing a word: beyond the isolated symbol

### The N-gram problem

Before neural networks, the standard statistical approach to modeling language counted the frequency of short word sequences, or **N-grams**:

> The number of possible N-grams is on the order of $V^N$, where $V$ is the vocabulary size, so taking into account a context of more than a handful of words would require very large training corpora. N-grams treat each word as an atomic unit, so they cannot generalize across semantically related sequences of words.[^lecun2015]

The problem is twofold. First, a combinatorial problem: with a vocabulary of $V = 50{,}000$ words and a context of $N = 3$ words, the number of possible combinations is $50{,}000^3$, a gigantic number. Second, and more importantly, a generalization problem: every word is an **atomic** symbol, unrelated to any other. The model has no "knowledge" that "dog" and "wolf" are semantically close; they're just two different tokens among tens of thousands.

### The solution: distributed representations

Neural networks solve this problem by abandoning the idea of an isolated symbol, in favor of a **distributed representation**:

> Deep-learning theory shows that deep nets have two different exponential advantages over classic learning algorithms that do not use distributed representations. […] learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example, $2^n$ combinations are possible with $n$ binary features).[^lecun2015]

The idea: rather than assigning a single neuron to each concept (a "local" representation), a concept is encoded as a **pattern of activation** spread across several units, each representing a feature shared across multiple concepts. With just $n = 10$ independent binary features (has-fur, has-four-legs, barks…), one can represent $2^{10} = 1024$ different combinations, without ever needing to see all 1024 of those combinations during training.

Concretely, a system with a local representation, faced with a never-seen category (say "fox"), has **no structural way** to represent it. A system with a distributed representation can instead build a coherent representation for "fox" by combining features already learned separately from other animals (has-pointed-snout, has-bushy-tail), even without ever having seen that exact combination.

### Word vectors

Applied to language, this principle produces **word vectors** (or *word embeddings*):

> Each word in the context is presented to the network as a one-of-N vector, that is, one component has a value of 1 and the rest are 0. In the first layer, each word creates a different pattern of activations, or word vectors.[^lecun2015]

Each word enters the network as a *one-hot* vector: a 1 at the position corresponding to that word, 0s everywhere else, across the entire vocabulary size. At this stage, it's still a purely local representation. The network's first layer then transforms this sparse vector into a **dense vector**, just a few hundred dimensions long; this vector becomes the word's true representation inside the network.

Why does this transformation capture semantic proximity? Because the network, trained on a task like predicting the next word in a sentence, discovers that words appearing in **similar contexts** are more efficiently represented by nearby vectors:

> The network learns word vectors that contain many active components each of which can be interpreted as a separate feature of the word. […] When trained to predict the next word in a news story, for example, the learned word vectors for Tuesday and Wednesday are very similar, as are the word vectors for Sweden and Norway.[^lecun2015]

The key point not to confuse: this proximity comes from **substitutability** in similar contexts ("I'll see you on Tuesday" / "I'll see you on Wednesday" work interchangeably), not merely from co-occurrence within the same text. "Tuesday" and "Sweden" might well appear in the same news articles without ever being grammatically substitutable for one another; their vectors therefore stay far apart.

## Recurrent neural networks (RNNs)

### Processing a sequence one element at a time

To make use of these representations when processing a whole sentence, we need an architecture capable of reading a sequence one element at a time, while keeping track of what's already been seen:

> RNNs process an input sequence one element at a time, maintaining in their hidden units a 'state vector' that implicitly contains information about the history of all the past elements of the sequence.[^lecun2015]

Concretely, for a sentence like "the cat sleeps": at each time step $t$, the network receives a word and updates a **state vector** $s_t$, based on both the current word **and** the previous state $s_{t-1}$. The state $s_3$, after reading "sleeps," thus contains a compressed trace of the entire sentence "the cat sleeps."

An important point: the same weight matrices are **reused at every time step**; this is the same weight-sharing principle seen with ConvNets, but applied here along the temporal dimension rather than the spatial one.

### "Unrolling" and backpropagation through time

This structure lets us apply backpropagation directly, exactly as seen in the first article:

> When we consider the outputs of the hidden units at different discrete time steps as if they were the outputs of different neurons in a deep multilayer network, it becomes clear how we can apply backpropagation to train RNNs.[^lecun2015]

If we "unroll" the network through time, one copy of the network per time step, laid end to end, the structure looks like a classical deep network, except that all these stages share the same weights.

### The problem of long sequences

This analogy to a very deep network has a direct consequence: a 100-word sentence is equivalent, once unrolled, to a 100-layer network. And we saw in the first article that backpropagation multiplies the error by a factor at every layer traversed; if that factor is consistently below 1, it crushes the gradient exponentially as it moves back toward the first layers (the *vanishing gradient*).

> RNNs are very powerful dynamic systems, but training them has proved to be problematic because the backpropagated gradients either grow or shrink at each time step, so over many time steps they typically explode or vanish.[^lecun2015]

Concretely, on a long sequence, a classical RNN struggles a great deal to learn **long-term dependencies**: since the learning signal becomes essentially zero for words far upstream, the network learns almost nothing about their importance for the final prediction.

## LSTM: a memory built to last

### The memory cell principle

Faced with this problem, the LSTM (*Long Short-Term Memory*) idea is to augment the network with an explicit memory, whose default behavior is to **preserve** information rather than let it degrade:

> A special unit called the memory cell acts like an accumulator or a gated leaky neuron: it has a connection to itself at the next time step that has a weight of one, so it copies its own real-valued state and accumulates the external signal, but this self-connection is multiplicatively gated by another unit that learns to decide when to clear the content of the memory.[^lecun2015]

The diagram below represents this mechanism: the memory cell has a connection to itself, whose **weight equals exactly 1**, and a learned gate decides when to let a new signal through or clear the existing content.

![LSTM memory cell mechanism](/assets/img/posts/lstm-memory-cell.svg)
_The self-connection with weight 1 is the only stable equilibrium point: below it, the signal vanishes exponentially; above it, it explodes._

**Why precisely a weight of 1, no more, no less?** Over a sequence of 100 time steps, a connection weight of 0.5 would give $0{,}5^{100} \approx 8 \times 10^{-31}$, a negligible signal; that's the vanishing gradient again. A weight of 1.5 would give $1{,}5^{100} \approx 4 \times 10^{17}$, a value that explodes numerically, making training unstable. A weight exactly equal to 1 is the only equilibrium point: $1^{100} = 1$, no matter how many time steps are traversed. This precise choice is what lets information flow across long sequences without degrading or exploding.

### The practical impact

> LSTM networks have subsequently proved to be more effective than conventional RNNs, especially when they have several layers for each time step, enabling an entire speech recognition system that goes all the way from acoustics to the sequence of characters in the transcription.[^lecun2015]

The LSTM became, for a long time, the standard component for any system handling long sequences: machine translation, speech recognition.

## Machine translation: the encoder-decoder architecture

### The principle

One concrete application nicely illustrates all these mechanisms combined: translating a sentence from one language to another, using two jointly trained networks.

> After reading an English sentence one word at a time, an English 'encoder' network can be trained so that the final state vector of its hidden units is a good representation of the thought expressed by the sentence. This thought vector can then be used as the initial hidden state of […] a jointly trained French 'decoder' network, which outputs a probability distribution for the first word of the French translation.[^lecun2015]

The **encoder** reads the source sentence word by word, and keeps only its final state, a compressed vector, called the "thought vector," meant to summarize the meaning of the whole sentence. The **decoder** starts from this vector to generate the translation, one word at a time: each generated word is fed back in as input to predict the next one, until an end-of-sentence symbol.

One result from this approach carries an almost philosophical weight:

> This rather naive way of performing machine translation has quickly become competitive with the state-of-the-art, and this raises serious doubts about whether understanding a sentence requires anything like the internal symbolic expressions that are manipulated by using inference rules.[^lecun2015]

This system manipulates no explicit grammatical rule, no symbolic logical structure, just a vector of real numbers, obtained by composing learned transformations. And yet, it rivals the best translation approaches of its time.

### The limit of a single vector, and the attention mechanism

Compressing the entire meaning of a sentence into a single **fixed-size** vector becomes a bottleneck as the source sentence grows longer: fitting a 200-word paragraph into the same space as a 5-word sentence necessarily implies a loss of information. The text anticipates the solution to this problem:

> We expect systems that use RNNs to understand sentences or whole documents will become much better when they learn strategies for selectively attending to one part at a time.[^lecun2015]

The principle, known today as the **attention mechanism**: instead of keeping only the encoder's final state, we keep the hidden state at **every** word of the source sentence. At each generation step, the decoder compares its current state to each of these source states, computes a **relevance score** for each, then combines these source states according to their relevance, a weighted average, recomputed for every generated word.

![Attention mechanism in a decoder](/assets/img/posts/attention-mechanism.svg)
_To generate "black", the decoder assigns a high attention weight to the source word "black" and a low weight to the others; this weighting is learned, not fixed in advance._

These relevance weights are themselves learned through backpropagation, exactly like the rest of the system; the network discovers, from the data, which part of the source sentence is relevant for generating each word of the translation. The text illustrates this idea applied to image caption generation:

> When the RNN is given the ability to focus its attention on a different location in the input image […] as it generates each word, we found that it exploits this to achieve better 'translation' of images into captions.[^lecun2015]

This principle (replacing a single, fixed summary with a dynamic weighting, recomputed at every step) is what will become, two years after this article's publication, the core of the **Transformer** architecture.

## Beyond simple memory: external memories

The text goes further than the LSTM, toward architectures equipped with an even more structured external memory, capable of more complex reasoning:

> Proposals include the Neural Turing Machine in which the network is augmented by a 'tape-like' memory that the RNN can choose to read from or write to, and memory networks, in which a regular network is augmented by a kind of associative memory.[^lecun2015]

One striking example illustrates the capabilities of these Memory Networks:

> In one test example, the network is shown a 15-sentence version of the *The Lord of the Rings* and correctly answers questions such as "where is Frodo now?"[^lecun2015]

To answer correctly, the network must have tracked, across 15 sentences, the character's successive movements, and retrieve the most recent and relevant piece of information: a principle that rests on the same logic as attention, comparing a query (the question asked) against a set of stored elements, then weighting their relevance. Memory Networks can be seen as a generalization of this principle, applied to a memory that's more durable and larger than a single sentence.

## What the article anticipated for the future

In its conclusion, the text takes a step back to consider the field's future, with observations that have largely proved well-founded:

> Unsupervised learning had a catalytic effect in reviving interest in deep learning, but has since been overshadowed by the successes of purely supervised learning. […] We expect unsupervised learning to become far more important in the longer term. Human and animal learning is largely unsupervised: we discover the structure of the world by observing it, not by being told the name of every object.[^lecun2015]

This prediction did come true, though not immediately; it's mainly with the rise of large language models, trained through self-supervised learning on vast corpora of unlabeled text, that this anticipation has been fully realized. A topic for later in this program.

The text closes on a note that remains, even today, an active research direction:

> Ultimately, major progress in artificial intelligence will come about through systems that combine representation learning with complex reasoning.[^lecun2015]

## Key takeaways

- **Word vectors** replace the atomic symbol with a distributed representation, where the proximity of two words reflects their substitutability in similar contexts, not their mere co-occurrence.
- **RNNs** process a sequence one element at a time, sharing the same weights at every time step, which makes it possible to apply backpropagation by "unrolling" them through time.
- On long sequences, classical RNNs suffer from the same vanishing/exploding gradient problem as a very deep network.
- The **LSTM** solves this problem with a memory cell whose self-connection has a weight exactly equal to 1: the only equilibrium point that avoids both vanishing and exploding the signal.
- The **attention** mechanism, born from the need to overcome the bottleneck of a fixed-size vector, dynamically weighs the importance of each source element, an idea that will become the core of Transformers.

This closes our reading of LeCun, Bengio, and Hinton's foundational article. Next week, the program turns to the Transformer architecture itself, the direct heir of the attention mechanism we've just watched emerge.

---

*The code accompanying this article, a from-scratch implementation of an RNN and an LSTM cell, applied to a simple sequence-prediction task, is available on [GitHub](https://github.com/LaryConseiga/Learning-by-Building/tree/main/projects/rnn-lstm-from-scratch){:target="_blank"}.*

[^lecun2015]: LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. *Nature* **521**, 436–444 (2015). [https://doi.org/10.1038/nature14539](https://doi.org/10.1038/nature14539){:target="_blank"}
