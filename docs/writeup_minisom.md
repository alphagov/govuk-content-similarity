# Document Similarity: SOMs

- Avision Ho
- 20th October 2020

# Introduction
## Aims
- Create rudimentary document vectors using doc2vec for experimenting with SOMs.
- Applies Self-Organising Maps (SOMs) to identify and cluster semantically similar GOV.UK content.
- Visualises these clusters interactively using [bokeh](https://docs.bokeh.org/en/latest/index.html).

## What are SOMs?
SOMs is primarily a dimensionality-reduction algorithm that takes an input space and represents it in a lower-dimensional space. It is an artificial neural network that is trained using unsupervised learning.

Unlike other artificial neural networks, they apply competitive learning instead of error-correction learning (such as backpropagation with gradient descent). They use a neighbourhood function to preserve the topological properties of the input space.

The algorithm works as follows:
1. A set of document vectors, `k`, are created; this is your training data.
1. A set of `mxn` nodes with their weights are initialised.
    + Note, `m * n < k`.
    + A rule of thumb for determining `m` and `n` is that they should approximately be the ratio of the two largest eigenvalues of the training data’s covariance matrix.
1. A document vector is chosen randomly from the training data.
1. Every node is examined to compute which one's weights are closest to the randomly chosen document vector in the previous step.
    + The node that is closest to the randomly chosen document vector, the winning node, is commonly known as the **Best Matching Unit (BMU)**.
    + The measure of distance to define *'closeness'* is typically Euclidean distance though for some implementations of SOMs, this can be configured to take other measures like cosine-similarity or manhattan distance.
1. The neighbourhood of the BMU is computed. The amount of neighbours decreases over time.
1. The weight associated to the BMU becomes more like the chosen document vector; where the closer a node is to the BMU, the more its weights get altered and the further away the neighbour is from the BMU, the less it gets altered.
1. Repeat steps 2-5 for `k` iterations.

[![Figure 1](https://en.wikipedia.org/wiki/File:Somtraining.svg)](https://en.wikipedia.org/wiki/Self-organizing_map)

[![Figure 2](https://en.wikipedia.org/wiki/File:TrainSOM.gif)](https://en.wikipedia.org/wiki/Self-organizing_map)

## Sounds like PCA...
Can be considered a non-linear generalisation of PCA, with many advantages over conventional feature extraction methods such as PCA.

## What's the catch?
Drawbacks of SOMs are:
1. It does not build a generative model for your data so it does not understand how the data is created. This means it does consider the distribution of your data so cannot tell you how likely a given example is.
1. Time preparing the model is slow so hard to train against evolving data.
1. Does not work well with categorical data and even less so for mixed types data.

[[*Python Data Science, 2020*](https://python-data-science.readthedocs.io/en/latest/unsupervised.html)]

[[*Ralhan, 2018*](https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4)]

***

# Let's apply it!

## Data load and preprocessing

#### `notebooks/SOMs/01_load_clean_.py`
We download the pre-processed content store of all the pages on GOV.UK from AWS on store it in our data directory.

> Is approximately over 530,000 GOV.UK pages.

Then remove non-alphabetic characters, NAs and duplicates from the GOV.UK text.

To remove possible noise further from our text data, we also lemmatise the text and remove stopwords.

## Document vector creation

#### `notebooks/SOMs/02_embedding.py`
We use [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html) to create the document vectors. The decision to use this technique was to quickly get some vectors representing documents set-up so that we can test out SOMs.

There are probably better alternative ways to generate these vectors, especially given there are literature concerns over doc2vec. A few are provided below:
- Universal Sentence Encoder sentence embeddings and then a form of averaging of these. This is actually done in a parallel workstream by @whojammyflip

## Applying SOMs

#### `notebooks/SOMs/03_som_visualise_minisom.py`

We use the [minisom](https://github.com/JustGlowing/minisom) implementation of SOMs for its ease-of-use, continual maintenance and comprehensive examples.

We apply the SOM via:
1. Taking the ~500,000 document vectors with 300 features and normalise them for better SOM performance.
1. Initialising a 1 x 839 set of nodes, 2D SOM, with randomly generated weights.
1. Training a SOM on this using cosine-similarity as our distance metric to obtain the BMU.

After training the SOM, have mapped these ~500,000 samples into (1*839=)839 nodes. In this way, Swe have mapped from X to the reduced space, Y, where X is of ~500,000 and Y is of 839.

We then compute the coordinates of the winning neuron, BMU, from the set of 1 x 839 nodes, and convert these coordinates into an index which act as our clusters. This gives us a `cluster_index` for each document vector which we can then column-bind to our each `base_path` or `text` so we know what cluster each GOV.UK page belongs to.

[stackoverflow, ASantosRibeiro 2014](https://stackoverflow.com/a/26926949/13416265)

[superdatascience, 2018](https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms)

***

Thinking now is that we have identified 100 clusters from our 294 node mapping of our 3462 samples dataset of document vectors.
- Want to then visualise these 100 clusters and possibly have a drill-through (stretch goal) or extract list of each document vector relating to each category.

## Plotting
Due to SOMs being a dimensionality-reduction technique, they can be used to create clusters for identifying semantically-similar content. This can then be visualised by a U-Matrix of the SOM, which illustrates the Euclidean distance between weight vectors of neighbouring cells.

In terms of interpreting a U-Matrix, a good explanation is provided [here](https://stackoverflow.com/a/13642262/13416265). In particular, the colours on the U-Matrix represents how close the neurons in the Y map are, with lighter colours representing smaller distances between neurons and darker colours representing larger distances between neurons. In this way, the light shades of a U-Matrix signify clusters of similar neurons whereas darker shades are cluster boundaries. These are clusters of the SOM nodes.
