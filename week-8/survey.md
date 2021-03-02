---
bibliography: [index.bib]
title: Graph convolutional neural network survey
subtitle: Graph convolutional neural network survey
author:
  - name: Nikita Dolgoshein
    affiliation: Beihang University
    location: Beijing
    email: nikita01051997@gmail.com
csl: [ieee.csl]
autoSectionLabels: True
---

# 1. Introduction

Convolutional neural networks have shown a high performance in several fields during recent years. Power of convolutional neural networks (CNNs) is mainly in their ability to take a local features into account. It turns out that CNN architecture is extremely good and effective when we deal with *structured data*. In early beginning of CNNs one of the milestone works was VGG-19 [@Simonyan_Zisserman_2015] and AlexNet [@Krizhevsky_Sutskever_Hinton_2017], here authors proved that convolutional architectures can be way more computationally effective. It has also been shown that learned convolutional filters can reflect a local structure of sample.

Nowadays CNNs are used in a variety of tasks such as classification, generation, clustering, dimensionality reduction etc. The main disadvantage of CNNs, however, is that it is domain specific. Due to grid nature of CNN, it is quite challenging to apply such architecture on such objects as road map, molecule or even human skeleton etc. Let me remind that all the listed objects are traditionally modeled as some kind of graph. It is clear that it's impossible to apply CNN on graph data directly due to it's non-euclidean nature.

`prepare a picture of graph and example of non-euclidean data`

# 2. Preliminary and background

There are some theoretical knowledge that are required to understand graph neural networks (GNNs). In this part author going to give reader an introduction to graph processing.

## 2.1. Graph representation

Let graph $G$ have a node set $v\in V$ and edge set $e\in E$. Such graph can be represented as square adjacency matrix $A\in \mathbb{Z}^{N\times N}$, where $A_{i,j}=0$ if nodes $v_i$ and $v_j$ are not connected. Otherwise, if nodes are connected, then $A_{i,j}$ is equal to the weight of the edge between nodes $v_i$ and $v_j$.

It is important to notice that if graph $G$ is undirected, then $A$ is *symmetric*. This property is quite useful and it will be used later.

## 2.2. Convolution on graph

Conceptually, convolution operation in CNNs is used to *aggregate* data from fixed neighborhood. For example in pictures it is neighboring pixels (first-order neighbors in case of $3\times 3$ filter, second-order neighbors in case of $5\times 5$ filter etc.). After that an activation function is being applied. This is the simplest pipeline, and, of course, there is quite a bit of optional layers such as pooling or residual [@He_Zhang_Ren_Sun_2015] etc.

After summarizing the main concepts of CNNs we can then start thinking about applying similar architecture on graph domain. Traditionally there are two completely different group of methods in graph learning.

**Spectral methods** are based on graph Laplacian matrix and it's eigen-decomposition (details will be shown next). There are several points to notice at first: Spectral methods are slightly more difficult and intuition is not obvious; To apply spectral convolution a fixed size graph is required for both learning and evaluation process (so, it can be used on such datasets as human skeleton or other fixed-size small graphs); Graph has to be undirected.

**Spatial methods** are more intuitive for those who familiar with CNNs since in these methods neighbor aggregation is almost similar with neighbor aggregation in CNNs. Spatial methods are less computationally effective(???prove???).

# 3. Learning on graphs

## 3.1. Spectral learning

## 3.2. Spacial learning

# Reference 