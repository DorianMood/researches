---
bibliography: [index.bib]
title: Nikita weekly report
subtitle: Nikita weekly report
author:
  - name: Nikita Dolgoshein
    affiliation: Beihang University
    location: Beijing
    email: nikita01051997@gmail.com
csl: [ieee.csl]
autoSectionLabels: True
---

# Literature review

Currently graph neural nets are able to take into account only k-hop neighbors. But it's clear that this approach can be quite unfair. In [@wang_atpgnn_2021] authors built such an architecture that it can consider both local k-hop neighbors structure and distant nodes that are located in similar topological context. What do they do is basically perform three independent and completely different embedding procedures (discussed below) and then make a decision based on the embeddings results.

# 1. ATPGNN: Reconstruction of Neighborhood in Graph Neural Networks With Attention-Based Topological Patterns

First let me show their architecture [@fig:atpgnn_architecture]. Knowing general idea it would be easier to explain all the deep concepts.

![ATPGNN architecture](atpgnn_architecture.png){#fig:atpgnn_architecture}

Let me start from the output of the algorithm then going to the input. There is an attention layer, similar to GAT [@Velickovic2017], but having three different attention mechanism. The first is feature attention [@eq:atpgnn_feature_attention]. Feature attention doesn't differ from GAT really much.

$$
e^l_{i,j}=\delta([W^l\vec{h}^l_i||W^l\vec{h}^l_j]),v_j\in N_i
$${#eq:atpgnn_feature_attention}

Where $\vec{h}^0_i=\vec{x}^T_i$ and $\vec{h}^0_j=\vec{x}^T_j$.

Next is topology attention [@eq:atpgnn_topology_attention], which contains information about remote similar nodes information.

$$
f^l_{i,j}=\delta([W^{l}\vec{p}^l_i||W^l\vec{p}^l_j]),v_i\in N_i
$${#eq:atpgnn_topology_attention}

Where $\vec{p}^0_i=\vec{s}^T_i$ is a representation of remote node with similar structure, obtained by independently performed message passing.

Let me explain this message passing meaning. We do it to obtain structural information of node local neighborhood, and then it's easy to find nodes with similar representation. In case of linear data structure (here we have just node embedding list) it's a trivial task.

The final formula of convolution is [@eq:atpgnn_convolution]

$$
\vec{h}^l_i=concatenate^K_{k=1}(\sigma(\sum_{v_j\in N_i}A^{k(l-1)}_{i,j}W^{k(l-1)}\vec{h}^{l-1}_j))
$${@eq:atpgnn_convolution}

Here $A$ is graph representation obtained by combining previously mentioned attentions.

Generally speaking concatenation is used in case when we need to separate some groups of features from each other.

Architecture is quite interesting. They combine several architecture to obtain something new. The results [@fig:atpgnn_results], however, are just a little better then plain GCN.

![ATPGNN results](atpgnn_results.png){#fig:atpgnn_results}

We can see that they perform slightly better then just plain architecture. I think in some domains this architecture is applicable and can be used as an example of complex GNN architecture.

# References