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

# References