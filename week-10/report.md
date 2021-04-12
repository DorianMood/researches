---
bibliography: [index.bib]
title: Nikita 道尔格 LS1906205
subtitle: weekly report
author:
  - name: Nikita Dolgoshein
    affiliation: Beihang University
    location: Beijing
    email: nikita01051997@gmail.com
csl: [ieee.csl]
autoSectionLabels: True
---

# 1. Scene graphs: A survey of generations and applications

In [@Chang_Ren_Xu_Li_Chen_Hauptmann_2021] authors give a short introduction of scene graph generation models and approaches.

Scene graph obtained after Scene Graph Generation (SGG) has a variety of applications. Those are just a few of them:

1. visual-textual transformers.
2. image-text retrieval.
3. visual question answering (VQA).
4. image understanding and reasoning.
5. 3D scene understanding.
6. human-object interaction (HOI) and human-human interaction (HHI).

There are two main directions to work on:

1. *Improving accuracy*. Improve existing approaches in SGG.
2. *Utilizing a prior knowledge*. Images already contain a lot of information, which can be utilized as a prior knowledge.

The general pipeline of graph scene generations consists of following steps:

1. Object detection. On this step primary detection happens. Objects and their bounding boxes are extracted.
2. Object-relation-subject SGG triplets are constructed from primary detected objects.
3. One from the variety of Scene Graph Generation (SGG) algorithms is applied.

Since the first step is an independent CV field, one of the existing algorithm can be used, such as Faster RCNN [@Ren_He_Girshick_Sun_2016].

For the second and third steps different algorithm are being used. Next authors describe some of them:

**CRF-based SGG** is a classical tool, which has been widely used before, but now is not really popular.

**TransE-based SGG** regards the relationship as a translation between the head entity and the tail entity. Relationship is modeled as a simple vector transformation. There are two modules: visual module which extracts visual features and language module which labels triplets with corresponding values.

**CNN-based SGG** uses Spatial, Visual and Semantic modules to learn corresponding features of triplets. There are many different approaches here, one of the key problems is a computational complexity, which need to be reduced.

**RNN/LSTM-based SGG** utilizes recursive architecture to label triplets. There are many different architectures. Some use visual attention, some use context information fusion.

**Graph-based SGG** uses Graph convolution to generate a scene graph. First object regions proposals are detected and being used as nodes in graph.

Let me give a small introduction on methods used specifically in Graph-based approaches:

1. Factorizable Net [@Li_Ouyang_Zhou_Shi_Zhang_Wang_2018] builds a fully connected graph from object proposals, then merge edges corresponding to similar regions.
2. Graph R-CNN [@Yang_Lu_Lee_Batra_Parikh_2018] does it similarly taking object proposals, building fully-connected graph, then pruning edges and utilizes GAT to label relations.

There is a prior knowledge we can extract from image. Language prior is quite obvious and it stands for the fact that some object could "borrow" relation information from similar objects. For example "girl rides horse" is quite similar to "girl rides elephant" or "woman rides elephant". Another prior is statistical prior, which basically stands for adequateness of predicted relation triplet. For example "cat eats fish" is quite probable prediction, however "fish eats cat" is not really probable. The last prior is knowledge graph, which stands for real world ground-truth data. In other words, according to knowledge graph we can understand could given two objects relate to each other or not. 

Prior knowledge has been proven to significantly improve the quality of scene graph generation.



# Reference