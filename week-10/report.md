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

Since the first step is an independent CV field, one of the existing algorithm can be used, such as 

# Reference