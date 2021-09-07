---
marp: true
---

# The problem

Make use of SG while having bounding boxes a-priori.

---

# Layout to image

![](pipeline.jpg)

---

# What do they use a Graph representation for

![](architecture.png)

---

# Semantic image generation using scene graph

![](teaser_simsg.png)

---

# Semantic image manipulation using SG

![](sim.png)

---

# The problem

## SGG pipeline:

1. Object detection (bounding boxes, labels, convolutional features)
2. Scene graph generation (relations)

## Image generation pipeline:

1. SG to layout (bounding boxes)
2. Layout to image (image)

---

# What do I have after object detection

![](object%20detection.png)




whether the reconstruction using BBs will fail when the BBs overlaps with specific relations (such as "hold")