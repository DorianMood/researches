---
marp: true
---

# End-to-End Object Detection with Transformers


## Nikita LS1906205 report

---

# Table of content

1. Introduction
2. Transformer recap
3. Architecture
4. Visual attention
5. Intuition

---

# 1. Introduction

- Object detection
- Transformers and attention
- DETR parallelism
- Duplicates problem

---

# 2. Transformer recap

- Encoder-Decoder architecture
- Parallel encoder
- Attention

---

# 2. Transformer recap
## Self-attention

$$
Attention(Q,K,V)=softmax(\dfrac{QK^T}{\sqrt{d_k}})V
$$

---

# 2. Transformer recap

![](self-attention.png)

---

# 2. Transformer recap

![height:750px](transformer.png)

---

# 2. Transformer recap

![](transformer-encoder.png)

---

# 3. Architecture

![Architecture](../week-6/transformer-object-detection-architecture.png)

---

# 3. Architecture

![Deep architecture](../week-6/transformer-object-detection-architecture-deep.png)

---

# 4. Visual attention

![Self-attention map](../week-6/transformer-object-detection-architecture-self-attention-map.png)

---

# 4. Visual attention

![Image clustering](../week-6/transformer-object-detection-architecture-image-clustering.png)

---

# 4. Visual attention

![Image clustering](../week-6/transformer-object-detection-image-clustering.png)

---

# 5. Intuition

![Queries visualization](../week-6/transformer-object-detection-queries.png)

---

# 5. Intuition

![Object detection results](../week-6/transformer-object-detection-results.png)

---

# 6. Conclusion

![Code](../week-6/transformer-object-detection-code.png)

---

// put loss, put some text, some formulas. The information should be complete within a presentation.

// background overview and motivation, contribution, other works review.

// Stanford university online course on CNN

// CS231n