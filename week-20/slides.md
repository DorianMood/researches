---
marp: true
math: katex
---

<style>
div.twocols {
  margin-top: 35px;
  column-count: 2;
}
div.twocols p:first-child,
div.twocols h1:first-child,
div.twocols h2:first-child,
div.twocols ul:first-child,
div.twocols ul li:first-child,
div.twocols ul li p:first-child {
  margin-top: 0 !important;
}
div.twocols p.break {
  break-before: column;
  margin-top: 0;
}
</style>

# Weekly report

## Nikita 道尔格 LS1906205

---

# High-fidelity generative image compression

![](architecture.png)

---

## Scores

![](scores.png)

---

## Curves

![](curves.png)

---

## Channel normalization

![](channel-norm.png)

---

## Model details

- There is an implementation avaliable on GitHub
- Results look reliable
- Architecture is not overcomplicated
- Model size is not big (less than 1Gb)

---

## Architecture

Accourding to proposed GAN nature of the model, there are only two **losses**:

1. Adverserial loss
2. Distortion $L2$ loss

### Modules:

1. Encoder
2. Hyperior
3. Decoder
