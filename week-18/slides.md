---
marp: true
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

# Previous work

 - Results were reproduced
 - Decreasing latent code $z$ loss and $KL$ loss lead to increasing accuracy of reconstruction

---

# Detailed model structure

![h:600](layout2im-model.jpg)

---

# Crop encoder

![](crop-encoder.svg)

---

# Layout encoder

![](layout-encoder.svg)

---

# Decoder

![](decoder.svg)

---

# Loss structure

![h:600](layout2im-loss.jpg)

---

# Separate training pipelines

![](layout2im.jpg)

---

# Training

Thy used adversarial training approach employing two discriminators: $D_O$ and $D_I$ to discriminate real/fake objects  and images correspondingly.

## Original paper losses

1. Discriminator losses: $L_{D_O}$, $L_{D_I}$, $L_{D_{cls}}$
2. Generator losses: $L_{G_I}$, $L_{G_z}$, $L_{KL}$, $L_{D_O}$, $L_{D_I}$, $L_{D_{cls}}$

## My losses

1. Discriminator losses: $L_{D_O}$, $L_{D_I}$, $L_{D_{cls}}$
2. Generator losses: $L_{G_I}$, $L_{D_O}$, $L_{D_I}$, $L_{D_{cls}}$

---

# Proposed architecture

![w:1500](architecture.svg)

---

# Email to school

![](email.png)