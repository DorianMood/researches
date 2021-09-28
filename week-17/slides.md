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

 - Layout to image model was chosen as a base for my model
 - I managed to reproduce their results using pretrained model
 - I failed to train their model using their code and hyperparameter values

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

# Adversarial training

