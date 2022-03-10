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

# Training

We perform training of our network with different configurations.

We find that 5 residual layers show the highest performance.

---

# Eval

We completed codes for evaluation of a model with several metrics, such as bpp, PSNR and SSIM.

---

# Training details

We use one part of OpenImages dataset (~160,000 images) to train and validate.

One epoch takes ~3h. We loop 10 epochs compression + 7 epochs GAN models.

---

# Evaluation results

Let us move to detailed results analysis.

---

![](compression_10.png)

---

![](compression_10_metrics.png)

---

![](gan_4.png)

---

![](gan_4_metrics.png)

---

![](gan_5.png)

---

![](gan_5_metrics.png)

---

![](gan_6.png)

---

![](gan_6_metrics.png)

---

![](results.png)

---

![](metrics.png)
