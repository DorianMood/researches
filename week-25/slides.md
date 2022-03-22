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

# Deblocking network

For deblocking we used ARCNN and L8 networks. These networks didn't show significant performance growth.

---

# DnCNN network

DnCNN is a popular network, which is based on following architecture.

![](dncnn.png)

---

# IRCNN network

Denoise network arhitecture is based on following architecture. Here residual connections are used.

![](denoiser_arch.png)

---

# Evaluation results

Let us move to detailed results analysis.

We use term "actual mean BPP". We calculate actual mean BPP as an aposteriori BPP mean over entire validation set. Actual BPP in most cases is much lower than the BPP model was trained for.

---

![w:1400](metrics.png)

---

# Conclusion

From these experiments we can see that the highest performance growth was acheaved by IRCNN denoise network.
