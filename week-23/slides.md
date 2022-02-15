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

# Generator

We are working on generator improvements. Possible improvements are:

1. Reduce size of generator model
2. Improve accuracy of generator

---

# Generator structure

Following generator is widley used in a couple of papers on image compression.

```
c3s1-960, R960, R960,
R960, R960, R960, R960, R960, R960,
R960, u480, u240, u120, u60, c7s1-3
```

---

# Training details

We use one part of OpenImages dataset (~160,000 images) to train and validate.

For now we've only completed one epoch.

---

# Results

Results are just normal, which means we only have proven existing methods work well.

We need to make a minor improvements in generator architecture to aceave a higher decompression quality.
