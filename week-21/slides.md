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

# I cover

 - Image compression pipeline
 - JPEG
 - Arithmetic coding

---

# Model training results

Open [link](http://localhost:6006/).

---

# Model compression results

<div class="twocols">

## Original

![](orig.png)

<p></p>

## Reconstruction

![](rec.png)

</div>

---

# My thoughts

1. I am not able to do a good project in one month
2. I can just modify one of the existing frameworks (no guarantee I can get better results than original)

---

# Some ideas on my project #1

1. FCN. Feature extraction.
2. Object detection.
3. Crop features.
4. Compress objects separately using HiFIC model.
5. Store compressed data.
6. Decompress objects using HiFIC model.
7. Compose them into one tensor.
8. FCN. Deconvolution to restore image.

---

# Some ideas on my project #2

1. Object detection to extract layout.
2. Use layout2im model for reconstruction.
