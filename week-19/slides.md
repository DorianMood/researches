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

# Existing methods

There are existing methods to compress images using convolutional architectures. The main concept here is to remove all the operations that cause a loss of information (usually those are all sorts pooling, interpolation etc.).

---

# Fully Convolutional Networks for Semantic Segmentation

An example of use a fully convolutional networks

![](fcn.png)

---

# Learning Convolutional Networks for Content-weighted Image Compression

Image compression with convolutional networks. Blue blocks are up/down-sampling.

![](content-weighted.png)

---

# Full Resolution Image Compression with Recurrent Neural Networks

![](full-resolution-recurrent.png)

---

# Model I did

I implemented a simple model of encoder/decoder architecture. I used `torch.nn.TransposeConv2d` for upsampling.

---

# Encoder

![](encoder.svg)

---

# Decoder

![](decoder.svg)

---

# Results

<div class="twocols">

## Original

![width:400px](original.png)

<p class="break">

## Reconstructed

![width:400px](reconstructed_64.png)

</div>

---

# Results (channel size 64)

<div class="twocols">

## Original

![width:400px](original_32.png)

<p class="break">

## Reconstructed (channel size 32)

![width:400px](reconstructed_32.png)

</div>

---

# Academic activities


![](mail.png)