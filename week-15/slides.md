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

# Summer report
## Nikita 道尔格 LS1906205

---

# The problems I have encountered

 - Building HDF5 datasets to optimize memory usage
 - While trying to fine-tune FasterRCNN on VisualGenome dataset I am getting an unexpected results

---

# Pytorch dataset

Almost finished a best version of Visual Genome dataset for PyTorch.

Currently I use a different one for a couple of reasons:

- Existing `layout to image` is pretrained using a different implementation
- Need a mechanism to convert labels into numeric values and filter out unneeded objects

---

# Training problems

 - I have encountered several problems during training
 - Mainly connected to input data format
 - Cannot overfit

---

# Training loss

![](train-loss.png)

---

# Validation loss

![](val-loss.png)

---

# Test loss

![](test-loss.png)

---

# Mind map

![](mind-map.png)

---

# Semester todo

➖ Attend math class (the last one)
➖ Finish the simplest version of the system
➖ Come up with convolutional features architecture
➖ Prepare materials to publish
➖ Midterm assessment
➖ Look for a job in ML related field
➖ Decide on future life strategy
❔ Come back (hopefully)
