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
# The problem

Make use of SG while having bounding boxes a-priori.

---

# Layout to image generated images

## Hand holding banana

<div class="twocols">

![](layout2im-0.svg)

<p class="break">

![](layout2im-1.svg)

</div>

https://github.com/zhaobozb/layout2im

---

# Scene graph to image

![height:500](sg2im-0.png)

https://github.com/google/sg2im

---


# TODO : 

whether the reconstruction using BBs will fail when the BBs overlaps with specific relations (such as "hold")