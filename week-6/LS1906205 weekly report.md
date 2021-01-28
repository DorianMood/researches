---
bibliography: [index.bib]
title: Nikita weekly report
subtitle: Nikita weekly report
author:
  - name: Nikita Dolgoshein
    affiliation: Beihang University
    location: Beijing
    email: nikita01051997@gmail.com
csl: [ieee.csl]
autoSectionLabels: True
---

# Literature review

There were many approaches to apply deep-learning models on video domain. It is not as easy as it seems to be however. Naive approach is just to add an extra dimension (this is called 3D convolution). Such an operation is extremely hard to perform on real-world devices, it takes too much computational resources. The second problem with naive approach is that connectivity patterns in temporal dimension could be more complicated and therefore a more flexible architecture is needed. Authors in [@karpathy_large-scale_nodate] propose standardized approaches to build such a system that can capture temporal dimension features more accurate then models used before. They acheaved up to 63% accuracy on their dataset compared to 43% baseline model.

While [@karpathy_large-scale_nodate] enjoys temporal dimension as-it-is, in [@lin_tsm_2019] authors propose an architecture to separate channels and shift some of them along time axes. 

# 1. Large-scale Video Classification with Convolutional Neural Networks

Authors tried to solve task of video recognition given a video clip. There were several questions needed to answer, one of them was "What temporal connectivity pattern in CNN is the best at taking advantage of local motion information presented in video [@karpathy_large-scale_nodate].

The standard approach usually follows this scheme:

1. Extract local features that describe a region of video.
2. Then from features we got we can build an entire video representation.
3. Lastly apply some kind of Machine learning technique such as SVM etc.

On the [@fig:large_scale_approaches] you can see main approaches proposed by authors. So, basicaly this figure shows different approaches to select target frames for convolution.

![Different approaches to capture temporal information](large-scale-video-recognition-models.png){#fig:large_scale_approaches}

After convolving we get a low-dimensional data, which then can be used as input for any classifier.

They also propose using two separates flows for better performance as shown on [@fig:large_scale_flows].

![Two flows](large-scale-video-recognition-models-multiresolution.png){#fig:large_scale_flows}

Using fine-tuning technique they acheaved 65% accuracy.

# 2. TSM: Temporal Shift Module for Efficient Video Understanding

In [@lin_tsm_2019] divide regular convolution of time series on two parts: *shift* and *multiply-accumulate*. This makes it possible to change each of those two independently.

We first define a general architecture. Here we shift some of channels as it shown on [@fig:shift-tsm].

![Shift operation on time series](shift-tsm.png){#fig:shift-tsm}

Shift is applied parallel to residual connection, as it shown on [@fig:in-place-residual-tsm]. It's done to stabilize operation.

![Residual connection](in-place-residual-tsm.png){#fig:in-place-residual-tsm}

And finally, we can see that We can cache data in memory [@fig:uni-ditectional-ts]. We can do that because we shift channels along axes, so some information can be reused in next farms computation.

![Uni-directional TSM](uni-directional-tsm.png){#fig:uni-ditectional-tsm}

They aceaved a great performance on the same dataset as [@karpathy_large-scale_nodate], but what is more important, purposed framework makes convolution 10 times faster due to its non-filter nature. We don't use sliding window at all, we shift channels and convolve it linearly instead.

# 3. End-to-End Object Detection with Transformers

# 4. Very Deep Convolutional Networks for Large-Scale Image Recognition

# Reference 