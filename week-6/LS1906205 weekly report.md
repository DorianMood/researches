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

Another very important domain in computer vision is object detection. In [@carion_end--end_2020] they propose an interesting idea to use transformer architecture in object detection. Attention mechanism here quite naturally lies on the logics of bounding boxes detection.

One of the most important milestone papers on CNNs is [@simonyan_very_2015]. Authors here for the first time propose using deep architecture. They also claim that receptive field of 7x7 filter layer can be easily represented as a sequence of 3x3 filter layers. 3x3 layers, however, can be computed much more efficiently.

# 1. Large-scale Video Classification with Convolutional Neural Networks

Authors tried to solve task of video recognition given a video clip. There were several questions needed to answer, one of them was "What temporal connectivity pattern in CNN is the best at taking advantage of local motion information presented in video [@karpathy_large-scale_nodate].

The standard approach usually follows this scheme:

1. Extract local features that describe a region of video.
2. Then from features we got we can build an entire video representation.
3. Lastly apply some kind of Machine learning technique such as SVM etc.

On the [@fig:large_scale_approaches] you can see main approaches proposed by authors. So, basically this figure shows different approaches to select target frames for convolution.

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

They acheaved a great performance on the same dataset as [@karpathy_large-scale_nodate], but what is more important, purposed framework makes convolution 10 times faster due to its non-filter nature. We don't use sliding window at all, we shift channels and convolve it linearly instead.

# 3. End-to-End Object Detection with Transformers



# 4. Very Deep Convolutional Networks for Large-Scale Image Recognition

Authors in [@simonyan_very_2015] propose a general architecture pattern for any deep CNN. To summarize their impact I listed their main contributions:

1. 3x3 layers should be used as a main building block in deep CNN. Convolution with bigger filter size can be decomposed through 3x3 filter stack (with non-linearity injected in between).
2. 1x1 layers can be seen as non-linearity. It takes not a lot to propagate through such layers, because of low computation cost.
3. Batch training is efficient. It is better to average all the gradients within full batch.
4.  The classification error decreases with increased depth of CNN. They figure out that 19 layers in their experiments show dramatically higher performance than 11 layers.

# Reference 