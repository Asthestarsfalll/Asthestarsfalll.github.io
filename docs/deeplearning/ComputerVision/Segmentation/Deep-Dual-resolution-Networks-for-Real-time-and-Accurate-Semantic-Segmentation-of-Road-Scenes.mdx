---
title: Deep-Dual-resolution-Networks-for-Real-time-and-Accurate-Semantic-Segmentation-of-Road-Scenes
authors:
  - Asthestarsfalll
tags:
  - PaperRead
  - segmentation
description: 双路网络捏。
hide_table_of_contents: false
---

> 论文名称：Deep-Dual-resolution-Networks-for-Real-time-and-Accurate-Semantic-Segmentation-of-Road-Scenes
>
> 作者：Y uanduo Hong, Huihui Pan, Weichao Sun
>
> Code：https://github.com/ydhongHIT/DDRNet

## 摘要和介绍

1. 语义分割是车辆理解周围场景的关键技术，但是其繁重的计算和漫长的推理的时间是自动驾驶所**不能接受**的；
2. 实时语义分割的相关工作大多使用 encoder-decoder 或 two-pathway 的轻量级架构，或是对低分辨率的图像进行预测，虽然能够实现较快的场景解析，但是这些方法与基于 dilation 的主干的模型在**精度表现**上仍有**很大差距**；
3. 为了解决上述问题，提出了一系列（a family）专门为实时语义分割设计的高效主干网络，称为 Deep Dual-resolution Network，DDRNets 由两个分支组成，在这个两个分支会进行多次的双边融合（bilateral fusions）；
4. 设计了一个上下文语义抽取器，被称为（Deep Aggregation pyramidpooling Module），用于扩大感受野，同时基于低分辨率特征图融合多尺度的上下文信息，而推理时间几乎没有增加；
5. SOTA！

<img src="/images/2022/03/27/image-20220118170841596.png" alt="image-20220118170841596" style={{zoom:"67%"}} />

## 相关工作

### High-performance Semantic Segmentation

目前，大多数最先进的（卷积）语义分割模型都是基于 dilation 的主干网络，同时需要网络保持高分辨率以获得良好的性能，然而高分辨率的计算量需求和 dilation convolution 的不充分优化注定了实时语义分割实现高性能的困难性。

<img src="/images/2022/03/27/image-20220118172904733.png" alt="image-20220118172904733" style={{zoom:"80%"}} />

### Real-time Semantic Segmentation

几乎所有的实时语义分割模型都采用了两种基本方法：encoder-decoder 和 two-pathway，同时轻量级的编码器在这两种方法中扮演着重要作用。

1. **编码器 - 解码器结构**

   这种结构很直观的减少了计算量和推理时间，并且其编码器是可以在 ImageNet 上预先训练的轻量级主干网络，也可以是从头训练的高效变体，相关工作有 SwiftNet、FANet、SFNet 等。

   <img src="/images/2022/03/27/image-20220118172921225.png" alt="image-20220118172921225" style={{zoom:"80%"}} />

2. **双路结构**

   编码器 - 解码器结构虽然降低了计算量，但是在重复下采样中丢失了一些信息，为了缓解这个问题，提出了双路结构，除了一个提取语义信息的路径之外，另一条高分辨率的路径作为补充提供丰富的空间细节，相关工作有 BiSeNetv1 v2、Fast SCNN、CABiNet 等。

   <img src="/images/2022/03/27/image-20220118173534151.png" alt="image-20220118173534151" style={{zoom:"80%"}} />

3. **轻量级编码器**

   许多轻量的主干都可以用作编码器——MobileNet、ShuffleNet、Xception 等，然而这些网络包含许多如可分离卷积等**不能高效实现**的组件，这就导致了其理论计算量（FLOPs）可能很低，但是速度并不是很快，另外现有的轻量级骨干网络可能不适合语义分割，因为他们通常对图像分类进行了**过度调整**。

### Context Extraction Modules

语义分割的一个关键是如何捕捉丰富的上下文信息，目前已有的上下文提取模块如 ASPP、PPM 等都是为高分辨率而设计，过于耗时。

## Method

接下来介绍两个主要组件——Deep Dual-resolution Network 和 the Deep Aggregation Pyramid Pooling Module

### Deep Dual-resolution Network

<img src="/images/2022/03/27/image-20220118175649123.png" alt="image-20220118175649123" style={{zoom:"67%"}} />

对一些通用的分类主干网络（如 ResNet）添加额外的高分辨率分支，为了实现分辨率和推理速度之间的均衡，在特征图大小为 $\frac18$ 时添加高分辨率分支，高分辨率分支**不包含任何下采样操作**，并且与低分辨率分支具有一一对应的关系，然后会在不同的阶段进行多次双边融合，如上图所示。

具体看论文。

### Deep Aggregation Pyramid Pooling Module

<img src="/images/2022/03/27/image-20220118190540337.png" alt="image-20220118190540337" style={{zoom:"67%"}} />

DAPPM 的结构如图所示，其会进行不同尺度（$\frac12,\frac14,\frac18,GAP$）的下采样，虽然看起来计算量比较大，但是该模块应用在低分辨率分支，对于 1024×1024 的输入图像，改模块的输入仅为 16×16.  

### overall architecture

<img src="/images/2022/03/27/image-20220118191013459.png" alt="image-20220118191013459" style={{zoom:"67%"}} />

网络的整体结构大致如图所示，其实和 HRNet 比较像，区别就是分支数量不同，并且额外添加了多尺度的特征提取模块，与 HRNet 的对比如下所示：

![image-20220118212416473](/images/2022/03/27/image-20220118212416473.png)
