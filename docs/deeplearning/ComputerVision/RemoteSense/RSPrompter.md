---
title: Learning to Prompt for Remote Sensing Instance Segmentation based on Visual Foundation Model
description: emmm
authors:
  - Asthestarsfalll
tags:
  - RemoteSense
hide_table_of_contents: false
---

> 论文名称：[RSPrompter: Learning to Prompt for Remote  Sensing Instance Segmentation based on Visual  Foundation Model](https://arxiv.org/abs/2306.16269)

> 代码： https://github.com/KyanChen/RSPrompter

> 主页： https://kychen.me/RSPrompter/

## 摘要

1. 作为一种类别不可知的实例分割方法，SAM 在很大程度上依赖于先前的手动指导，包括点、框和粗粒度掩码。此外，其在遥感图像分割任务中的性能仍然未被广泛探索和验证；遥感图像的背景复杂，场景多样性，以及缺乏明确定义的对象边缘，这些都对 SAM 的分割能力提出了挑战；
2. 借鉴提示学习的思想，我们提出了一种学习方法，为 SAM 生成适当的提示。这使得 SAM 能够为遥感图像产生语义可辨识的分割结果，我们将其称为 RSPrompter。我们还提出了几种针对实例分割任务的持续衍生方法，借鉴了 SAM 社区的最新进展，并与 RSPrompter 的性能进行了比较。

## 介绍和相关工作

### 实例分割

实例分割是遥感图像分析中的关键任务，它有助于对图像中每个实例进行语义级别的理解。这一过程提供了关于每个对象的位置（在哪里）、类别（是什么）以及形状（如何）的关键信息。

深度学习算法在遥感图像的实例分割中展现出了显著的潜力，证明了它们从 **原始数据** 中提取深层次、可辨识特征的能力。

目前的算法包括二阶段的 R-CNN，如 Mask R-CNN , Cascade Mask R-CNN, Mask Scoring R-CNN, HTC 和 HQ-ISNet 等，以及一阶段的 YOLACT, BlendMask, EmbedMask, Condinst, SOLO 和 Mask2Former 等。

### 基础模型（Foundational Model）

近年来，在基础模型方面取得了显著进展，如 GPT-4、Flamingo 和 SAM 等，这些模型对社会进步做出了重要贡献。**尽管遥感自诞生以来就具有大数据属性，但迄今为止尚未开发出专门针对该领域的基础模型。**

*本文的主要目的不是为遥感创建一个通用的基础模型，而是研究起源于计算机视觉领域的 SAM 基础模型在遥感图像实例分割中的适用性，并促进遥感领域的持续进步和发展。*
