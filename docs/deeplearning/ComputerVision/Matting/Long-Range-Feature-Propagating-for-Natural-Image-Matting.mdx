---
title: Long-Range-Feature-Propagating-for-Natural-Image-Matting
description: 基于远程特征传播的自然图像抠图
authors:
  - Asthestarsfalll
tags:
  - PaperRead
  - matting
hide_table_of_contents: false
---

> 论文名称：[Long-Range Feature Propagating for Natural Image Matting](https://arxiv.org/ftp/arxiv/papers/2109/2109.12252.pdf)
>
> 作者：Qinglin Liu, Haozhe Xie, Shengping Zhang, Bineng Zhong, Rongrong Ji
>
> Code：

## 摘要

1. 自然图像抠图从三分图（`trimap`）中的未知区域来预测 `alpha`。近来，基于深度学习的方法根据已知区域（`known region`）与未知区域（`unknown region`）的相似性将 `alpha` 从已知区域传播到未知区域。然而我们发现，受限于普通卷积神经网络的较小感受野，未知区域中有超过 50% 的像素不能与已知区域相关联起来，这会导致预测不准确。
2. 本文提出了 Long-Range Feature Propagating Network (LFPNet)，其能够在感受野之外学习长距离语义信息。实验结果表明，它在 AlphaMatting 和 Adobe Image Matting 数据上实现了 SOTA。

## 介绍及相关工作

### matting problem

数学上，一张图片 $I$ 可以被建模为前景 $F$ 和背景 $B$ 的线性组合:

$$
I_i=\alpha_iF_i+(1-\alpha_i)B_i
$$

然而，上式只有 $I$ 是已知的，因此这是一个定义不明确的（`ill-defined`）

问题。

为了解决这个问题，多数方法需要一个 `trimap` 来表示一张图片中已知的前

景、背景和未知的部分。

这些方法可以分为三类：基于采样（sampling-based），基于传播

（propagation-based），基于学习（learning-based）。

### Sampling-based methods

基于采样的方法基于颜色关联的假设，从 `已知区域` 的 `前景` 和 `背景` 中为 `未知区域` 中的每一个像素选取最优的颜色对（`color pairs`）来进行预测。

这些方法通常会设计一个指标来衡量已知像素与未知像素的相似度，如:

- 使用线性函数通过已知像素来得到未知像素的 `alpha`；
- 使用静态方法对前景和背景颜色进行聚类；
- 使用贝叶斯定理最大化后验概率；
- 使用临近区域（region）空间信息进行预测；
- 使用边界（boundary）空间信息加速预测；
- 同时考虑颜色和空间信息，并使用稀疏编码建立一个目标函数等……

### Propagation-based methods

基于传播的方法根据已知区域和未知区域的相似性将 `alpha` 从已知区域传播到未知区域，其通常使用局部平滑的先验（`priori of local smoothing`）设置一个代价函数来实现传播，如：

- 使用泊松方程进行预测；
- 基于 `alpha` 的封闭解建立线性颜色模型进行预测；
- 使用大尺寸的拉普拉斯矩阵（Laplace matrix）进行预测；
- 在高维空间中使用 KNN 进行预测；
- 结合局部和全局信息进行预测等……

### Learning-based methods

上述的两种传统方法对前景和背景的颜色分布重叠（`color distributions overlap`）都十分敏感（比如前景和背景颜色很接近的情况），然而这在自然图像中是十分常见的，而基于学习的方法可以同时学习图像中的颜色分布和自然结构，从而获得更好的表现。

- DIM 提出了第一个大规模的抠图数据集和第一个端到端的模型；
- SampleNet 使用前背景信息来监督网络进行学习；
- AdaMatting 预测时优化 trimap 的表示；
- IndexNet 通过学习下采样过程来优化上采样的恢复精度；
- GCAMatting 设计了一个语义指导注意力模块捕捉语义信息；
- FBAMatting 设计了一个网络同时预测前景背景和 alpha 并使用一阶贝叶斯公式来优化结果等……

然而上述的所有方法都只能学习到局部的图像特征，因此提出了 Long-Range Feature Propagating for Natural Image Matting.

## Motivation

基于传播的方法通常根据已知区域与未知区域的相似度进行传播和预测，对 Adobe Image Matting 数据集中已知与未知区域的前背景像素计算欧氏距离，结果如下：

<img src="/images/2022/03/27/image-20211228171252233.png" alt="image-20211228171252233" style={{zoom:"50%"}} />

其中：

- 平均最短的距离为未知背景到已知背景，超过半数的像素之间的最近距离超过 58px；
- 平均最长的距离为未知背景到已知前景，超过半数的像素之间的最近距离超过 167px，这远远超过了常用网络的感受野，如 ResNet 的 75px；
- 此外，25% 的已知与未知像素之间的最短距离超过了 500px。

显然，我们需要考虑感受野之外更长距离的信息。

## Method

LFPNet 是一种基于 patch、crop 和 stitch 的方法，即将输入的图片和 trimap crop 为一个个 patch，对每个 patch 进行预测 alpha 值，最后再 stitch 起来。

本网络使用两个输入，分别被称为 `inner center patch` 和 `context patch`，后者面积为前者的四倍。

LFPNet 包含两个模块——`propagating module` 和 `matting module`，`propagating module` 以 `context patch` 作为输入，生成 `context feature` 和 `context alpha matt`；`matting module` 生成 `inner center alpha matt`、`inner foreground\background patch`。

<img src="/images/2022/03/27/G2gD3s6NEdSzrTA.png" alt="image-20211228170835998" style={{zoom:"50%"}} />

### Propagating Module

该模块主要负责获取长距离信息，由三部分组成：the context encoder, Center Surround Pyramid Pooling, the context decoder.

#### Context Encoder

语义编码器旨在以低算力消耗提取语义信息，为了实现该目的：

1. 使用 `bicubic interpolation` 和 convolution 对输入的 `context patch` 进行下采样；
2. 使用 ResNet-50 作为主干网络；
3. 为了继续增加感受野，将 ResNet-50 中 block3 和 block4 中的卷积替换为空洞卷积，参数分别为 2 和 4.

####  Center-Surround Pyramid Pooling

<img src="/images/2022/03/27/image-20211228175916841.png" alt="image-20211228175916841" style={{zoom:"50%"}} />

CSPP 包含两个部分: Center Surround Pooling 和 Atrous Spatial Pyramid Pooling

**Center-Surround Pooling (CSP)**
