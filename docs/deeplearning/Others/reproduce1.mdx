---
title: 快速上手框架，从零复现论文——准备篇
description: 丢人咯
authors: [Asthestarsfalll]
tags: [others]
hide_table_of_contents: false
---

本文将会带你使用 MegEngine 框架复现 RepVGG，请确保你对 pytorch 等至少一个深度学习训练框架足够了解。

## 为什么要复现论文

为什么要（使用其他框架）复现论文？

1. 赛题限定了框架，比如各大深度学习框架公司发起的比赛（奖金在召唤），不得不用；
2. 导师要求使用其他框架复现论文，不得不复现；
3. 白嫖算力，各大框架的公司为了推广自己的框架会提供一些线上的算力支持，比如百度的 AIStudio、旷视的 MegStudio、华为的 ModelArts 等；
4. 深入理解论文，提升编程能力，论文复现带能够提升对论文的理解程度和细节的关注程度，同时对框架源码的理解程度和编程能力的提升也有所帮助。

## 准备工作

当我们决定要**快速**复现一篇论文时，我们应该做什么呢？

1. 数据集/源码：该篇论文所使用的的数据集/源码是否是开源的，或者是否能向作者申请到；
2. 论文：至少要读过一遍论文，知道该工作基于什么（引用了之前的什么工作，比如主干使用了 ResNet，neck 使用了 FPN 等，这点有助于我们更快速的复现论文，**避免造轮子**），有什么创新和贡献等；
3. 源码：大概看一下，了解代码结构，有什么复现难点（比如论文使用了 cuda 编写的算子，这就需要你自己改写编译，构建到模型中，了解难点有助于我们**分配优先级**，**及时止损**）
4. 环境配置：将官方代码所需的环境和复现所需环境配置好，各个框架安装方式在官网都有教程；
5. 框架：各个框架都有快速入门教程，大致阅读一下，了解简单的 api，比如 tensor 的创建、数据类型的声明（比如 torch 使用 `torch.float`，Paddle 使用 `'float32'`，MegEngine 使用 `np.float32`）、组网的基类（`nn.Moudle`, `nn.Layer`, `M.moudle`）等；
6. 训练模板：各个框架官方都有 modelhub、modelzoo 之类的仓库，或者是一些开发套件，诸如 pytorch 的 mmcv 系列（mmdetection、mmsegmentation、mmclassification 等等等），paddle 的 paddledetection，paddleseg 等等，megengine 的 basecls，我们需要找到与复现论文相近的任务和数据集（比如图像分类和 ImageNet）的训练代码，备好以便后续修改使用；
7. API 文档：需要同时打开源码所使用的框架和目标框架的 API 文档，即用即查，务必保持对齐，“**对齐**”是论文复现的关键，包括模型结构对齐、loss 对齐、优化器对齐、学习率对齐、训练策略对齐等等等，当然可以根据自身进行选择。

## get started

既然知道了做什么接下来便开始行动吧。

[RepVGG]( https://github.com/DingXiaoH/RepVGG ) 及其数据集 ImageNet 都是开源的，论文阅读笔记在之前的博客有所介绍，可惜的是 RepVGG 没有使用之前的工作，如 ResNet 等（不然就可以直接 ctrl c v 了），还好源码较为简单，难度不是很大。

训练模板也在 MegEngine 官方提供的 Modles 仓库中找到，本文将使用 resnet 的 [训练代码](https://github.com/MegEngine/Models/blob/master/official/vision/classification/resnet/train.py)，后续按照需要修改。

同时打开了 pytorch 的 [官方文档](https://pytorch.org/) 和 MegEngine 的 [官方文档](https://megengine.org.cn/doc/stable/zh/reference/index.html)。

万事俱备，下一篇博客将会介绍模型的复现过程。
