---
title: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation
description: 导航
tags:
  - RL
  - EmbodiedIntelligence
hide_table_of_contents: false
---

> Paper: [VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation](https://arxiv.org/abs/2312.03275)
>
> Page: http://naoki.io/vlfm
>
> Code: https://github.com/bdaiinstitute/vlfm

## Abstract

理解人类如何利用语义知识在不熟悉的环境中导航并决定下一步探索的位置，对于开发具有类似人类搜索行为的机器人至关重要。

本文介绍了一种零样本导航方法，视觉 - 语言前沿地图（Vision-Language Frontier Maps，VLFM），该方法受人类推理的启发，旨在导航至新环境中未见的语义对象。

VLFM 利用**深度观测**数据构建占据地图（occupancy maps）以识别可探索的前沿区域（frontiers），并利用 RGB 观测数据和预训练的视觉语言模型（vision-language model）生成基于语言的价值图（language-grounded value map）。随后，VLFM 利用此价值图确定最有希望找到给定目标物体类别实例的前沿区域进行探索。

我们在 Habitat 模拟器中使用 Gibson、Habitat-Matterport 3D (HM3D) 和 Matterport 3D (MP3D) 数据集的照片级逼真环境对 VLFM 进行了评估。引人注目的是，在以路径加权的成功率（Success weighted by Path Length, **SPL**）为衡量指标的物体目标导航（Object Goal Navigation）任务中，VLFM 在所有三个数据集上均取得了最先进（state-of-the-art）的结果。

## Introduction

:::tip

人类在不熟悉环境中导航的过程是复杂的，通常依赖于明确的地图和内在知识的结合。这种内在知识**通常是语义知识的积累**，可用于推断空间布局，包括特定物体的位置和几何配置。例如，我们知道厕所和淋浴通常在浴室中一起出现，且通常靠近卧室。自然语言可以进一步增强这种先验语义知识，具体取决于上下文。

:::

VLFM 从深度观测中构建占据地图以识别已探索地图区域的前沿。为了寻找语义目标对象，**VLFM 提示预训练的 VLM 选择哪个前沿最有可能通向语义目标**。与以往基于语言的零样本语义导航方法相比，VLFM 不依赖于目标检测器和语言模型（例如 ChatGPT、BERT）仅使用文本语义推理来评估前沿。VLFM 使用视觉 - 语言模型直接从 RGB 图像中提取**语义值**，形式是与**涉及目标对象的文本提示的余弦相似度分数**。VLFM 使用这些分数生成基于语言的价值地图，用于识别最有希望探索的前沿。这种基于空间的联合视觉 - 语言语义推理提高了计算推理速度和整体语义导航性能。

## Related Works

**目标导航（ObjectNav）**

目标导航涉及在新环境中执行语义目标驱动的导航，其性能主要通过机器人到达给定目标对象类别的实例的路径效率来衡量。这基于有效利用语义先验应使机器人能够更高效地定位对象的前提。训练具有语义导航能力的机器人的学习方法通常利用强化学习、从演示中学习 或预测语义俯视地图，在这些地图上可以使用航点规划器。然而，这些特定任务的训练方法仅适用于它们所训练的封闭对象类别集，并且通常仅在模拟数据上进行训练，这可能会阻碍这些策略在现实世界平台上的部署。

**零样本目标导航（Zero-shot ObjectNav）。**

近期关于零样本目标导航方法的研究涉及改进由《A Frontier-Based Approach for Autonomous Exploration》提出的基于边界（frontier-based）的探索方法。基于边界的探索是指访问地图上已探索和未探索区域之间的边界，该地图由智能体在探索过程中迭代构建。

人们提出了许多选择下一个要探索的边界的方法，例如基于智能体预期获得信息量来选择边界的经典方法（Histogram Based Frontier Exploration，Learning-Augmented Model-Based Planning for Visual Exploration）。“轮子上的 CLIP”（CLIP on Wheels, CoW）采用了一种简单直接的方法：机器人探索最近的边界，直到使用 CLIP 特征或开放词汇物体检测器检测到目标物体。LGX 和 ESC 使用大型语言模型（LLM）处理以文本形式呈现的物体检测结果，以识别哪些边界最可能包含目标物体的实例。SemUtil 则没有使用 LLM，而是使用 BERT 来嵌入在边界附近检测到的物体类别标签，然后将它们与目标物体的文本嵌入进行比较，以选择接下来要探索的边界。

:::info

然而，这些方法引入了一个瓶颈：在能够语义评估边界之前，来自环境的视觉线索必须通过物体检测器转换为文本。此外，依赖 LLM 需要大量的计算资源，这可能需要机器人连接远程服务器。相比之下，**VLFM**使用一个视觉语言模型（VLM），该模型可以轻松加载到消费级笔记本电脑上，直接从 RGB 观测数据和文本提示生成语义价值评分，而无需从视觉观测生成任何文本。

:::


