---
title: 船新分割架构--MaskFormer
description: 船新分割架构
authors:
  - Asthestarsfalll
tags:
  - PaperRead
  - segmentation
hide_table_of_contents: false
---

>论文名称：[Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/pdf/2107.06278.pdf)
>
>作者：Bowen Cheng， Alexander G. Schwing， Alexander Kirillov
>
>Code：https://github.com/facebookresearch/MaskFormer

## 摘要

1. 经典的语义分割采用逐像素分类的方法，而实例级别的分割（实例分割、全景分割）则使用掩码分类进行预测。作者认为**掩码分类具有足够的通用性**，可以使用完全相同的模型、损失和训练过程以**统一**的方式解决语义级别的和实例级别的分割任务。
2. 提出了 MaskFormer 架构，用于预测一组二进制掩码，每个掩码与单个全局类标签相关。该方法有效地简化了各种分割任务，并取得了良好的实验结果，尤其是当预测类别较大时，MaskFormer 的效果由于逐像素的分割 baseline。
3. ADE20K：55.6 mIoU，COCO：52.7 PQ。

## 介绍

![image-20220713163738338](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713163738338.png)

如上图所示，掩码分类通过预测一组二进制掩码，并为每个掩码预测一个类别，此方式多用于实例级别的分割任务。

在语义分割方面，逐像素分类和掩码分类都得到了广泛的研究。早期的工作中基于 mask classification 的方法性能更好，直到 FPN 大幅度提升了 mIoU 指标。

那么**单个掩码分类模型能否简化语义和实例级分割任务？这种掩码分类模型能否优于现有的逐像素语义分割方法？**

掩码分类方法如 Mask RCNN，DETR 都需要预测边界框，这限制了其在语义分割上的应用。近期提出的 Max Deeplab 使用条件卷积消除了全景分割对于预测框的影响，但是其除了 mask loss 之外，还需要一系列的辅助 loss。

## 从逐像素到掩码分类

掩码分类的几个要点：

1. 预测类别数量为 k+1,多出的一个类别为 no object（∅），表示无类别；

2. 可以同时进行实例分割和语义分割（Note, mask classification allows multiple mask predictions with the same
   associated class, making it applicable to both semanticand instance-level segmentation tasks.）;

3. 训练过程中，通常预测数量大于标签数量，因此使用∅进行填充以保持一一对应；

4. 对于语义分割，当预测数量 N 与标签数 K 相同（？），可以采用固定匹配的策略，第 i 个预测对应第 i 个类别，如果某个类别在标签中不存在，则对应∅，这里即表示特征图中是否含有这个类；

5. 实验发现，基于二分匹配的分配方式效果更好。

   ## MaskFormer

   ### 结构

   ![image-20220713170821204](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713170821204.png)

三个模块：像素级模块、transformer 模块和分割模块。

**像素级模块：**

backbone 用于生成低分辨率的特征图，本文中为 32 倍下采样，decoder 进行上采样，生成与输入图像相同大小的 per-pixel embeddings。

**transformer 模块：**

使用标准的 transformer 解码器，输入包含 backbone 输出的低分率特征图，以及 N 个可学习的 positional embeddings（N queries）。

**分割模块：**

对 transformer 的结果使用 MLP 以及 softmax，生成类别概率和掩码嵌入（mask embeddings）。

将 mask embedings 与 per-pixel embeddings 对应相乘和 Sigmoid 即可得到最终的二进制掩码预测。

注意，经过实验发现使用 Sigmoid 而不是 Softmax 效果更好。

### 推理

**General inference：**

对于 N 个预测结果的中的每一个使用对应概率最高的类别直接与掩码相乘，得到概率类别的 mask。

$$
argmax_{i:c_i\neq \varnothing}p_i(c_i)\cdot m_i[h, w]
$$

对于语义分割，直接将相同类别的 mask 合并，对于实例级分割，预测结果天然拥有索引，对于全景分割，首先过滤低置信度的预测，再删除被其他预测掩码遮挡的掩码（以 0.5 二值化）。

**Semantic inference：**

根据经验发现，对掩码概率的边缘化比 General inference 的硬像素分配拥有更好的结果。

$$
argmax_{c\in \{1,...,K\}}\sum_{i=1}^N p_i(c)\cdot m_i[h, w]
$$

## 消融实验

在与 MaskFormer 架构完全相同的 pre-pixel baseline 上进行了实验：

![image-20220713175305000](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713175305000.png)

探究固定匹配和二分图匹配：

![image-20220713175332073](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713175332073.png)

探究 queries 的数量：

![image-20220713175402913](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713175402913.png)

推理策略：

![image-20220713175500372](C:/Users/11864/AppData/Roaming/Typora/typora-user-images/image-20220713175500372.png)
