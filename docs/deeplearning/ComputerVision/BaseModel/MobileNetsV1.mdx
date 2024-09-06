---
title: 轻量级网络：MobileNet，从V1到V3         
description: 轻量级网络的巡礼
authors: [Asthestarsfalll]
tags: [PaperRead, deeplearning, computer vision, base model]
hide_table_of_contents: false
---

# MobileNet V1

开门见山，为了实现一个大小与速度都足够优秀的轻量级网络，V1 做了一件事——**将 VGG 中所有标准卷积都替换成深度可分离卷积**。

## 可分离卷积

可分离卷积主要有两种：**空间可分离**和**深度可分离**

[参考](http://www.atyun.com/39076.html)

### 空间可分离卷积

顾名思义，空间可分离卷积将标准卷积核分离成**高度**和**宽度**方向的两个小卷积，下式是一个非常著名的边缘检测算子——$Sobel$：

$$
\left[\begin{array}        
{c}
-1&0&1\\
-2&0&2&\\
-1&0&1\\
\end{array}\right]=
\left[\begin{array}
{c}
1\\
2\\
1\\
\end{array}\right]\times
\left[\begin{array}
{c}
-1&2&1
\end{array}\right]
$$

 ![可分离卷积基本介绍](/images/2022/03/27/2-2.png)

实际运算时，两个小卷积核分别与输入进行三次乘法运算，而非标准卷积的 9 次，以达到减少时间复杂度的目的，从而提高网络运行速度。

然而并非所有的卷积核都可以被分为两个更小的核，在训练时十分麻烦，这意味着网络只能选取所有小内核的一部分，因此空间可分离并没有被广泛应用于深度学习。

### 深度可分离卷积

对于一个特征图而言，其不止有高度和宽度，还拥有表示通道的深度

**标准卷积**：

对于一个标准卷积核，其通道数与输入特征图通道数相同，我们通过控制**卷积核的数量来控制输出的通道数**，如下图

<img src="/images/2022/03/27/image-20210803152057639.png" alt="image-20210803152057639" style={{zoom:"50%"}} />

**深度可分离卷积**：

简单来说，深度可分离卷积 $=$ 深度卷积 $+$​逐点卷积（$Point-wise$)

深度卷积就是使用与输入**特征图通道数相同个数**的**单通道卷积核**，示意图如下：

<img src="/images/2022/03/27/67.png" style={{zoom:"50%"}} />

逐点卷积即是我们经常说的 $1\times 1$​​卷积，其作用是将深度卷积的输出进行**升维**，示意图如下：

<img src="/images/2022/03/27/image-20210803154113917.png" alt="image-20210803154113917" style={{zoom:"50%"}} />

结合上述三副图可以看到深度可分离卷积与标准卷积的输入大小是相同的，那么深度可分离卷积有什么优点呢  ？

我们知道，当输入和输出的通道数都很大时，标准卷积参数量和计算量都是惊人的：

$$
Params:k_w\times k_h\times C_{in}\times C_{out}\tag1
$$

$$
Flops:k_w\times k_h\times C_{in}\times C_{out}\times W\times H\tag2
$$

而对于深度可分离卷积：

$$
Params:k_w\times k_h\times C_{in}+C_{in}\times C_{out}\tag3
$$

$$
Flops:k_w\times k_h\times C_{in}\times W\times H+C_{in}\times C_{out}\times W\times H\tag4
$$

将 $(3)$ 和 $(1)$、$(4)$ 和 $(2)$ 相比，可以得到他们参数量和计算量的比值：

$$
\frac{k_w\cdot k_h+C_{out}}{k_w\cdot k_h\cdot C_{out}}=\frac{1}{C_{out}}+\frac{1}{k_w\cdot k_h}
$$

对于我们常用的 $3\times3$ 卷积核，深度可分离卷积可以降低参数量和计算量到原本的 $\frac{1}{9}$ 到 $\frac18$​左右。

## 网络结构

### 卷积层

<img src="/images/2022/03/27/image-20210803160050704.png" alt="image-20210803160050704" style={{zoom:"50%"}} />

上图左侧是一个常见的 $ConvX：Conv+BN+ReLU$​，对于深度可分离卷积，将卷积的后续处理分别塞进了深度卷积和逐点卷积之后

需要注意的是，这里使用了 $ReLU6=min(max(0,x),6)$，即相对于标准 $ReLU$ 来说，其激活值有一个上限 6，作者认为其**在低精度计算下具有更强的鲁棒性**

### 主体结构

<img src="/images/2022/03/27/image-20210803161201493.png" alt="image-20210803161201493" style={{zoom:"67%"}} />

仅在第一层使用标准 $3\times 3$​卷积，后续堆叠深度可分类卷积，中间会进行不间断的下采样，最后使用平均池化 + 全连接层 +$softmax$ 实现分类任务。

需要注意的是，深度可分离卷积在这里表示成两个层——$Conv\ dw+Conv\ 1\times1$

### 更小的模型

尽管最基本的 MobileNet 已经非常小了，但很多时候特定的案例或者应用可能会要求模型更小更快。为了构建这些更小并且计算量更小的模型，引入了一个超参数 $\alpha\in(0,1] $​​​​，其减少每一层的输入输出通道数以减小模型。这时，整体参数量和计算量如下：

$$
Params:k_w\times k_h\times \alpha C_{in}+\alpha C_{in}\times C_{out}\\
Flops:k_w\times k_h\times \alpha C_{in}\times W\times H+\alpha C_{in}\times \alpha C_{out}\times W\times H
$$

$\alpha$ 用来限制模型的宽度，同时还可以限制输入图像的分辨率，为此，引入了超参数 $\rho\in (0,1]$ ​，其应用在每一层的特征图上，此时模型整体参数量和计算量可以表示为：

$$
Params:k_w\times k_h\times \alpha C_{in}+\alpha C_{in}\times C_{out}\\
Flops:k_w\times k_h\times \alpha C_{in}\times \rho W\times \rho H+\alpha C_{in}\times \alpha C_{out}\times \rho W\times \rho H
$$

## 效果

![image-20210803163120024](/images/2022/03/27/image-20210803163120024.png)

十分优秀

# MobileNet V2

时隔一年，谷歌提出 $MobileNetV2：Inverted\ Residuals\ and\ Linear\ Bottlenecks$​

从论文名称可以看出 $V2$ 的主要改进点——提出了一种新型残差和瓶颈层，此外 $V2$ 使用了全卷积的结构以适应任意尺寸的输入。

## ReLU、数据坍塌和流形

人们在 $MobileNetV1$​​中发现了一些问题——很多深度卷积核炼出来是空的，即有很多 $0$​

作者认为是 $ReLU$ 的问题，给出了以下实验：

![image-20210803204325265](/images/2022/03/27/image-20210803204325265.png)

可以看到输入是一个在二维空间内的螺旋线，通过 $N$ 维的随机矩阵 $T$ 将其嵌入到 $N$ ​维空间之中，使用 $ReLU$ 函数最后使用 $T^{-1}$ ​将其映射回 2 维空间，我们可以得到公式：

$$
X_{out}=ReLU(TX_{in})T^{-1}
$$

作者在这里表明：**如果输入流形可以嵌入到激活空间的一个低维子空间中，那么 ReLU 变换保留了信息，同时将所需的复杂性引入到可表达函数集中**，其实和上面的总结意思相同，激活空间便是 $ReLU(TX_{in})$ 所处的空间，作者将其嵌入回到二维空间内，此时若信息保存良好，则认为流形被嵌入进**激活空间的一个低维子空间中**。当嵌入空间维度过低时，会丢失大量信息，也就是数据坍缩，而当嵌入维度较高时，信息则会得到良好的保存

**总结**：简单来说 ，这个实验表明，由低维度（通道数）到高维度之后使用 $ReLU$​​​​​​函数，信息会得到良好的保存；当维度变化很小时，则会丢失大量信息。这也解释了深度卷积核是空的原因。

**Insight**：

作者通过这个实验提出了两点见解（$insight$）：

1. If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation.（如果在 ReLU 之后兴趣流形仍然保持非零体积，则认为它对应线性变换）
2. ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space.（ReLU 函数拥有保存输入流形完整信息的能力，但是仅限于输入流形位于输入空间的低维子空间内）

首先，我们需要了解**流形**：

流形是几何学和拓扑学中重要的概念，在数据科学中，我们认为数据在其嵌入空间中形成低维流形，也就是说，我们所观察到的数据是由一个**低维流形映射到高维空间上的**，在高维空间存在一定的冗余，数据可以在更低的维度表示。

关于流形的一个最典型的应用就是分类任务，其目的就是从根本上分离出一堆混乱的流形，这也被称为**流形学习**。

那么上述两个见解是什么意思呢？笔者仅根据自己的理解解释：

1. 这里可能是将 $ReLU$ 近似的看作线性变换，也为下文提出 $Linear\ Bottleneck$ 做一定的铺垫 ；
2. 我们已经知道数据实际上是存留在数据空间的低维流形上，当输入的维度（在网络中表现为通道数）很低时，此时，输入流形并不符合**处于输入空间的低维子空间内**，因此会造成大量的信息损失，反过来说，输入流形并不能嵌入到激活空间的一个低维子空间中，$ReLU$​并不能保留大量信息，作者于是提出了一种 $Linear\ Bottleneck$​​​​用于升维，之后再使用 $ReLU$ 函数，就能一定程度上保留信息了。

## Inverted Residuals with Linear Bottlenecks

为了解决 $ReLU$​对低维输入造成的数据坍缩，作者提出了 $Inverted\ Residuals\ with\ Linear\ Bottlenecks$​，与标准的 $Linear\ Bottlenecks$​不同的是，其对输入先进行升维再进行降维。
