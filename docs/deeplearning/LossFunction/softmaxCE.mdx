---
title: softmax+交叉熵
description: 分类任务的基础知识——softmax和交叉熵损失函数，本文将会从多个角度介绍softmax和交叉熵损失函数，详细说明交叉熵的由来，网络正向反向传播过程。
authors:
  - Asthestarsfalll
tags:
  - BaseLearn
  - loss
hide_table_of_contents: false
---

在常规的多分类任务中，我们通常使用全连接层将特征通道映射至类别数目，获得每个类别的分数，使用 softmax 获得各类别的概率，以交叉熵作为损失函数优化模型。本文将会简单介绍一下 softmax 和交叉熵函数：

## softmax

$$
y_i=\frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}}
$$

这样的形式使得 softmax 能将输入的分布映射到 $[0,1]$ 之间，一方面 softmax 拉大了输入值之间的差距，因为我们希望模型能够为正确的类别分配更大的概率，而线性归一化的方法难以达到这种效果；另一方面 softmax 与交叉熵函数相结合较为光滑，并且其梯度求解也更为简单。

## 交叉熵

说起交叉熵就不得不谈到概率论里的最大似然估计了，最大似然估计用于样本分布已知时来估计原分布的参数，在这里，我们可以求解模型的参数，回顾一下概率论中的求解步骤：

1. 写出似然函数
2. 取对数
3. 求导数
4. 解方程

似然函数定义如下：

$$
\begin{align}L(\theta|x) &=P(x|\theta )\\
&=P(x_1, x_2,\cdots,x_n|\theta)\\
&= \prod_{i=1}^{n}p(x_i|\theta)\end{align}
$$

P 即是样本的概率，带入取对数

$$
\begin{align} 
log\prod_{i=1}^{n}x_i &= log(x_1\cdot x_2\cdots \cdot x_n) \\ &= log(x_1)+log(x_2)+...+log(x_n) \\ &= \sum_{j=1}^{n}log(x_i)\end{align}
$$

这样看并不直观，比如已知 10 个球中有 6 个为白球，4 个为黑球，假设原分布服从参数为 p 的伯努利分布，p 为白球的概率我们可以得到似然函数为：

$$
L=p^6*(1-p)^4 \tag1\\
$$

$$
log(L)=6log(p)+4log(1-p) \tag2
$$

求导很容易解出来 p=0.6，那么最大似然和交叉熵有什么关系呢？

我们将公式 2 两边同除以 $N=10$，即样本的总数：

$$
\frac{log(L)}{N}=\frac6N log(p)+\frac4 N log(1-p)\\
let \ \frac6N=q,\frac{4}{N}=1-p\\
\frac{log(L)}{N}=plog(p)+(1-p)log(1-p)
$$

由于梯度下降需要最小化损失函数，将上式取负便得到了二元交叉熵损失函数，多元的也是同理：

$$
CE=-\sum q_ilog(p_i)
$$

另一方面，交叉熵也是一种熵，熵是信息论中的概念，能够描述信息的冗余，定义如下是：

$$
Entropy=-\sum q_ilog(q_i)
$$

其实两者十分相似，其中 p 为样本的概率，q 为真实的概率（在实际训练中 q 为 one-hot 或者 smooth 的标签），交叉熵的目的是为了是的**p 尽可能的接近 q**，当二者足够接近时，我们便可以得到熵，这样一来，我们也能从信息论的角度了解交叉熵。

顺便一提 KL 散度其实就是交叉熵减去熵，所以 KL 散度也叫相对熵。

## 梯度求解

在概率论中我们可以直接解出分布的参数，但是对于网络的中的这么多参数我们如何求解呢？

我们使用梯度下降法进行逼近，梯度下降的首要前提便是求解梯度。

softmax+ 交叉熵复合之后的导函数形式也是十分简单，接下来求解 loss 关于输入 $z$ 的梯度。

先求 softmax 的梯度，需要分成两种情况 $i=j$ 和 $i\neq j$，由高中学过的求导法则和大学学过的偏导法则可得

相等时：

$$
\begin{align}
\frac{\partial a_i}{\partial z_i}
&= \frac{\partial(\frac{e^{z_i}}{\sum_{k}e^{z_k}})}{\partial z_i}\\
&=\frac{\sum_ke^{z_k}e^{z_i}-(e^{z_i})^2}{(\sum_ke^{z_k})^2}\\
&=(\frac{e^{z_i}}{\sum_{k=1}^ne^{z_k}})(1-\frac{e^{z_i}}{\sum_{k}e^{z_k}})\\
&=a_i(1-a_i)
\end{align}
$$

不相等时：

$$
\begin{align}
\frac{\partial a_j}{\partial z_i} 
&= \frac{\partial(\frac{e^{z_j}}{\sum_{k}e^{z_k}})}{\partial z_i}\\
&=\frac{-e^{z_j}}{(\sum_ke^{z_k})^2}e^{z_i}\\
&=-a_ia_j
\end{align}
$$

对于交叉熵函数：

$$
\frac{\partial L_i}{\partial a_j}=-y_j\frac{1}{a_j}
$$

根据链式法则;

$$
\begin{align}
\frac{\partial L}{\partial z_i}
&= \sum_j(\frac{\partial L_j}{\partial a_j}\frac{\partial a_j}{\partial z_i})\\
&=\sum_{j \neq i} a(\frac{\partial L_j}{\partial a_j}\frac{\partial a_j}{\partial z_i}) 
+ \sum_{j=i}(\frac{\partial L_j}{\partial a_j}\frac{\partial a_j}{\partial z_i})\\
&=\sum_{j\neq i}a_iy_j + (-y_i(1-a_i))\\
&=a_i\sum_jy_j-y_i
\end{align}
$$

对于 one-hot 向量的标签，我们可以将其化简为

$$
\frac{\partial L}{\partial z_i}=a_i-y_i
$$
