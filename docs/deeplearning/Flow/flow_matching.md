---
title: Flow_matching
authors:
  - Asthestarsfalll
tags:
  - PaperRead
description: None
hide_table_of_contents: false
---

https://zhuanlan.zhihu.com/p/685921518

## Flow-based Model

**Flow-based Model**是一种基于**Normalizing Flows(NFs)** 的生成模型，它通过一系列概率密度函数的变量变换，将复杂的概率分布转换为简单的概率分布，并通过逆变换生成新的数据样本。而**Continuous Normalizing Flows(CNFs)** 是**Normalizing Flows**的扩展，它使用常微分方程 (**ODE**) 来表示连续的变换过程，用于建模概率分布。

**Flow Matching(FM)** 是一种训练**Continuous Normalizing Flows**的方法，它通过学习与概率路径相关的向量场 **(Vector Field)** 来训练模型，并使用 **ODE** 求解器来生成新样本。

扩散模型是**Flow Matching**的一个应用特例，使用 FM 可以提高其训练的稳定性。此外,使用最优传输 (**Optimal Transport**) 技术构建概率路径可以进一步加快训练速度，并提高模型的泛化能力。

## Overview

给定一个随机变量 $z$ 及其概率密度函数 $z∼\pi(z)$ ，通过一个一对一的映射函数 $f$ 构造一个新的随机变量 $x=f(z)$ 。如果存在逆函数 $f^{−1}$ ，那么新变量 $x$ 的概率密度函数 $p(x)$ 计算如下：

**（1）当**z**为随机变量：**

$$
p (x)=\pi(z)|\frac{dz}{dx}|=\pi(f^{−1}(x))|\frac{df^{−1}}{dx}|=\pi(f^{−1}(x))|(f^{−1})′(x)|
$$

**（2）当**z **为随机向量：**

$$
p(x)=\pi(z)|\mathrm{\det}⁡ \frac{dz}{dx}|=\pi(f^{−1}(x))|\mathrm{\det}⁡⁡ \frac{df^{−1}}{dx}|
$$

其中，$det$ 是行列式， $\frac{df}{dx}$ 是雅可比矩阵。

**特例：如果 $x∼N(\mu,\sigma^2)$，当 a,b 为实数时，则有 $z =f(x)=ax+b∼N(aμ+b,(aσ)^2)$**

## Normalizing Flows

一种可逆的概率密度变换方法，它的核心思想是通过一系列可逆的变换函数来逐步将一个简单分布（通常是高斯分布）转换成一个复杂的目标分布。**这个过程可以被看作是一连串的变量替换的迭代过程，每次替换都遵循概率密度函数的变量变换原则**。

可以通过变换关系和对数似然将目标分布展开到初始分布，并且雅可比矩阵易于计算，训练时优化目标即为负对数似然。

## Continuous Normalizing Flows

**Continuous Normalizing Flows (CNFs)** 是 **Normalizing Flows** 的一种扩展，它可以更好地建模复杂的概率分布。在传统的 **Normalizing Flows** 中，变换通常是通过一系列 **可逆的离散函数** 来定义的，而在 **CNFs** 中，这种变换是 **连续的**，这使得模型能够更加平滑地适应数据的分布，提高了模型的表达能力。**CNFs** 过程通过常微分方程（**ODE**）来表示：

$$
\frac{dz_t}{dt}=v(z_t,t)
$$

其中， $t \in [0,1]$ ，$z_t$ 是**Flow Map** 或者 **Transport Map**，可简单理解为时间 $t$ 下的数据点， $v(z_t,t)$ 是一个**向量场**，它定义了每一个数据点在状态空间中随时间的变化方向和大小，**通常为神经网络预测**。

如果知道了这个向量场 v(zt,t) ，那么通过求解这个 **ODE**就可以找到从初始概率分布到目标概率分布的连续路径，从而将简单分布转换成复杂分布。这个**ODE**可以采用**欧拉方法**来求解。

给定一个初始概率分布（通常是标准高斯分布），向量场 $v(z_t,t)$ 可以描述这个分布随时间的演变，最终达到目标分布。这是 **CNFs** 建模复杂概率分布的基础，即可以通过学习向量场来学习数据的变换过程。

## Flow Matching

训练**Continuous Normalizing Flows**的直观方法是，在给定初始条件 x0 ​的情况下，通过**ODE**求解得到的 x1 ​的分布，然后通过一种最小化差异度量（如 KL 散度）来约束 x1 与真实数据的分布保持一致。然而，由于中间轨迹多而且未知，推断 x1（通过采样或者计算似然概率） 需要反复模拟**ODE**，计算量非常巨大。为此，论文提出了新的方法**Flow Matching（FM）**。

**Flow Matching**是一种适用于训练**Continuous Normalizing Flows**的技术，它是**Simulation-Free**的，即无需通过 ODE 推理目标数据分布。**它的核心思想在于，通过确保模型预测的向量场与描述数据点实际运动的向量场之间的动态特性保持一致性，从而确保通过 CNFs 变换得到的最终概率分布与期望的目标分布相匹配。**
