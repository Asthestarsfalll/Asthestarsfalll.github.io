---
title: 数理统计的基本概念
tags: [probability theory and mathematical statistics]
hide_table_of_contents: false
---

## 总体、样本

**总体**：数理统计中所研究对象的某项数量指标 $X$ 的全体成为总体。

对于随机变量 $X$，则其概率分布为 **总体概率**，总体中的每个元素成为 **个体**。

对于随机变量序列 $X_{1},X_{2},\cdots X_{n}$ 相互独立，并且都与总体 $X$ 同分布，则称其为来自总体的 **简单随机样本**，简称为 **样本**，其中 $n$ 为 **样本容量** ，样本中的具体观测值 $x_{1},x_{2},\cdots,x_{n}$ 称为 **样本值**，或者称总体 $X$ 的 $n$ 个 **独立观测值**。

:::tip

由于样本间相互独立，$\sum X_{i}$ 的分布函数和概率函数都是总体的乘积，如

$$
F_{n}(x_{1},x_{2},\cdots,x_{n})=\prod F(x_{i})
$$

$$
f_{n}(x_{1},x_{2},\cdots,x_{n})=\prod f(x_{i})
$$

:::

## 统计量

:::tip 定义

样本中不含未知参数的函数 $T=T(X_{1},X_{2},\cdots,X_{n})$，因此统计量也是一个随机变量，并且与随机变量序列 $X_{i}$ 具有相对应的观测值

$$
\{x_{1},x_{2},\cdots,x_{n}\}\sim T(x_{1},x_{2},\cdots,x_{n}).
$$

:::

### 样本数字特征

**样本均值**：$\overline{X}=\frac{1}{n}\sum X_{i}$，有

$$
E(\overline X)=\frac{1}{n}E(X_{i})=\frac{1}{n} nE(x)=E(x)
$$

:::info

这是一个无偏估计，因此 $E(\overline X)=\mu$，证明见后面的样本方差的证明。

:::

**样本方差**

:::tip 无偏

指的是样本值应该围绕真实值上下波动，而不能只在一边，更标准一点是其与真实值在数学期望上相同。

:::

$S^2=\frac{1}{n-1}\sum(X_{i}-\overline X)^2$，除以 $n-1$ 被称为 **无偏估计**，样本方差的作用就是为了近似总体方差 $\sigma^2$，根据其定义为

$$
\sigma^2=E[(X-\mu)^2]
$$

由于实际统计过程中，不知道数据的具体分布，需要使用采样来替换

$$
S^2=\frac{1}{n}\sum(X_{i}-\mu)^2 
$$

而总体均值实际上常常也不知道，因此使用样本均值代替

$$
S^2=\frac{1}{n-1}\sum(X_{i}-\overline X)^2
$$

上述过程引出了两个问题：

1. 为什么可以使用 $S^2$ 代替 $\sigma^2$？
2. 为什么使用 $\overline X$ 代替 $\mu$ 之后分母需要减去 1.

[参考](样本方差公式是如何推导出来的？ - 王赟 Maigo 的回答 - 知乎 https://www.zhihu.com/question/52367271/answer/130217781)

第一个问题是高斯分布（参考中心极限定理）总体方差的最大似然估计。  可以设似然函数为

$$
L(u,\sigma^2)=\Pi_{i=1}^n \frac{1}{\sqrt{ 2\pi  }\sigma} e^{- \frac{(x_{i}-\mu)^2}{2\sigma^2}}
$$

取对数

$$
\ln L(\mu,\sigma^2)=-\frac{n}{2}\ln 2\pi-n\ln\sigma-\frac{1}{2\sigma^2}\sum_{i=1}^n(X_{i}-\mu)^2
$$

令 $\frac{\partial \ln L}{\partial\mu}=\frac{1}{\sigma^2}\sum_{i=1}^n(X_{i}-\mu)=0$，得 $\mu$ 的最大似然估计是

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^nX_{i}=X
$$

将其带回原式，令 $\frac{\partial \ln L}{\partial\sigma}=-\frac{n}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^n(X_{i}-\overline X)^2=0$，解得其最大似然估计是

$$
\hat{\sigma^2}=\frac{1}{n}\sum_{i=1}^n(X_{i}-\overline X)^2
$$

计算其期望：

$$
\begin{aligned}
E(\hat{\sigma^2})&=E\left[\frac{1}{n}\sum_{i=1}^n(X_{i}-\overline X)^2\right]\\
&=E\left[\frac{1}{n}\sum_{i=1}^n[(X_{i}-\mu)-(\overline X-\mu)]^2\right]\\
&=E\left[\frac{1}{n}\sum_{i=1}^n(X_{i}-\mu)^2 -\frac{2}{n}\sum_{i=1}^n(X_{i}-\mu)(\overline X-\mu)+\frac{1}{n}\sum_{i=1}^n(\overline X-\mu)^2\right]\\
&=E\left[\frac{1}{n}\sum_{i=1}^n(X_{i}-\mu)^2 -2(\overline X-\mu)^2+(\overline X-\mu)^2\right]\\
&=E\left[\frac{1}{n}\sum_{i=1}^n(X_{i}-\mu)^2 -(\overline X-\mu)^2\right]\\
&=\sigma^2-\frac{1}{n}\sigma^2\\
&=\frac{n-1}{n}\sigma^2

\end{aligned}
$$

 因此这是一个有偏的，需要进行修正；另外一个证明方法是求出其偏差的期望，计算是相似的，见下。

首先要证明样本值与其均值之差的平方和最小，设 $X$ 的二阶矩存在，要证

$$
E[(X-\mu)^2]\leq E[(X-c)^2]
$$

即为

$$
\begin{aligned}

&E[(X-c)^2]\\

=&E[(X-\mu+\mu-c)^2]\\

=&E[(X-\mu)^2]+E[(X-c)^2]+2E[X-\mu](\mu-c)\\

\end{aligned}
$$

由于 $E[X-\mu]=E(X)-\mu=0$，则

$$
E[(X-c^2)]=E[(X-\mu)^2]+E[(X-c)^2]\geq E[(X-\mu)^2
$$

因此，可以说明，样本方差最小

$$
\sum(X_{i}-\overline X)^2\leq \sum(X_{i}-\mu)^2
$$

说明样本均值离总体均值越远，则右侧值越大，在进行一部递推

$$
E\left[ \frac{1}{n}\sum(X_{i}-\overline X)^2 \right]\leq E\left[ \frac{1}{n}\sum(X_{i}-\mu)^2 \right]=\sigma^2
$$

即这时的样本均值总是小于总体均值的，因此 **不是无偏估计**，可以通过计算来看看到底小了多少

$$
\begin{aligned}

E{S^2}&=E\left[ \frac{1}{n}\sum (X_{i}-\overline X)^2 \right]\\

&=E\left[ \frac{1}{n} \sum \left((X_{i}-\mu)-(\overline X-\mu)\right)^2 \right]\\

&=E\left[ \frac{1}{n}\sum(X_{i}-\mu)^2-\frac{2}{n}\sum(\overline{X}-\mu)\sum(X_{i}-\mu)+(\overline X-\mu)^2  \right]

\end{aligned}
$$

其中

$$
\overline X-\mu=\frac{1}{n}\sum X_{i}-\mu=\frac{1}{n}\sum X_{i}-\frac{1}{n}\sum\mu=\frac{1}{n}\sum(X_{i}-\mu)
$$

带入得

$$
\begin{aligned}

E[S^2]&=E\left[ \frac{1}{n}\sum(X_{i}-\mu)^2-(\overline X-\mu)^2  \right]\\

&=E\left[ \frac{1}{n}\sum(X_{i}-\mu)^2 \right]-E\left[(\overline X-\mu)^2\right]\\

&=\sigma^2-E\left[(\overline X-\mu)^2\right]\\

\end{aligned}
$$

其中

$$
\begin{aligned}

E\left[(\overline X-\mu)^2\right]&=D(\overline X-\mu)+\left[E(\overline X-\mu)\right]^2\\

&=D(\overline X)\\

&=D\left(\frac{1}{n}\sum X_{i}\right)\\

&=\frac{n\sigma^2}{n^2}\\

&=\frac{\sigma^2}{n}

\end{aligned}
$$

:::tip

这里实际上证明了 $D(\overline X)=\frac{1}{n}D(X)=\frac{\sigma^2}{n}$.

:::

因此

$$
\begin{aligned}

E[S^2]&=E\left[\frac{1}{n}\sum(X_{i}-\overline X)^2\right]\\

&=\sigma^2-\frac{1}{n}\sigma^2\\&=\frac{n-1}{n}\sigma^2

\end{aligned}
$$

即

$$
\frac{n}{n-1}E\left[\frac{1}{n}\sum(X_{i}-\overline X)^2\right]=E\left[ \frac{1}{n-1}\sum(X_{i}-\overline X)^2 \right]=\sigma^2=D(X)
$$

因此得到无偏估计

$$
S^2=\frac{1}{n-1}\sum(X_{i}-\overline X)^2
$$

:::info

上式实际上证明了 $E(S^2)=D(X)=\sigma^2$.

:::

:::tip 从自由度的角度考虑

通常所样本方差需要除以自由度 $n-1$，一个比较直观的解释是 **均值已知**，给定 $n-1$ 个数则最后一个数字已经确定下来了，因此自由度只有 $n-1$.

:::

**样本 k 阶原点矩、中心矩**：形式上和总体的原点矩、中心矩一致，注意样本二阶中心矩并不等于样本方差，而总体二阶中心矩等于总体方差。

## 正态总体的抽样分布

### 一个正态总体

于随机变量 $X\sim N(\mu,\sigma^2)$ 及其简单随机样本，有

$$
\overline X\sim N\left( \mu, \frac{\sigma^2}{n} \right)
$$

因为 $D(\overline X)=  \frac{\sigma^2}{n}$

$\overline X$ 和 $S^2$ 相互独立，且有

$$
\chi^2= \frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$

:::info

这里说的是样本方差和样本均值独立，对于正态总体独立，但是其他就不一定了。

:::

$$
T= \frac{\overline X-\mu}{S / \sqrt{ n }}\sim t(n-1)

\chi^2=\frac{1}{\sigma^2}\sum(X_{i}-\mu)^2\sim\chi^2(n)
$$

### 两个正态总体

$$
\overline X-\overline Y\sim N\left( \mu_{1}-\mu_{2}, \frac{\sigma^2}{n_{1}}+\frac{\sigma^2}{n_{2}} \right)
$$

若两个正态总体的方差相同，则

$$
T= \frac{\overline X-\overline Y-(\mu_{1}-\mu_{2})}{S_{w}\sqrt{ \frac{1}{n_{1}}+\frac{1}{n_{2}} }}
$$

其中 $S_{w}= \frac{(n_{1}-1)S_{1}^2+(n_{2}-1)S_{2}^2}{n_{1}+n_{2}-2}$

$$
F= \frac{S_{1}^2 / \sigma_{1}^2}{S_{2}^2 /\sigma_{2}^2}\sim F(n_{1}-1,n_{2}-1)
$$
