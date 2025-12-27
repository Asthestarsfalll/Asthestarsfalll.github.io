---
title: 随机变量及其概率分布
tags: [probability theory and mathematical statistics]
hide_table_of_contents: false
---

## 随机变量及其分布函数

在样本空间上 $\Omega$ 上的实值函数 $X=X(\omega),\omega\in\Omega$，则称 $X(\omega)$ 为随机变量，简记 $X$.

**离散性随机变量**：其取值可以是有限多个或可数无穷多个。

**概率分布**：与 $X$ 一一对应的概率值，满足非负，并且所有关于 $X$ 的概率值求和为 1.

**分布函数**：设 $X$ 是一个随机变量，对于任意实数 $x$，有

$$
F(x) = P\{X\leq x\}
$$

分布函数性质：

1. 至于在 0 到 1 之间，单调非减，$F(-\infty)=0,F(\infty)=1$；
2. 右连续，即 $F(x+0)=F(x)$；
3. 对任意的 $x$，$P\{X=x\}=F(x)-F(x-0)$
