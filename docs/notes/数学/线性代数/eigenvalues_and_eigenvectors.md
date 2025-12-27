---
title: 特征值和特征向量
tags: [linear algebra]
description: 特征值、特征向量、相似矩阵和实对称矩阵。
hide_table_of_contents: false
---

# 特征值和特征向量

## 定义

:::caution

只有方阵有特征值和特征向量。

非方阵表示的线性变换为投影，无法分解为线性无关的基向量。

特征值分解找到了某变换下只有伸缩效果的基，而奇异值分解可以找到在某变换下旋转、缩放和投影的基。

:::

对于 $n$ 阶矩阵 $A$，存在一个数 $\lambda$ 和**非零**的 $n$ 维列向量 $\alpha$，使得

$$
A\alpha=\lambda\alpha  \tag1
$$

成立，其中 $\lambda$ 为一个常数，表示特征值，非零向量 $\alpha$ 为其对应的特征向量。

:::info

从几何意义上来说，矩阵的特征向量指向在某个线性变换（旋转、缩放）中（上文中即 $A$）只缩放而不旋转的方向，特征值即是其缩放因子。

因此从几何意义可以得知：旋转矩阵无特征值和特征矩阵。

:::

由式 $(1)$ 移项 $(\lambda E-A)\alpha=0$，而 $\alpha$ 为非零向量，则其为齐次方程组 $(\lambda E-A)x=0$ 的非零解，而要使非零解存在，可推出

$$
 |\lambda E-A|=0 \tag2
$$

式 $(2)$ 可以得出 $n$ 个特征值 $\lambda_{i}$，再有 $(\lambda_{i}E-A)x=0$ 解出**基础解系**，即为其对应的特征向量。

:::note

通过上述方程组可以得到，$\lambda_{i}E-A$ 的秩为 $0$，即都不可逆，因此在实际求解过程之中，可以直接令某一行为 $0$.

:::

## 相关定理或结论

若 $\lambda_{i}$ 是 $A$ 的互不相同的特征值，$\alpha_{i}$ 是其对应的特征向量，则 $\alpha_{i}$ 线性无关。

$A$ 为 $n$ 阶矩阵，$\lambda_{i}$ 为其特征值，则有

$$
\begin{align*}
\sum \lambda_{i}&=\sum a_{ii} \\
|A|&= \prod\lambda_{i}
\end{align*}
$$

[参考](https://www.cnblogs.com/qizhou/p/12583084.html)

:::info

若对于矩阵 $A$ 的某个特征值 $\lambda$ 的所有特征向量 $\alpha_{i}$，则 $\sum k_{i}\alpha_{i}\neq 0$ 仍为其特征向量（只有一个特征向量同样适用）。

证明：

$$
\begin{align*}
A\left( \sum k_{i}\alpha_{i} \right) &= \sum Ak_{i}\alpha_{i} \\
&=\sum k_{i} A\alpha_{i} \\
&=\sum k_{i}(\lambda \alpha_{i}) \\
&=\lambda\sum k_{i}\alpha_{i}
\end{align*}
$$

实际上根据特征向量和特征值的定义可以发现，特征向量就是求给定特征多项式齐次方程的非零解，而齐次方程的解的线性组合仍然是其解。

:::

若 $\lambda$ 是 $A$ 的特征值，则 $\lambda+k$ 是 $A+kE$ 的特征值。使用定义证明

$$
\begin{align*}
(A+kE)\alpha &= A\alpha+kE\alpha \\
&=\lambda\alpha+k\alpha \\
&=(\lambda+k)\alpha
\end{align*}
$$

另有 $\lambda^m$ 是 $A^m$ 的特征值。同样使用定义证明

$$
\begin{align*}
A^m\alpha&=A^{m-1}A\alpha \\
&=\lambda A^{m-1}\alpha \\
&=\lambda^m\alpha
\end{align*}
$$

:::note

由上述式子，同样可以证明 $AB$ 的一个特征值是二者特征值之积。

:::

**特征值和原矩阵的关系**

$f(A)=O\implies f(\lambda)=O$ ，例如有 $A^2-A=O$，则有 $\lambda^2-\lambda=0$ ，特征值一定在 $f(\lambda)=0$ 的解之中，但反过来其解就不一定是特征值。

若有形如 $A^2-(a+b)A+abE=O\implies(A-aE)(A-bE)=O$ 可以因式分解的形式，且 $a\neq b$ 则 $A$ 一定可以相似对角化，证明如下：

1. 由于二者相乘为 $0$，则 $r(A-aE)+r(A-bE)\leq n$；
2. $n\geq n-r(A-aE)+n-r(A-bE)\implies r(A-aE)+r(A-bE)\geq n$.

**每行和为定值的矩阵**

若 $A$ 每行之和为 $k$，

$$
A\begin{pmatrix}1\\1\\1\end{pmatrix}=\begin{pmatrix}k\\k\\k\end{pmatrix}\implies A\xi=k\xi
$$

更进一步可以推出

$$
AB=kB
$$

其中 $B$ 是由对应特征值 $k$ 的特征向量组成。

## 特征值与特征向量的推广

| 矩阵     |  $A$    |  $aA+bE$    |  $A^n$    |  $f(A)$    |   $A^{-1}$   |  $A^*$    |  $A^T$    | $P^{-1}AP$     |
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
|  特征值    | $\lambda$     |  $a\lambda+b$    |  $\lambda^n$    |   $f(\lambda)$    |  $\frac{1}{\lambda}$    |  $\frac{\mid A\mid}{\lambda}$    |   $\lambda$   | $\lambda$    |
|   特征向量   |    $\xi$  |    $\xi$  |     $\xi$ |    $\xi$  | $\xi$     | $\xi$     |  其他   |  $P^{-1}\xi$    |

由于前六种的特征向量都是 $\xi$，因此可以任意组合而特征向量不变。

:::tip

由于伴随矩阵的特征值在分母上，如果有一个为 $0$ 的，则不好计算，因此可以使用 $|A|=\lambda_{1}\lambda_{2}\lambda_{3}\cdots$ 代替，刚好约掉分子的特征值。

:::

:::tip

若矩阵 $A$ 的特征值为 $\pm1$，则有 $A=A ^{-1}$，同理，若 $A$ 的特征值只有 $\sqrt{ |A|}$，则说明 $A=A^*$

:::

### 伴随矩阵的特征值和特征向量

:::caution

要求 $A$ 可逆。

:::

伴随矩阵和原矩阵的关系

$$
A^*A=A A^*=|A|E
$$

由特征向量定义

$$
A\alpha=\lambda\alpha
$$

左乘 $A^*$

$$
A^*A\alpha=\lambda A^*\alpha
$$

即

$$
\frac{|A|}{\lambda}E\alpha =\frac{|A|}{\lambda}\alpha=A^*\alpha
$$

则其特征值为 $\frac{|A|}{\lambda}$，且 $\alpha$ 仍是特征向量。

### 转置矩阵的特征值和特征向量

由定义

$$
|\lambda E-A^T| = |(\lambda E-A)^T| = |\lambda E-A|
$$

因此具有相同的特征值。而 $A\alpha=\lambda\alpha$ 无法推出 $A^T\alpha=\lambda\alpha$.

### 相似矩阵的特征值和特征向量

$$
B=P^{-1}AP\implies BP ^{-1}=P^{-1}A\implies B(P^{-1}\xi)=\lambda(P ^{-1}\xi)
$$

# 相似矩阵

:::caution

要求方阵

:::

## 定义

设 $n$ 阶矩阵 $A,B$ ，若存在可逆矩阵 $P$ ，使得

$$
P^{-1}AP=B
$$

则称 $B$ 是 $A$ 的相似矩阵，记作 $A \sim B$.

:::info

其拥有如下性质

1. 反身性，$A\sim A$；
2. 对称性，$A\sim B \implies B\sim A$；
3. 传递性，$A\sim B,B\sim C\implies A\sim C$.
:::
:::caution
两矩阵相似的必要条件
1. 特征多项式相同即 $|\lambda E-A|=|\lambda E-B|$；
2. 具有相同的特征值；
3. 秩相同；
4. $|A|=|B|=\prod\lambda_{i}$；
5. $\sum a_{ii}=\sum b_{ii}=\sum \lambda_{i}$.

这里的特征值之和就是 **迹**，记作 $tr(A)$.

:::

## 相关结论

$A\sim B\implies A^n\sim B^n$，证明如下

$$
\begin{align*}
B^n &= (P^{-1}AP)^n \\
&=(P ^{-1}AP)(P ^{-1} AP)\cdots(P ^{-1}AP) \\
&=P ^{-1}A^nP
\end{align*}
$$

$A\sim B\implies A+kE\sim B+kE$，证明如下

$$
\begin{align*}
B+kE&=P^{-1}(A+kE)P \\
&=P ^{-1}AP+kP ^{-1}EP \\
&=P ^{-1} AP+kE
\end{align*}
$$

$A \sim B\implies A ^{-1}\sim B ^{-1}$，证明如下

由 $A\sim B\implies |A|=|B|$ ，$B$ 可逆

$$
\begin{align*}
B ^{-1}&=(P ^{-1}A P)^{-1} \\
&=P ^{-1} A ^{-1} (P ^{-1})^{-1} \\
&=P ^{-1} A ^{-1}P
\end{align*}
$$

$A \sim B\implies A ^{T}\sim B ^{T}$，$B^T=(P^{-1}AP)^T=P^TA^T(P^T)^{-1}=Q^{-1}A^TQ$

$A \sim B\implies AB \sim BA$，$A^{-1}ABA=BA$.

## 相似对角化

:::caution

首先要证明可相似对角化。

:::

若 $A\sim \Lambda$，其中 $\Lambda$ 是对角矩阵，则称 $A$ 可相似对角化，$\Lambda$ 是 $A$ 的**相似标准形**。

:::info

相似对角化的充分必要条件是 $A$ 拥有 $n$ 个线性无关的特征向量，也即每个特征值线性无关的特征向量个数，等于该特征值的重数，也即 $r(\lambda_{i}E-A)=n-n_{i}$，$n_{i}$ 为重数。

:::

通过相似的性质，可以推出，$A$ 可相似对角化为

$$
A\sim \begin{bmatrix} \lambda_{1} & & & \\ & \lambda_{2} & & \\ & & \ddots&\\ & & &\lambda_{n} \end{bmatrix}
$$

:::info

相似对角化中的可逆矩阵 $P$ 可以由特征向量构成，$P=(\alpha_{1},\cdots,\alpha_{n})$

$$
\begin{align*}
A(\alpha_{1},\cdots,\alpha_{n})&=(\lambda_{1}\alpha_{1},\cdots,\lambda_{2}\alpha_{n}) \\
&=(\alpha_{1},\cdots,\alpha_{n}) \begin{bmatrix} \lambda_{1} & & & \\ & \lambda_{2} & & \\ & & \ddots&\\ & & &\lambda_{n} \end{bmatrix} \\
&=P\Lambda
\end{align*}
$$

即 $AP=P\Lambda\implies P ^{-1}AP=\Lambda$.

由于 $P$ 要求可逆，这也能解释为什么充要条件是特征向量线性无关。

:::

## 实对称矩阵

实对称矩阵 $A$ 的元素都为实数，且满足 $A=A^T$ .

:::info

实对称矩阵必定可以相似对角化，且属于不同特征值所对应的特征向量相互正交（属于相同特征值的特征向量则可以不正交）；此外还有迹不为 $0$ 的秩 $1$ 矩阵和主对角线元素全不相同（特征值全不相同）的上/下三角矩阵一定可以相似对角化。

存在正交矩阵 $Q^TQ=E$，使得 $Q ^{-1}AQ=Q^TAQ=\Lambda$

:::

由上，实对称矩阵可以使用正交矩阵来进行相似对角化，步骤如下

1. 求出特征值及其相应特征矩阵；
2. 若特征值无重根，则所有特征向量已正交；若有重根，由于属于相同特征值的特征向量不一定正交，则需要先进行正交化；
3. 特征向量单位化，即可得到正交阵 $Q$，有 $Q^TAQ=\Lambda$.

## 通过特征值和特征向量反求矩阵

最基本的方法是利用相似 $P ^{-1}AP=\Lambda\implies A=P\Lambda P ^{-1}$，但是计算量很大，易错，这里的 $P\Lambda$ 可以进行一步化简，实际上就是特征值乘以对应特征向量组成的矩阵，因此原式化简为 $(\lambda_{1}\alpha_{1},\cdots\lambda _{n}\alpha_{n})P^{-1}$；特别地，若 $A$ 是实对称矩阵，可以直接化简为 $\lambda_{1}\alpha_{1}\alpha_{1}^T,\cdots,\lambda _{n}\alpha_{n}\alpha_{n}^T$.

上述化简是基于列分块，不好记忆，可以利用特征值和特征向量来推导；对于普通矩阵有

$$
A=APP ^{-1}\implies(\lambda_{1}\alpha_{1},\cdots,\lambda_{n}\alpha_{n})P^{-1}
$$

对于实对称矩阵 $A$，则 $Q$ 是正交矩阵，有

$$
A=AQQ ^{-1}=AQQ^T=A(e_{1},e_{2},\cdots,e_{n})\begin{pmatrix}e_{1}^T\\ \vdots\\e_{n^T} \end{pmatrix}=\lambda_{1} e_{1}e_{1}^T+\cdots+\lambda_{n}e_{n}e_{n}^T
$$

:::tip

显然，当上述矩阵的拥有多重 $0$ 特征值，计算量会大大化简，因此我们可以将其转换为这种形式，若矩阵有多重特征值，则先减去或加上 $\lambda E$，先反求出 $A\pm kE$ 这个整体，因为其拥有多重 $0$ 特征值，可以简化计算，最后在求出 $A$ 即可。

除了有重复特征值的情况，也可用于简化实对称矩阵的求解过程，如知道两个特征向量需要求第三个，这时只需要让第三个特征值 $k$ 用上述方式变为 $0$ 即可。

实际上，这种方式在求特征值中也可以用到，如用于化简原矩阵。

:::
