---
title: 秩
tags: [Linear Algebra]
description: 矩阵的秩
hide_table_of_contents: false
---

## 矩阵乘的秩

$$
r(AB)\leq min\{r(A),r(B)\}\leq\left\{\begin{aligned}
r(A,B)\\or\\ r\begin{pmatrix}
A\\B
\end{pmatrix}
\end{aligned}\right.\leq r(A)+r(B)
$$

矩阵乘的秩会越乘越小，但是拼起来会变大（可以取等号，不是严格的）。

若 $AB=O$ ，则

$$
r(A)+r(B)\leq n
$$

这里的 $n$ 是中间的维度。

## 分块矩阵的秩

:::tip

从初等变换的角度考虑最简单。

:::

对于最简单的对角分块矩阵，其秩为

$$
\begin{aligned}
r \begin{pmatrix}A&O\\O&B\end{pmatrix}=r(A)+r(B)\\r \begin{pmatrix}O&A\\B&O\end{pmatrix}=r(A)+r(B)
\end{aligned}
$$

若只有一个零块，则相等关系变为大于等于关系

$$
\begin{aligned}
r \begin{pmatrix}A&C\\O&B\end{pmatrix}\geq r(A)+r(B)\\r \begin{pmatrix}A&O\\C&B\end{pmatrix}\geq r(A)+r(B)
\end{aligned}
$$

如果 $A,B$ 可以消去 $C$，则转换为第一种情况；消不掉则秩增加；若 $A$ 列满秩或 $B$ 行满秩，则 $\begin{pmatrix}A&O\\C&B\end{pmatrix}$ 可以取得等号，对 $\begin{pmatrix}A&C\\O&B\end{pmatrix}$，请思考什么情况能取得等号？

<details>
  <summary>🤔</summary>
A 列满秩或 B 行满秩，为方阵时就是其中一个矩阵可逆。
</details>

## 伴随矩阵的秩

伴随矩阵的秩序与原矩阵的秩有如下关系

$$
r(A^*)=\begin{cases}
n,\ r(A)=n\\1,\ r(A)=n-1\\0,\ r(A)<n-1
\end{cases}
$$

当 $r(A)=n-1$ 时，说明至少有一个 $n-1$ 阶子式不为 $0$，又因为 $AA^*=0$，根据矩阵相乘的性质 $r(A)+r(A^*)\leq 1$，则 $r(A^*)=1$；或者从方程组解的角度来说明，$A^*$ 的所有列向量都是齐次线性方程组的解，而 $n-r(A)=1$，则其基础解系只有一个向量，因此 $r(A^*)=1$.

## 秩与线性表示

若 $A$ 可以由 $B$ 线性表示，则被表示的秩可以更小

$$
\begin{aligned}
r(A)\leq r(B)\\
r(A)=r(A,B)
\end{aligned}
$$

若 $AB=C$，从列向量的角度看

$$
r(A)=r(A,AB)
$$

从行向量的角度看

$$
r(B)=r\begin{pmatrix}
B\\AB
\end{pmatrix}
$$

具体看 [矩阵乘法](./matrix.md#矩阵乘法)

## n 秩相等

$2$ 秩表示矩阵等价，$3$ 秩相等表示向量组等价和方程组同解 $r(A)=r(B)=r\begin{pmatrix}A\\B\end{pmatrix}$ ，$4$ 秩相等常用于转置矩阵 $r(A)=r(A^T)=r(AA^T)=r(A^TA)$，可以证明 $Ax=O$ 和 $A^Tx=O$ 同解，进一步地，可以推出 $6$ 秩相等 $r(A)=r(A^T)=r(A^TA)=r(A A^T)=r(A^TA A^TA)=r(A A^TA A^T)$.

## 秩与方程组的解

若 $Ax=0$ 的解均是 $Bx=0$ 的解，则 $r(A)\geq r(B)$；显然前者的解被包含在后者的解当中，即 $n-r(A)\leq n-r(B)\to r(A)>r(B)$.

同样利用方程组可以解释矩阵乘法中 $AB=O\implies r(A)+r(B)\leq n$，证明方式和上述伴随矩阵一致，显然 $B$ 中的所有列向量是 $Ax=O$ 的解，即 $r(B)\leq n-r(A)\implies r(A)+r(B)\leq n$.

## 行列满秩矩阵的性质

若 $A_{m\times n},r(A)=n$ 则有以下角度的考虑：

1. 秩：若 $r(AB)=r(B)$，则至少有一个 $n$ 阶子式不为零 （$n+1$ 阶都为零）；
2. 向量：列向量线性无关，$n$ 个行向量线性无关；
3. 方程：$Ax=0$ 仅有零解，$Ax=b$ 可能无解，$A^Tx=b$ 一定有解；
4. 空间：由于秩不超过行列的最小值，因此有 $m\geq n$，即维数大于等于向量个数；
5. 变换：存在 $n$ 阶可逆矩阵 $P$，使得 $PA=\begin{pmatrix}E_{n}\\O\end{pmatrix}$；
6. 正定：$A^TA$ 为正定矩阵。

若 $A_{m\times n},r(A)=m$ 则有以下角度的考虑：

1. 秩：若 $r(BA)=r(B)$，则至少有一个 $m$ 阶子式不为零 （$m+1$ 阶都为零）；
2. 向量：行向量线性无关，$m$ 个列向量线性无关；
3. 方程：$Ax=b$ 一定有解，$A^Tx=b$ 仅有零解，$A^Tx=b$ 可能无解；
4. 空间：由于秩不超过行列的最小值，因此有 $n\geq m$，即维数小于等于向量个数；
5. 变换：存在 $n$ 阶可逆矩阵 $P$，使得 $AP=\begin{pmatrix}E_{n},O\end{pmatrix}$；
6. 正定：$AA^T$ 为正定矩阵。
