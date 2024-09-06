---
title: Gumbel Softmax
authors:
  - Asthestarsfalll
tags:
  - PaperRead
  - BaseLearn
description: 什么是gumbel softmax?
hide_table_of_contents: false
---

## argmax

在学习 gumbel softmax 之前，我们首先需要了解它的远方亲戚 argmax。

假设我们有一个概率分布向量如下：

$$
[0.3,0.1, 0.1, 0.5]
$$

对于 argmax 来说，显然每次的结果都会是 3，因为该位置的概率值最大。但是从概率上来说，只有 50% 的概率会选到第 3 个位置，而使用 argmax 则会有 100% 的概率选中第 3 个位置，这显然是不合理的。

基于 argmax 的采样如下：

```python
pos = argmax(logits)
sample = logits[pos] #就是max
```

## softmax

argmax 能直接得到最大概率的位置，我们通常需要在分类、分割任务中这样做。但是 argmax 是不可微的，这样会阻碍反向传播，于是提出了 softmax。

softmax 是 argmax 的光滑近似，其可以拉大输入向量之间的差距，并且可微，能够正常的计算梯度反向传播。

基于 softmax 的采样如下：

```python
pro = sotfmax(logits)
sample = np.random.choice(len(logits)，1, p=pro)
```

虽然 softmax 可微，但是基于 softmax 的采样仍然不能反向传播。

## 	gumbel max

让我们首先在 argmax 中引入随机性——gumbel 分布，其是一种极值分布，表示某个随机变量在不同时间段中极值的概率分布。比如一个人每天喝 8 次水，显然这 8 次中的极值也是一个随机变量，该随机变量随着时间的分布即为 gumbel 分布。

其累积分布函数为

$$
F(x)=e^{-e^{-x}}
$$

我们可以通过求解其反函数来利用概率生成随机数：

$$
G=-log(-log(x))
$$

我们通过生成与输入向量维度相同的均匀分布向量，从 gunbel 分布中进行采样，以此获得随机性：

$$
G_i = -log(-log(\varepsilon_i)),\varepsilon_i \in U(0,1)
$$

于是可以得到最终的公式：

$$
x = argmax(log(p_i)+G_i)
$$

这其实是一种重参数化的过程，[具体见此](https://kexue.fm/archives/6705)

并且我们可以证明，gumbel max 输出 i 的概率刚好对应 $p_i$。

首先我们证明输出 1 的概率是 $p_1$，输出 1 意味着 $\log p_1 - \log(-\log \varepsilon_1)$ 最大，也就是说以下不等式成立：

$$
\begin{equation}\begin{aligned} 
&\log p_1 - \log(-\log \varepsilon_1) > \log p_2 - \log(-\log \varepsilon_2) \\ 
&\log p_1 - \log(-\log \varepsilon_1) > \log p_3 - \log(-\log \varepsilon_3) \\ 
&\qquad \vdots\\ 
&\log p_1 - \log(-\log \varepsilon_1) > \log p_k - \log(-\log \varepsilon_k) 
\end{aligned} 
\end{equation}
$$

注意这里每个不等式是独立的，$p_1$ 与 $p_2$ 的关系并不影响 $p_1$ 和 $p_3$ 的关系。

首先分析第一个不等式，化简可得：

$$
\begin{equation}\varepsilon_2 < \varepsilon_1^{p_2 / p_1}\leq 1 \end{equation}
$$

由于 $\varepsilon$ 是从均匀分布中采样的，因此我们知道 $\varepsilon_2 < \varepsilon_1^{p_2 / p_1}$ 的概率就是 $\varepsilon_1^{p_2 / p_1}$，对于某一个固定的 $\varepsilon_1$，当所有不等式同时成立时，概率为：

$$
\begin{equation}\varepsilon_1^{p_2 / p_1}\varepsilon_1^{p_3 / p_1}\dots \varepsilon_1^{p_k / p_1}=\varepsilon_1^{(p_2 + p_3 + \dots + p_k) / p_1}=\varepsilon_1^{(1/p_1)-1}\end{equation}
$$

对于所有的 $\varepsilon_1$ ，我们可以得出其概率：

$$
\begin{equation}\int_0^1 \varepsilon_1^{(1/p_1)-1}d\varepsilon_1 = p_1\end{equation}
$$

## gumbel softmax

由于 argmax 不可导，我们可以使用其近似函数——softmax

$$
x = softmax\big ((log(p_i)+G_i)/\tau\big )
$$

$\tau$ 表示温度，是一种退火技巧，其值越小，输出结果越接近 one hot 的形式，但同时梯度消失的情况就越严重。
