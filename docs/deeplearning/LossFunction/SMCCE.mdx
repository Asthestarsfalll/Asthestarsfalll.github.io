---
title: 稀疏版多标签分类交叉熵损失函数
description: 最近偶然接触了苏剑林大佬所提的稀疏版的多标签分类交叉熵损失函数，觉得十分有意思，并且github上鲜有代码，于是使用了pytorch进行复现，故将相关学习过程记录在此。
authors:
  - Asthestarsfalll
tags:
  - nlp
  - loss
  - BaseLearn
hide_table_of_contents: false
---

大佬博客：

https://kexue.fm/archives/7359

https://kexue.fm/archives/8888

## 从单标签到多标签

在上篇文章中已经介绍过了处理常规多分类问题（也就是单标签分类）的基本操作——softmax 和交叉熵损失函数，那么什么是多标签分类呢？

单标签分类是从 `n` 个候选类别中选取一个 `1` 个目标类别进行分类，损失函数的优化目标则是使目标类别的得分最大，可以参考上篇文章的交叉熵损失函数；

对于多标签分类，我们从 `n` 个候选类别中选取 `k` 个目标类别（当做正例，即是与不是的问题），换种理解就是我们同时进行 `n` 个二分类任务。

直观上，我们可以直接选择使用 `sigmoid` 激活，使用二分类的交叉熵之和作为 loss，然而当 `n>>k` 时，会有很严重的类别不均衡问题，当 k 极小时，网络只需要简单将结果全部预测为负例也可以得到很小的 loss 值；但是单标签分类中，k=1 并没有这种类别不均衡的问题，因为我们使用了 `softmax`，使得交叉熵能够不偏不倚的对每个预测获得合适的损失。

因此，一种直觉上的思路是多标签分类的损失函数可以有 softmax 进行外推，换言之，当 k=1 时，该损失函数会退化成 softmax。

## 组合 softmax

苏剑林大佬首先考虑了 `k` 固定的情形，显然推理时我们只需要输出得分的 top-k 即可，那么训练时的 loss 怎么办呢？

类比单标签的 `n` 选 `1`，我们可以将多标签表示为 $C_n^k$ 选 1，这样便得到其 loss 应该为：

$$
-\log \frac{e^{s_{t_1}+s_{t_2}+\dots+s_{t_k}}}{\sum\limits_{1\leq i_1 < i_2 < \cdots < i_k\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_k}}}=\log Z_k - (s_{t_1}+s_{t_2}+\dots+s_{t_k}) \tag 1
$$

上式最难计算的地方便是分母，苏剑林大佬提出利用牛顿恒等式来简便计算，设 $S_k = \sum\limits_{i=1}^n e^{k s_i}$，可得：

$$
\begin{aligned} 
Z_1 =&\, S_1\\ 
2Z_2 =&\, Z_1 S_1  - S_2\\ 
3Z_3 = &\, Z_2 S_1 - Z_1 S_2 + S_3\\ 
\vdots\\ 
k Z_k = &\, Z_{k-1} S_1 - Z_{k-2} S_2 + \dots + (-1)^{k-2} Z_1 S_{k-1} + (-1)^{k-1} S_k 
\end{aligned}
$$

我们不在这里过度纠结，说一些苏剑林大佬没有说的，回到这个 loss 本身的形式，其与 softmax 的形式几乎完全一致，只不过对象从一个 $s_i$ 变为了一组 $\{s_{t_i} \}$，仔细分析一下就会发现一个问题：

对于 softmax，我们希望目标的 $s_i$ 变的足够大，而其他的 $s_i$ 足够小，而对上式来说，我们希望这一组 $s_{t_i}$ 的**和**变的足够大，但是如果其中的一个 $S_{t_i}$ 变得足够大，loss 也会变得足够小，这时候优化便停止了。

在此我尝试进行证明：

$$
log(Z_k)=log(\sum\limits_{1\leq i_1 < i_2 < \cdots < i_k\leq n}e^{s_{i_1}+s_{i_2}+\dots+s_{i_k}})
$$

注意到上式其实是 LogSumExp，而 LogSumExp 是 Max 函数的光滑近似，因此 loss 就可以变形为：

$$
L\approx MAX(e^{s_{m_1}+s_{m_2}+\dots+s_{m_k}})-(s_{i_1}+s_{i_2}+\dots+s_{i_k})\qquad \\(1\leq m_1 < m_2 < \cdots < m_k\leq n)
$$

因此，当其中的一个 $S_{t_i}$ 变得足够大，loss 就会变得足够小。

## 不确定的 k

通常在多标签分类任务中，其输出的个数往往是不固定的，因此确定了一个最大目标标签数 K，使用 0 标签作为填充，输出的标签数不会多于 K，这样 loss 就变为：

$$
\log \overline{Z}_K - (s_{t_1}+s_{t_2}+\dots+s_{t_k}+\underbrace{s_0+\dots+s_0}_{K-k\text{个}})
$$

这样的做就是为了过滤掉得分小于 $S_0$ 的标签，比如我们只需要输出 2 个标签，最大目标标签数为 10，制作标签时我们只需要添加相应的标签，剩下 8 位使用 0 标签填充，这是一个无效的标签（但是网络需要预测这个标签，即将 num_classes 变为 num_classes+1，不然推理时依然无法输出不固定个数的标签），允许重复输出，推理时照样输出 topK，但是将其中的 0 标签去除，$\overline{Z}_K$ 同样可以使用递归求解，这里不再赘述。

## 统一的 loss 形式

苏剑林大佬在验证上述 loss 的有效性的同时请教了另外一些大佬，发现了 [Circle Loss](https://arxiv.org/abs/2002.10857)（有时间就看）里统一的 loss 形式，意识到了这个统一形式蕴含了一个更简明的推广方案，并且 Circle Loss 的作者也曾说过上述方法的错误性：https://www.zhihu.com/question/382802283。

统一的 loss 形式如下：

$$
\begin{align}
L_{uni} &= log[1+\sum_{i=1}^K\sum_{j=1}^Lexp(\gamma(s_n^j-s_p^i+m))]\\
&=log[1+\sum_{i=1}^Kexp(\gamma(s_n^j+m))\sum_{j=1}^Lexp(\gamma(-s_p^i))]
\end{align}
$$

上述公式将正例和负例分开进行计算，我们将交叉熵函数也写成类似的形式：

$$
-\log \frac{e^{s_t}}{\sum\limits_{i=1}^n e^{s_i}}=-\log \frac{1}{\sum\limits_{i=1}^n e^{s_i-s_t}}=\log \sum\limits_{i=1}^n e^{s_i-s_t}=\log \left(1 + \sum\limits_{i=1,i\neq t}^n e^{s_i-s_t}\right)
$$

这个公式是不是十分眼熟，这就是前面所提到的 LogSumExp 函数，Max 的光滑近似，先来说说其推导过程吧。

### LogSumExp

参考：

https://kexue.fm/archives/3290

http://www.matrix67.com/blog/archives/2830

http://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/

当 $x\geq0,y\geq0$ 时：

$$
\max(x,y)=\frac{1}{2}\left(|x+y|+|x-y|\right)\tag2
$$

为了近似表示 max 函数，我们可以先寻找绝对值的近似函数，绝对值函数的导数如下：

$$
f'(x) = \left\{\begin{aligned}1,&\,x > 0\\ 
-1,&\, x < 0\end{aligned}\right.\tag3
$$

我们使用 [单位阶跃函数](https://baike.baidu.com/item/%E5%8D%95%E4%BD%8D%E9%98%B6%E8%B7%83%E5%87%BD%E6%95%B0/1714368?fr=aladdin) 来进行近似：

$$
\theta(x) = \left\{\begin{aligned}1,&\,x > 0\\ 
0,&\, x < 0\end{aligned}\right.\tag4
$$

$$
f'(x)=2\theta(x)-1\tag5
$$

我们可以通过 $\theta(x)$ 的近似函数来近似 max 函数。在物理中其常用近似是：

$$
\theta(x)=\lim_{k\to +\infty} \frac{1}{1+e^{-k x}}
$$

将该式带入 (5) 式，积分可得：

$$
|x|=\lim_{k\to +\infty} \frac{1}{k}\ln(e^{kx}+e^{-kx})
$$

这样便可以得到 max 的近似公式：

$$
\max(x,y)=\lim_{k\to +\infty} \frac{1}{2k}\ln(e^{2kx}+e^{-2kx}+e^{2ky}+e^{-2ky})
$$

由于 $x\geq0,y\geq0$，$e^{-2kx}$ 和 $e^{-2ky}$ 趋近于 0，可以进一步化简为：

$$
\max(x,y)=\lim_{k\to +\infty} \frac{1}{k}\ln(e^{kx}+e^{ky})
$$

并且上式满足任意实数，同时甚至可以推广到多变量：

$$
\max(x,y,z,\dots)=\lim_{k\to +\infty} \frac{1}{k}\ln(e^{kx}+e^{ky}+e^{kz}+\dots)
$$

但是这里的 k 应该趋向于正无穷，和 LogSumExp 有什么关系呢？

在模型中，我们通常将 K 设置为 1，这等价于把 KK 融合到模型自身之中，让模型自身决定 K 的大小。

### 统一 loss 形式下的交叉熵函数

$$
\log \sum\limits_{i=1}^n e^{s_i-s_t}\approx max\begin{pmatrix}0 \\ s_1 - s_t \\ \vdots \\ s_{t-1} - s_t \\ s_{t+1} - s_t \\ \vdots \\ s_n - s_t\end{pmatrix}
$$

我们只需注意这个式子，他能说明 softmax+ 交叉熵损失函数为什么有效。

通过上文我们已经知道了上式是 max 的光滑近似，所以这个式子便等效于求其他非目标类别与目标类别之间的差值的最大值，并且希望让这个最大值能够小于 0，因为目标类别的分数减去自身等于 0，这样便可以保证目标类别得分都大于非目标类别。

### 多标签分类

我们在前面已经得到了统一形式下的交叉熵函数，我们仿照其形式将目标分为正例和负例，可得下式：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}\right)=\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:unified}
$$

当 k 固定式，可直接使用上式，如果 k 不确定，我们按照之前的方法添加一个额外的 0 类，希望目标的分数都大于 $s_0$，非目标的分数都小于 $s_0$，可得下式：

$$
\begin{aligned} 
&\log \left(1 + \sum\limits_{i\in\Omega_{neg},j\in\Omega_{pos}} e^{s_i-s_j}+\sum\limits_{i\in\Omega_{neg}} e^{s_i-s_0}+\sum\limits_{j\in\Omega_{pos}} e^{s_0-s_j}\right)\\ 
=&\log \left(e^{s_0} + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(e^{-s_0} + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\\ 
\end{aligned}
$$

如果指定阈值为 0，可化简为：

$$
\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\tag6
$$

因此这里训练时就不需要额外的添加一个类了，下面给出了代码实现：

```python
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred # 将正例乘以-1，负例乘以1
    y_pred_neg = y_pred - y_true * 1e12 # 将正例变为负无穷，消除影响
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # 将负例变为负无穷
    zeros = torch.zeros_like(y_pred[..., :1]) 
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 0阈值
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss
```

## 稀疏版多标签分类交叉熵

多标签分类交叉熵不仅仅可以用于多标签分类任务，很多任务都可以使用，只要满足是 $n$ 选 $k$ 即可，苏剑林大佬给出了一个例子 Global pointer，在 cv 领域，比如说代替目标检测的 focal loss。

当某些任务中正负例极不均衡（这里是正例远远小于负例），并且标签尺寸十分巨大时，我们可以更换策略：

$$
\begin{aligned} 
&\,\log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right) = \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i} - \sum\limits_{i\in \mathcal{P}} e^{S_i}\right) \\ 
=&\, \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right) + \log \left(1 - \left(\sum\limits_{i\in \mathcal{P}} e^{S_i}\right)\Bigg/\left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right)\right) 
\end{aligned}
$$

负例的 loss 可以写为全集减去正例，这样制作标签时我们就只需要保存正例的标签，训练时通过正例标签直接索引进行计算即可，作者经过实验发现在 Global Pointer 上训练速度提高 1.5 倍并且精度不会下降。

作者给出了 TensorFlow 的代码，然而网络上却找不到 Pytorch 版本的代码，因此我尝试进行了复现，并且发布在我的 github 上：https://github.com/Asthestarsfalll/Sparse_MultiLabel_Categorical_CrossEntropy

```python
def sparse_multilabel_categorical_crossentropy(label: Tensor, pred: Tensor, mask_zero=False, reduction='none'):
    """Sparse Multilabel Categorical CrossEntropy
        Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
    Args:
        label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
            should contain the indexes of the positive rather than a ont-hot vector
        pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
        mask_zero: if label is used zero padding to align, please specify make_zero=True.
            when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.
    """
    zeros = torch.zeros_like(pred[..., :1])
    pred = torch.cat([pred, zeros], dim=-1)
    if mask_zero:
        infs = torch.ones_like(zeros) * float('inf')
        pred = torch.cat([infs, pred[..., 1:]], dim=-1)
    pos_2 = batch_gather(pred, label)
    pos_1 = torch.cat([pos_2, zeros], dim=-1)
    if mask_zero:
        pred = torch.cat([-infs, pred[..., 1:]], dim=-1)
        pos_2 = batch_gather(pred, label)
    pos_loss = torch.logsumexp(-pos_1, dim=-1)
    all_loss = torch.logsumexp(pred, dim=-1)
    aux_loss = torch.logsumexp(pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-16, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = pos_loss + neg_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(self.reduction))
```

对其中的要点进行讲解：

1. 当标签需要使用 zero padding 来对齐时，需要将标签的值加一；
2. 指定 mask_zero 为 True，因为会通过 label 在 pred 中索引出正例，填充的 0 值会造成影响，因此在 pred 最前面 concat 一个无穷的量，输入 LogSumExp 进行计算时其结果为 0。
3. 关于是否要将类别数由 num_classes 改为 num_classes+1，我认为是不需要的，因为该 loss 已经显式的使用 0 来表示我们需要的额外类别数的得分，并且通过之前的分析，可以很直观的看到 (6) 式的目的其实就是为了让正例得分大于 0，负例得分小于 0，推理时直接输出得分大于 0 的类别即可。、

另外，pytorch 没有 batch_gather 的 API，因此根据 loss 的要求简单实现了一个：

```python
def batch_gather(input: Tensor, indices: Tensor):
    """
    Args:
        input: label tensor with shape [batch_size, n, L] or [batch_size, L]
        indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]
    Return:
        Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input 
        if m == n, the return shape is [batch_size, m, l]
        if m != n, the return shape is [batch_size, n, l*m]
    """
    if indices.dtype != torch.int64:
        indices = torch.tensor(indices, dtype=torch.int64)
    results = []
    for data, indice in zip(input, indices):
        if len(indice) < len(data):
            indice = indice.reshape(-1)
            results.append(data[..., indice])
        else:
            indice_dim = indice.ndim
            results.append(torch.gather(data, dim=indice_dim-1, index=indice))
    return torch.stack(results)
```
