---
title: Efficient Channel Attention for Deep Convolutional Neural Networks
authors:
  - Asthestarsfalll
tags:
  - PaperRead
  - attention
description: 新注意力机制：ECANet主要对SENet模块进行了一些改进，提出了一种不降维的局部跨信道交互策略（ECA模块）和自适应选择一维卷积核大小的方法，从而实现了性能上的提优。
hide_table_of_contents: false
---

> 论文名称：[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.03151)
>
> 作者：Qilong Wang , Banggu Wu , Pengfei Zhu , Peihua Li , Wangmeng Zuo , Qinghua Hu
>
> Code：https://link.zhihu.com/?target=https%3A//github.com/BangguWu/ECANet

## 摘要

> Recently, channel attention mechanism has demonstrated to offer great potential in improving the performmance of deep convolutional neural networks (CNNs).
> However, most existing methods dedicate to developing more sophisticated attention modules for achieving better performance, which inevitably increase model complexity.
> To overcome the paradox of performance and complexity trade-off, this paper proposes an Efficient Channel Attention (ECA) module, which only involves a handful of parameters while bringing clear performance gain. By dissecting the channel attention module in SENet, we empirically show avoiding dimensionality reduction is important for learning channel attention, and appropriate cross-channel interaction can preserve performance while significantly decreaseing model complexity. Therefore, we propose a local crosschannel interaction strategy without dimensionality reduction, which can be efficiently implemented via 1D convolution. Furthermore, we develop a method to adaptively select kernel size of 1D convolution, determining coverage of local cross-channel interaction. The proposed ECA module is efficient yet effective, e.g., the parameters and computations of our modules against backbone of ResNet50 are 80 vs. 24.37M and 4.7e-4 GFLOPs vs. 3.86 GFLOPs, respectively, and the performance boost is more than 2% in terms of Top-1 accuracy. We extensively evaluate our ECA module on image classification, object detection and instance segmentation with backbones of ResNets and MobileNetV2.
> The experimental results show our module is more efficient while performing favorably against its counterparts.

本文提出了一种无降维的局部跨通道交互策略，通过一维卷积实现了一个即插即用超轻量级的注意力模块，并且带来了明显的性能增益。

在图像分类、目标检测和实例分割方面 t 进行了广泛的评估，实验结果如下：

<img src="/images/2022/03/27/image-20210809123653844.png" alt="image-20210809123653844" style={{zoom:"67%"}} />

## 方法

本文分析了 $SENet$ 中通道降维和跨通道交互的相互影响，最终提出 $ECA$ 模块，并且开发了一种自适应确定 $ECA$ 参数的方法。

### 降维和跨通道交互

<img src="/images/2022/03/27/image-20210809130112961.png" alt="image-20210809130112961" style={{zoom:"67%"}} />

如上图，本文进行了对比实验，其中 $SE-Var$ 是 $SE$ 的变体，它们都没有进行降维，$Var1$ 直接应用 $GAP$ 的结果，$Var2$ 在每个通道上使用了可学习的参数，$Var3$ 使用了一层 $FC$。

上述结果可能表明，通道与其权重需要直接对应，避免维度减少比考虑通道间的非线性依赖更重要，Var2 与 Var3 的结果表明，通道间的交互也是十分重要的，因此再次进行实验。

$SE-GC$​表示使用了分组卷积，然而结果表明，分组卷积可能完全抛弃了不同组之间的依赖关系，这更加验证了通道交互的重要性。

### ECA

经过上述实验和分析，现提出一种轻量高效的方法在不降维的情况下来建模相邻通道的关系，而一维卷积恰好可以胜任这个任务。

<img src="/images/2022/03/27/image-20210809131747806.png" alt="image-20210809131747806" style={{zoom:"67%"}} />

#### 确定 k 的大小

设计了一种自适应方法

$$
K= \psi(C)=odd(\frac{ \log_2(C)}{\gamma}+\frac{b}{\gamma})
$$

显然，$K$ 的大小与通道数呈一种映射关系是合理的，鉴于通道一般都是 $2$ 的次方，卷积核通常是奇数，使用上述公式，$ood()$ 表示最近的奇数，本文中 $\gamma=2、b=1$​​。

<img src="/images/2022/03/27/image-20210809134014695.png" alt="image-20210809134014695" style={{zoom:"67%"}} />

通过实验验证了自适应方法的有效性。

#### 代码复现

```python
class ECA(nn.Module):
    def __init__(self, in_ch, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(in_ch, 2)+b)/gamma))
        k = t if t % 2 else t+1
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=int(k/2), bias=False)

    def forward(self, x):
        atten = self.conv(self.gap(x).squeeze(-1).premute(0, 2, 1))
        atten = torch.sigmoid(atten.permute(0, 2, 1).unsqueeze(-1))
        return atten*x

```

### 效果

在各个任务上都有良好的表现

## 总结

本文致力于学习低模型复杂度的深层 $CNN$​的有效通道。为此，提出了一种高效的信道注意（$ECA$​）模块，该模块通过一维卷积产生通道注意，其内核大小可通过通道维数的非线性映射自适应确定。

实验结果表明，$ECA$​是一种非常轻量级的即插即用模块，可提高各种深度 $CNN$​体系结构的性能，包括广泛使用的 $ResNet$​和轻量级 $MobileNetV2$​。此外，我们的 $ECA$​网络在目标检测和实例分割任务中表现出良好的泛化能力。
