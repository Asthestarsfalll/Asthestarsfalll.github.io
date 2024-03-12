---
title: 量化那些事
description: 神魔是量化捏。
authors:
  - Asthestarsfalll
tags:
  - BaseLearn
hide_table_of_contents: false
---

[参考](https://megengine.org.cn/doc/stable/zh/user-guide/model-development/quantization/index.html)

## 量化

常见的神经网络模型中所使用的数据类型多是 `float32`，但是针对不同的应用场景，我们通常需要不同的数据精度，来达到加速、轻量的目的，而把模型转化为 `int8` 这样的低精度数据类型的过程则被称为**量化**（Quantization）。

量化能将 32 位的浮点数转换成 8 位甚至是 4 位定点数，具有更少的运行时内存和缓存要求； 另外由于大部分的硬件对于定点运算都有特定的优化，所以在运行速度上也会有较大的提升。 相较于普通模型， **量化模型有着更小的内存容量与带宽占用、更低的功耗和更快的推理速度等优点。**

从直觉上来说，从 32 位量化到 8 位舍弃了大量的精度，这似乎会导致模型性能损失严重，但实际上，通过一系列精妙的量化处理之后，模型的性能损失微乎其微，并能正常地部署使用。

## 原理

量化就是将基于浮点数据类型的模型转换为定点数进行运算，其核心就是如何用定点数去表示模型中的浮点数，以及如何用定点运算去表示对应的浮点运算。  

以 float32 转 uint8 为例，一种最简单的转换方法是直接舍去 float32 的小数部分，只取其中的整数部分，并且对于超出 (0,255) 表示范围的值用 0 或者 255 表示。 这种方案显然是不合适的，尤其是深度神经网络经过 bn 处理后，其中间层输出基本都是 0 均值，1 方差的数据范围，在这种方案下因为小数部分被舍弃掉了，会带来大量的 精度损失。并且因为 (0,255) 以外的部分被 clip 到了 0 或 255，当浮点数为负或者大于 255 的情况下，会导致巨大的误差。

目前主流的浮点转定点方案基本采用均匀量化，因为这种方案对推理更友好。将一个浮点数根据其值域范围，均匀的映射到一个定点数的表达范围上。

### 均匀量化

均匀量化能将数据的值域均匀的缩放到 0 到 255 之间：

假设一个浮点数 x 的值域范围为 ${xmin,xmax}$，要转换到一个表达范围为 $(0,255)$ 的 8bit 定点数的转换公式如下

$$
x_{int}=round(x/s)+z\\
x_Q=clamp(0,255,xint)
$$

其中 s 为 scale，也叫步长，是个浮点数。z 为零点，即浮点数中的 0，是个定点数。

$$
scale=(xmax−xmin)/255\\z=round(0−xmin)/255
$$

由上可以看出均匀量化方案对于任意的值域范围都能表达相对不错的性能，不会存在类型转换方案的过小值域丢精度和过大值域无法表示的情况。 代价是需要额外引入零点 z 和值域 s 两个变量。同时我们可以看出，均匀量化方案因为 round 和 clamp 操作也是存在精度损失的，所以会对模型的性能产生影响。 如何减轻数据从浮点转换到定点的精度损失，是整个量化研究的重点。

**这里的零点是由于网络模型的 padding 与 relu 等算子对 0 较为敏感，因此需要求出零点。**

通过上式我们可以将量化后的数据“反量化”回浮点数：

$$
x_{floor} = (x_Q-z)*s
$$

接下来我们来看看如何用经过量化运算的定点卷积运算去表示一个原始的浮点卷积操作：

$$
\begin{align}
conv(x,w)&=conv((x_Q−z_x)∗s_x,(w_Q−z_w)∗s_w)
\\&=s_xs_wconv(x_Q−z_x,w_Q−z_w)\\
&=s_xs_w(conv(x_Q,w_Q)−z_x\sum_{k,l,m}x_Q−z_w\sum_{k,l,m,n}w_Q+z_xz_w)
\end{align}
$$

其中 k,l,m,n 分别是，kernel_size，output_channel 和 input_channel 的遍历下标。

当卷积的输入和参数的**零点**都是 0 时，上式可以化简为：

$$
conv(x,w)=s_xs_w(conv(x_Q,w_Q))
$$

可以直观地看到，定点卷积与浮点卷积的结果只有一个 scale 上的偏差，因此我们通常是用对称均匀量化。

### 对称均匀量化

如上文所言，我们将零点固定为 0，以 int8 为例，量化公式如下：

$$
scale=max(abs(x_{min}),abs(x_{max}))/127\\
x_{int}=round(x/s)\\
x_Q=clamp(−128,127,x_{int})
$$

出于利用更快的 SIMD 实现的目的，我们会把卷积的 weight 的定点范围表示成 (-127,127)，对应的反量化操作为

$$
x_{float}=x_Q∗s
$$

由此可见，对称均匀量化的量化和反量化操作会更加的便捷一些。除此之外还有随机均匀量化等别的量化手段。

### 非对称量化

略

### 值域统计

均匀量化里的关键就是 scale 和 zero_point，而它们是通过浮点数的值域范围来确定的。我们如何确定网络中每个需要量化的数据 的值域范围呢，一般有以下两种方案:

1. 根据经验手动设定值域范围，在缺乏数据的时候可以这样做；
2. 跑一批少量数据，根据统计量来进行设定，这里统计方式可以视数据特性而定。

### 量化感知训练

量化前后的误差主要取决于模型的参数和激活值分布与均匀分布的差异。对于量化友好的模型，我们只需要通过值域统计得到其值域范围，然后调用对应的量化方案进行定点化就可以了。但是对于量化不友好的模型，直接进行量化会因为误差较大而使得 最后模型的正确率过低而无法使用。**有没有一种方法可以在训练的时候就提升模型对量化的友好度呢**？

答案是有的，我们可以通过在训练过程中，给待量化参数进行量化和反量化的操作，便可以引入量化带来的精度损失，然后通过训练让网络**逐渐适应这种干扰**，从而使得网络在真正量化后的表现与训练表现一致。这便是量化感知训练，也叫 qat (Quantization-aware-training)

其中需要注意的是，因为量化操作不可导，所以在实际训练的时候做了一步近似，把上一层的导数直接跳过量化反量化操作传递给了当前参数。

### 量化网络的推理流程

对于现成网络，我们可以在每个卷积层前后加上量化与反量化的操作，这样就实现了用定点运算替代浮点运算的目的。  

我们也可以在整个网络推理过程中维护每个量化变量对应的 scale 变量，这样我们可以在不进行反量化的情况下走完 整个网络，这样我们除了带来极少量额外的 scale 计算开销外，便可以将整个网络的浮点运算转换成对应的定点运算。具体流程可以 参考下图。

![quantization-inference.jpg](/images/2022/03/27/quantization-inference.png)

只在输入输出进行量化和反量化。

## 量化基本流程

目前工业界主要有两种量化技术：

1. 训练后量化（Post-Training Quantization, PTQ）；
2. 量化感知训练（Quantization-Aware Training, QAT）。

### 训练后量化

正如上文提到的，训练后量化需要网络中权重的一些统计量——零点（zero_point）和缩放因子（scale），这些量的获取方式也在上文提到过，使用训练后量化技术，会导致量化后的模型掉点（即预测正确率下降），严重情况下会导致量化模型不可用，一种可行的方法就是在模型训练过程中插入观察者（Observer）来获取这些统计量，或者使用小批量数据在量化前对 Observer 进行校准。

### 量化感知

量化感知训练技术，即向浮点模型中插入一些伪量化（FakeQuantize）算子作为改造， 在训练时伪量化算子会根据 Observer 观察到的信息进行量化模拟， 即模拟计算过程中数值截断后精度降低的情形，先做一遍数值转换，再将转换后的值还原成原类型。 这样可以让被量化对象在训练时 “提前适应” 量化操作，缓解在训练后量化时带来的掉点影响。

新增的 FakeQuantize 算子会引入大量的训练开销，为了节省总用时，模型量化更通常的思路是：

1. 按照平时训练模型的流程，设计好 Float 模型并进行训练（等同于得到一个预训练模型）；
2. 插入 Observer 和 FakeQuantize 算子，得到 Quantized-Float 模型（简称 QFloat 模型），量化感知训练；
3. 进行训练后量化，得到真正的 Quantized 模型（简称 Q 模型），即最终被用作推理的低比特模型。

![image-20220226211626007](/images/2022/03/27/image-20220226211626007.png)

**此时的量化感知训练 QAT 可被看作是在预训练好的 QFloat 模型上微调（Fine-tune），同时做了校准。**

在不考虑训练开销的情况下，为了简化整体流程，可以直接构造 QFloat 模型，并进行训练与后量化：

<img src="/images/2022/03/27/image-20220226211818907.png" alt="image-20220226211818907" style={{zoom:"80%"}} />

## 代码

### MegEngine

在 MegEngine 中，最上层的量化接口是配置如何量化的 [`QConfig`](https://megengine.org.cn/doc/stable/zh/reference/api/megengine.quantization.QConfig.html#megengine.quantization.QConfig) 和模型转换模块里的 [`quantize_qat`](https://megengine.org.cn/doc/stable/zh/reference/api/megengine.quantization.quantize_qat.html#megengine.quantization.quantize_qat) 与 [`quantize`](https://megengine.org.cn/doc/stable/zh/reference/api/megengine.quantization.quantize.html#megengine.quantization.quantize) .

通过配置 QConfig 中所使用的 Observer 和 FakeQuantize 算子，可以对量化方案进行自定义。

大概的流程如下：

```python
import megengine.quantization as Q

model = ... # The pre-trained float model that needs to be quantified

Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig) # EMA is a built-in QConfig for QAT

for _ in range(...):
    train(model)

Q.quantize(model) # Truly quantized.
```

通过配置 QConfig 可以配置各种各样的量化方案，详细看官网。

### PyTorch

[参考](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#)

对于训练后的静态量化，我们需要对源码进行一些必要的修改：

1. 使用 `nn.quantized.FloatFunctional` 来替换 `+`；

2. 在前向过程的开头和结尾分别添加 `QuantStub` 和 `DeQuantstub`;

3. 对于某些网络，往往会使用 ReLU6 来替换 ReLU，因为 ReLU6 限制了激活值的范围，更适合定点量化。

4. 将 Conv Bn ReLu 融合起来，大致代码如下：

   ```python
   def fuse_model(self):
       for m in self.modules():
           if xxx:
               torch.quantization.fuse_modules(m, [xxx], inplace=True)
   ```

接下来我们在数据集上以 eval 模式统计少批次的数据， 在这之前需要先进行 `fuse_model` 来插入，并且使用 qconfig 来进行配置量化方案，大概流程如下：

```python
# Load pretrained model
float_model = load_model(saved_model_dir + float_model_file).to('cpu')
# Set eval
float_model.eval()

# Fuse Conv, bn and relu
float_model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
float_model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(float_model, inplace=True)

# Calibrate with the training set
evaluate(float_model, criterion, data_loader, neval_batches=32)

# Convert to quantized model
torch.quantization.convert(float_model, inplace=True)

# Evaluate preformance loss
evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)

```

训练后量化精度往往很差，我们可以使用不同的量化配置来提高精度，如 `per_channel_quantized`：

```python
# Load pretrained model
per_channel_quantized_model = load_model(saved_model_dir + float_model_file)

# Set eval
per_channel_quantized_model.eval()

# Fuse model
per_channel_quantized_model.fuse_model()

# Set pre_channel_quantize config
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)

evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
# Convert to quantized model
torch.quantization.convert(per_channel_quantized_model, inplace=True)

# Evaluate preformance loss
evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=32)

# Save
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
```

对于量化感知，像正常训练一样即可：

```python
qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
```

随后进行伪量化：

```python
torch.quantization.prepare_qat(qat_model, inplace=True)
```

对于不同的阶段，我们可以使用不同的训练策略，如冻结观察者、冻结 bn 层来对权重进行微调：

```python
# freeze observer
qat_model.apply(torch.quantization.disable_observer)
# freeze bn
qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
```
