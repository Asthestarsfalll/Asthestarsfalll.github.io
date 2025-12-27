---
title: Music_translate
description: 钢琴转谱是一项将钢琴录音转为音乐符号（如 MIDI 格式）的任务。在人工智能领域，钢琴转谱被类比于音乐领域的语音识别任务。然而长期以来，在计算机音乐领域一直缺少一个大规模的钢琴 MIDI 数据集。GiantMIDI-Piano 将所有古典钢琴作品转录成 MIDI 格式，并向全世界开放，此举旨在推动音乐科技和计算机音乐学的发展。
authors:
  - Asthestarsfalll
tags:
  - PaperRead
hide_table_of_contents: false
---

> 论文名称：High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times
>
> 作者： Qiuqiang Kong, Bochen Li, Xuchen Song, Yuan Wan, Yuxuan Wang
>
> Code：https://github.com/bytedance/piano_transcription

## 前言

`自动音乐转谱`（Automatic music transcription）是将音频转化为符号表示的任务，例如 Piano Rolls, guitar fretboard chart 和 Music Instrument Digital Interface(MIDI)，AMT 之于人工智能音乐就如同词嵌入之于自然语言处理一般不可或缺，其发展将会使下游任务受益匪浅。

钢琴转谱 AMT 中的一个具有较高挑战性的任务，它包含音高（pitch）、起音（onset）、偏移（offset）和速度（velocity，实际上指得是击键的速度，也就是音符的力度）（等音乐事件，其难点主要在于钢琴的复音（多个键同时被按下）

在之前的工作中，提出了用于多基音估计的概率谱平滑原理；频域和时域相结合的钢琴转录方法；非负矩阵分解 (NMF) 用于将频谱分解为复音音符，等等等等

这些转谱系统都需要先将音频分为帧（frames），每帧中储存着其音高、起音、偏移等信息，然而这存在一些问题：

1. 钢琴音符的起音可以持续几帧而不是一帧

2. 这些系统对标签和音频的错位很敏感，例如某个起始点错位了几帧，整个训练将会受到影响

3. 为起音和偏移事件分配标签时存在歧义，例如由于混响和淡出效果，音符的偏移并不明显

4. 系统的精度取决于跳帧的长度

5. 缺乏对延音踏板的研究

针对上述问题，提出了一个高分辨率的踏板预测钢琴转谱系统，它将音符视作一个个连续的量，分为起音、衰减、延音和释音，并且能实现任意分辨率的转谱

   ## 相关工作

**RAME-WISE TRANSCRIPTON**

这类转谱网络主要分为以下步骤：

1. 将音频的频谱转为 `对数梅尔谱`，输入大小为 $T\times F$，$T$ 为采样的帧数，$F$ 为对数梅尔谱的采样个数
2. 将其输入分类网络中（实际上就是分类 piano rolls 上的每一个音符的存在与否），输出为 $Piano-Rolls$，大小为 $T\times K$，$K$ 为钢琴音符的数量，一般为 88
3. 一般使用交叉熵函数，损失函数定义为 $l_{on}=\sum_{t=1}^{T}\sum_{k=1}^{K}l_{bce}(I_{on}(t,k),P_{on}(t,k))\\$,$l_{fr}=\sum_{t=1}^{T}\sum_{k=1}^{K}l_{bce}(I_{fr}(t,k),P_{fr}(t,k))\\$

**Onsets and frames transcription systems**

改进了 frame-wise transcription systems 需要对钢琴音符时间的进行精细处理，引入了 `onset` 与 `frame` 结合，能够得到更加丰富的信息。其中每一帧都会被分配 1 或 0 来表示 `onset` 或 `offset`

## HIGH-RESOLUTION PIANO TRANSCRIPTION

<img src="/images/2022/03/27/image-20210719195936787.png" style={{zoom:"80%"}} />

图中红线表示实际音符开始的位置，方格则表示帧

可以看到前三种转谱方法虽然能够愈发精细地建模钢琴声音的起落、延续和衰减，但是并不能得到音符精确的开始位置，因此其转谱的分辨率是有限的

通过预测钢琴音符的连续起始和结束时间来进行建模，而非对每一帧中起始和结束存在的概率进行预测，这个灵感来源于 `YOLO`，直接预测每帧中心和音符精确开始或结束时间之间的距离，因此，该方法理论上可以以任意分辨率捕获精确的起始和结束的信息。

在训练中，我们通过函数 $g$ 将时间差 $\Delta_i$ 编码为 $g(\Delta_i)$

$$
\begin{cases}
g(\Delta_i)=1-\frac{|\Delta_i|}{J\Delta},|i|<J\\
g(\Delta_i)=0,|i|>J
\end{cases}
$$

这里的 $J$ 是一个控制目标清晰度 `sharpness` 的超参数

较大的 $J$ 表示更 smoother 的目标

较小的 $J$ 表示更 sharper 的目标

当 $J=1,\Delta_0=0$ 时，等价与 `Onsets&frames`

下图表示了当 $J=5$ 时的情况

![image-20210720185219639](/images/2022/03/27/image-20210720185219639.png)

可以看到，它与 `Attack&decay` 有一定的相似性，然是该方法能够包含更精准的音符开始时间

## 细节

### Regress onsets and offsets times

如上述所说，通过预测钢琴音符的连续起始和结束时间来进行建模

在训练中，`onsets` 和 `offsets` 的回归目标都是形状为 $T\times k$ 的矩阵，其值介于 0 与 1 之间，设其为 $G_{on},G_{off}$，设预测值为 $R_{on},R_{off}$，则其 loss 定义为：

$$
l_{on}=\sum_{t=1}^{T}\sum_{k=1}^{K}l_{bce}(G_{on}(t,k),R_{on}(t,k))\\
l_{off}=\sum_{t=1}^{T}\sum_{k=1}^{K}l_{bce}(G_{off}(t,k),R_{off}(t,k))
$$

### Velocity estimation

$Velocity$ 在这里表示的是手指敲击琴键的速度，简介代表着声音的响度，在 MIDI 文件中，其值为 $[0,128)$，先将其归一化到 $[0,1)$，与预测 onset 和 offset 类似，其损失函数定义为：

$$
l_{vel}=\sum_{t=1}^{T}I_{on}(t,k)\cdot l_{bce}(I_{vel}(t,k),P_{vel}(t,k))
$$

这里的 $I_{vel},P_{vel}$ 分别代表标签和预测，$I_{on}$ 为 01 二值矩阵，1 表示 onset，因为只需要预测存在 onset 的帧的响度

最后乘以 128 放缩回去

###  Entire system

![image-20210720205303278](/images/2022/03/27/image-20210720205303278.png)

与其他系统的输入相同，将音频转换为形状为 $T\times F$ 的对数梅尔谱矩阵

整个系统主要分为四个模块——`速度回归`、`起始回归`、`逐帧分类` 和 `结束回归`

在每个模块中，使用多层卷积进行建模，然后使用双向门控循环单元 (biGRU)，其中卷积层用于提取高级信息，而 biGRU 用于提取长距离信息

在 GRU 之后会使用全连接层进行回归或者分类

在这里，使用**力度回归**的结果对**起始回归**的预测进行指导，显然它们存在着相互影响的关系

同理，使用**起始回归**和**结束回归**的结果对**逐帧分类**的预测进行指导

总的损失函数如下（都在上文提到过）：

$$
l_{note}=l_{fr}+l_{on}+l_{off}+l_{vel}
$$

### Inference

最终输出时，将结果处理为 $[note,onset,offset,velocity]$ 的格式，我们会得到如图所示结果，然而，这样的分辨率仍然受限于跳帧大小，因此提出一种高分辨率转谱方法

![image-20210720185219639](D:/UserData/Downloads/images-master/images-master/image-20210720185219639.png)

1. 首先，对上图的 `起始回归` 预测进行局部最大值的检测，若大于起始阈值（onset threshold），则说该帧附近存在 $onset$ 或 $offset$

2. 接下來分析其精确的开始或结束时间。对于具有局部最大值的帧，取其前后各一帧组成三帧三元组，分别表示为 A、B、C

   <img src="/images/2022/03/27/image-20210721201947507.png" alt="image-20210721201947507" style={{zoom:"80%"}} />

   点 G 则是精确时间，我们认为 AG 和 CG 关于垂线 GI 对称，假设，C 的输出值大于 A，由相似三角形容易得到：

   $$
   BH=\frac{x_B-x_{A}}{2}\frac{y_C-y_A}{y_B-y_A}
   
   
   $$

3. 同理，当 `结束回归` 超过结束阈值或者帧预测低于帧阈值，则检测到音符结束

整个转谱算法的伪代码如下:

<img src="/images/2022/03/27/image-20210721210115506.png" alt="image-20210721210115506" style={{zoom:"50%"}} />

### SUSTAIN PEDAL TRANSCRIPTION

**延音踏板是钢琴的重要组成部分之一**，当踩下延音踏板时，制动器将从琴弦上移开以使琴弦自由振动来达到延音的效果，然而，之前的很多工作都没有将踏板预测包含其中。

在 MIDI 格式中，踏板的值为 $[0,128)$，为了简化这个问题，本系统仅会预测踏板的 `on` 和 `off`，并且不考虑高级延音踏板技术，如半踏板等，MIDI 值大于 64 的被视为 on，小于 64 的被视为 off。

与音符预测类似，将踏板预测目标分为 `起始回归`、`结束回归` 和 `逐帧分类`，分别表示为 $G_{ped\_on},G_{ped\_off},I_{ped\_fr}$，逐帧目标依然是 01 二值

整个踏板预测系统的损失函数为：

$$
l_{ped\_on}=\sum_{T=1}^{T}l_{bce}(R_{ped\_on}(t),P_{ped\_on}(t))\\
l_{ped\_off}=\sum_{T=1}^{T}l_{bce}(R_{ped\_off}(t),P_{ped\_off}(t))\\
l_{ped\_fr}=\sum_{T=1}^{T}l_{bce}(I_{ped\_fr}(t),R_{ped\_fr}(t))\\
l_{ped}=l_{ped\_on}+l_{ped\_off}+l_{ped\_fr}
$$

踏板预测的伪代码如下：

<img src="/images/2022/03/27/image-20210721210340582.png" alt="image-20210721210340582" style={{zoom:"50%"}} />

### EXPERIMENTS

**数据集**：使用 MAESTRO 数据集，一个包含成对录音和 MIDI 文件的大规模数据集，集成了超过 200 小时的钢琴独奏曲，时间分辨率约为 3 毫秒，每个音乐录音都包含作曲家、名称和表演年份的信息

**预处理**：
