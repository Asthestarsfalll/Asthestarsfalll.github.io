---
title: MIDI
description: midi文件详解
authors: [Asthestarsfalll]
tags: [other]
hide_table_of_contents: false
---

 [参考](https://www.bilibili.com/read/cv1753143/)

## 数据结构介绍

MIDI 的数据结构被称为 `Chunk`，有 `Header Chunk` 和 `Track Chunk ` 两种

### Hearder Chunk

每个 MIDI 文件的开头都有如下的十六进制编码，即 Header Chunk：`4d 54 68 64 00 00 00 06 ff ff nn nn dd dd`

前四个是字符 `MThd` 的 ASCLL 码

接下来四个字节是指明文件头表述部分的字节数，总是 `00 00 00 06`

随后的 `ff ff` 表示 MIDI 文件的格式，有三种，0、1 和 2 格式分别用 `00 00`、`00 01` 和 `00 02` 表示

这三个格式分别表示：

- 只有一个音轨
- 多个音轨，同时播放
- 多个音轨，串行播放

`nn nn` 则表示 MIDI 文件的音轨数

- 若为 0 格式，则为 `00 01`
- 若为 1 格式，则为音轨数加 1,因为第一个音轨是 `Tempo Map`，不储存演奏信息
- 2 格式不详

最后的 `dd dd` 表示文件的时间类型，MIDI 常用的时间类型有两种，一种基于 `TPQN（Ticks Per Quarter-Note）`，每个四分音符表示的 `Ticks` 数量，Ticks 是 MIDi 中最小的时间单位，另一种是基于 SMPTE 时间码的时间度量法，不详谈

### Track Chunk

同样的 16 进制编码：`4D 54 72 6B pp pp pp pp xx yy xx yy … 00 FF 2F 00`

前四个字节表示 `MTrk`

`pp pp pp pp` 表示包含数据大小，是可变的

`xx yy` 表示 Delta-time 和 MIDI 事件

`00 FF 2F 00` 表示结束

### Delta-time

Delta 即 $\Delta$，表示变化量的意思，表示两个事件之间相差的时间

为了能够表示足够长的时间，Delta-time 使用可变长度数的格式，最长可以表示 0x0fffffff.

### MIDI 事件

MIDI 事件包括实际需要发送出去的 MIDI 事件，和 meta-event 事件。

这里要注意 MIDI 文件的“状态字省略”特点。为了减少 MIDI 文件的体积，人们规定：如果同一 Track Chunk 中的下一条 MIDI 事件，和上一条事件，属于同一类型同一通道的事件（即状态字相同）时，下一条事件的状态字可以省略，而只需记录数据。

MIDI 文件中的事件，大多数都是“Note-On”和“Note-Off”事件，其中“Note-Off”事件也可以用“Note-On”+“力度为 0”来表示 ，遇到这样的情况千万不要慌张

我们可以用事件种类 + 参数来表示，如下图所示

![image-20210622182027202](/images/2022/03/27/image-20210622182027202.png)
