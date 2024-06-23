---
title: GPT4RoI-Instruction Tuning Large Language Model on Region-of-Interest
tags:
  - PaperRead
  - LLM
  - multimodal
hide_table_of_contents: false
---

>论文名称：[GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest](https://arxiv.org/pdf/2307.03601)
>Code: https://github.com/jshilong/GPT4RoI

## 介绍
1. 视觉指令调整大型语言模型（LLM）在图像-文本对上已经实现了通用的视觉-语言能力。然而，区域-文本对的缺乏限制了它们在细粒度多模态理解方面的进展。
2.在本文中，我们提出了空间指令调整，它在指令中引入了对感兴趣区域（RoI）的引用。在发送到 LLM 之前，引用被 RoI 特征替换，并与语言嵌入交错作为序列。
3. 超越语言的交互：用户可以通过语言和绘制边界框与我们的模型交互，灵活调整引用的粒度。
4. 多样化的多模态能力：GPT4RoI 可以挖掘每个 RoI 内的多种属性信息，例如颜色、形状、材料、动作等。
5. 它可以基于常识对多个RoI进行推理。


关键词：spatial instruction learning。


![]('./images/240622_18h08m15s_screenshot.png')


## GPT4RoI

![](./images/240623_14h19m39s_screenshot.png)

整体由视觉编码器、图像级特征投影器、区域特征提取器和大型语言模型（LLM）组成，实际上还是LLaVA架构，这里的区域特征提取器采用了目标检测领域中广泛使用的模块，主要是基于视觉编码器构建了一个多级的特征金字塔，再采用RoIAlign根据用户指令中的区域引用在每个级别上提取区域特征，然后融合为一个单一的嵌入，直接作为区域表示

