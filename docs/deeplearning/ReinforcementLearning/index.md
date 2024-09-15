---
title: ReinforcementDeepLearning
hide_table_of_contents: true
sidebar_position: 1
---

旨在介绍 LLM 中所使用的 PPO，目前 LLM 的训练可以分为两个部分：

1. Pre-training，使用大量的 web 数据进行 next token prediction，属于无监督训练；
2. Post-training，旨在提升 LLM 的推理能力等，包括 SFT 有监督微调，使用高质量的数据，包括指令微调、COT 等方式，当没有太多高质量数据时，额外使用 RLHR（**_Reinforcement Learning from Human Feedback_**）训练一个 reward model 来对齐人类表现。

:::note

DeepSeek 后训练的高效之一就是其直接跳过了 SFT 直接进行 RL，除了提高计算效率，还消除了人工微调数据的影响，让模型有自我进化的能力（**Open-ended learning**）。

DeepSeek 使用 GRPO 来取代 PPO，将内存和计算需求减少约 50%

:::

## RLHR

RLHR 流程可以分为四个阶段：

1. 对一个 prompt 采样模型的多个回答；
2. 人工对回答进行排序；
3. 训练一个 **reward model** 来预测人类的偏好和排名；
4. 使用强化学习微调模型来最大化奖励模型的分数。

## Reward Model

实际上我们很难对所有模型回答进行排序，一种节省成本的方法是让注释者对 LLM 输出的一小部分进行评分，然后训练一个模型来预测人类的偏好。

最小化损失函数

$$
-\log \sigma\big(R(p,r_{i})-R(p,r_{j})\big)
$$

其中人类对 $r_{i}$ 的偏好好于 $r_{j}$

该式来源于**Bradley-Terry model**
