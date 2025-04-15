---
title: GRPO
description: GRPO
authors:
  - Asthestarsfalll
tags:
  - RL
hide_table_of_contents: false
---

## Workflow

1. 对于每个 prompt，采样一组回答 $\mathcal{G}$；
2. 计算每个回答的奖励；
3. 对每个回答计算组内标准化的 advatange，$A_i = \frac{R_\phi(r_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}$。

如此一来便去除了之前需要在每一步生成上计算价值函数。

## Objective

与 PPO 类似，GRPO 仍然使用了剪切的代理损失（clipped surrogate loss）以及 KL 散度惩罚（KL penalty）。这里没有使用熵奖励项，因为基于分组的采样本身已经鼓励了探索性。剪切的代理损失与 PPO 中使用的完全相同，但为了完整性，这里再说明一下

$$
\mathcal{L}_{\text{clip}}(\theta) =\frac{1}{N} \sum_{i=1}^N \left( \min\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)} A_i, \ \text{clip}\left( \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)}, 1-\epsilon, 1+\epsilon \right) A_i \right) \right)
$$

总的 loss 为

$$
\begin{align} \mathcal{L}_{\text{GRPO}}(\theta) &= \underbrace{\mathcal{L}_{\text{clip}}(\theta)}_{\text{Maximise reward}} - \underbrace{w_1\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{orig}})}_{\text{Penalise KL divergence}} \end{align}
$$
