Decoupled Clip and Dynamic Sampling Policy Optimization（解耦裁剪与动态采样策略优化）

---

## 背景

LLM的RLHF面临三大核心痛点：
1. 动作空间高维离散（词汇表数万级），传统对称裁剪抑制探索，易熵崩溃；
2. 奖励同质化导致梯度消失，样本效率低；
3. 长序列梯度稀释、截断样本奖励噪声大，训练不稳定。

## 核心改进与公式

### Clip-Higher（解耦非对称裁剪）

- **改进**：突破PPO/GRPO的对称裁剪（如±0.2），对正优势token放宽裁剪上限，鼓励探索低概率token，避免熵崩溃。
- **公式**：
  1.  概率比：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
  2.  非对称裁剪：$r_t^{\text{clip}} = \begin{cases}\min(r_t, 1 + \epsilon_{\text{high}}) & \text{if } A_t > 0 \\\max(r_t, 1 -\epsilon_{\text{low}}) & \text{if } A_t \leq 0  \end{cases}$，其中$\epsilon_{\text{high}} > \epsilon_{\text{low}}$
  3.  裁剪损失：$\mathcal{L}_{\text{clip}} = -\mathbb{E}_t\left[\min(r_t A_t, r_t^{\text{clip}} A_t)\right]$

### Dynamic Sampling（动态采样）

- **改进**：过滤组内奖励标准差为0的样本（全对/全错），确保批次梯度有效，提升训练效率。
- **公式**：
  1.  组内奖励标准化优势：$\hat{A}_{i,t} = \frac{R_i - \text{mean}(R_j)}{\text{std}(R_j)}$，$j=1..G$（$G$为组内样本数）
  2.  动态采样规则：仅保留$\text{std}(R_j) \neq 0$的组，否则重采样至批次填满

### Token-Level Policy Gradient Loss（token级策略梯度损失）

- **改进**：GRPO句子级归一化导致长序列梯度稀释，DAPO按token归一化，解决梯度偏置。
- **公式**：
  1.  Token级优势：$\hat{A}_{i,t}^{\text{token}} = \frac{A_{i,t}}{\sqrt{\frac{1}{T}\sum_{t=1}^T A_{i,t}^2 + \delta}}$（$T$为序列长度，$\delta$为微小常数）
  2.  Token级损失：$\mathcal{L}_{\text{token-pg}} = -\mathbb{E}_{i,t}\left[r_t^{\text{clip}} \cdot \hat{A}_{i,t}^{\text{token}}\right]$

:::info
实际上就相当于加了一个长度的权重。GRPO在轨迹内先求平均，然后再轨迹间求平均，得到最后loss。而DAPO是直接在token级别求平均，相当于每个token的贡献应该是一致的。

GRPO是轨迹级别，长度越长，每个token的贡献就被稀释了。
:::

### Overlong Reward Shaping（超长奖励塑形）

- **改进**：惩罚超阈值长度的回复，减少奖励噪声，稳定训练。
- **公式**：
  1.  塑形奖励：$R_{\text{shaped}} = R - \lambda \cdot \max(0, L - L_{\text{max}})$，其中$\lambda$为惩罚系数，$L$为生成长度，$L_{\text{max}}$为阈值
