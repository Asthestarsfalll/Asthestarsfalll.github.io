## 算法背景

GRPO（Grouped Relative Policy Optimization）虽解决了传统RL的价值函数依赖问题，但存在三个致命缺陷，尤其在MoE模型上表现突出：
- **优化粒度不匹配**：奖励是序列级（如整段回答的人类偏好评分），但GRPO采用token级**重要性权重更新**，导致梯度噪声大、训练不稳定；
- **MoE适配差**：MoE模型的专家路由动态变化，token级似然波动剧烈，需“路由重放（Routing Replay）”等复杂技巧才能收敛，成本极高；
- **样本效率低**：token级裁剪易丢弃有效学习信号，长序列生成时方差进一步放大。

GSPO的核心创新是**将优化粒度从token级提升到序列级**，让优化目标与奖励粒度完全对齐：
1. 基于**序列似然**定义重要性比率，避免token级波动；
2. 采用组内相对优势估计，无需独立价值网络；
3. 序列级裁剪，大幅降低训练方差，适配MoE模型无需额外技巧。

---

## 核心定义、公式与理论基础

### 1. 基础符号

| 符号 | 含义 |
|------|------|
| $q$ | 输入查询（prompt） |
| $o_i$ | 针对$q$生成的第$i$条候选序列（回答），$i \in \{1,2,…,N\}$（组内$N$条序列） |
| $\pi_\theta(o_i \mid q)$ | 策略网络（LLM）生成序列$o_i$的**序列级似然**（自回归生成的联合概率） |
| $r(o_i)$ | 序列$o_i$的全局奖励（如人类偏好、任务指标） |
| $\mathcal{G} = \{o_1,o_2,…,o_N\}$ | 同一查询$q$对应的候选序列组（GSPO的“Group”核心） |
| $s_i(\theta)$ | 序列$o_i$的重要性比率（序列级，区别于GRPO的token级） |
| $\epsilon$ | 序列级裁剪系数（控制策略更新幅度） |
| $\hat{A}(o_i)$ | 组内相对优势（替代传统价值函数，GSPO与GRPO共享） |

## 核心公式

GSPO沿用GRPO的相对优势思想，无需价值网络，直接通过组内奖励对比计算优势：

$$\hat{A}(o_i) = r(o_i) - \frac{1}{N}\sum_{j=1}^N r(o_j)$$

该优势表示序列$o_i$相对于组内平均奖励的优劣，正优势强化，负优势抑制。

区别于GRPO的token级比率，GSPO定义**序列级重要性比率**（策略$\theta$相对于旧策略$\theta_{\text{old}}$的似然比）：

$$s_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$$

该比率是整段序列的联合概率比，避免token级波动，适配MoE模型。

目标是最大化组内序列的加权优势，通过裁剪防止策略更新幅度过大：

$$J_{\text{GSPO}}(\theta) = \mathbb{E}_{\mathcal{G}} \left[ \sum_{i=1}^N \min\left( s_i(\theta) \cdot \hat{A}(o_i), \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}(o_i) \right) \right]$$

- 当$\hat{A}(o_i) > 0$：增强高奖励序列的生成概率；
- 当$\hat{A}(o_i) < 0$：削弱低奖励序列的生成概率；
- 序列级裁剪确保更新在信任域内，避免模式崩溃。

长序列的似然乘积易趋于0/无穷，GSPO可添加长度归一化项，确保公平性：

$$s_i(\theta) = \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \right)^{\frac{1}{L_i}}$$

其中$L_i$是序列$o_i$的长度，几何平均平滑长序列的似然波动。

---

## 伪代码

```python
import torch
import torch.nn.functional as F

# ======================== 1. 优势计算 (GAE) ========================
def compute_gae_advantages(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # 拼接最后一步的value，方便反向计算
    values = torch.cat([values, last_value], dim=1)
    
    # 反向计算优势
    advantage_t = torch.zeros(batch_size, device=rewards.device)
    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * values[:, t+1] * (1 - dones[:, t]) - values[:, t]
        advantage_t = delta + gamma * lam * (1 - dones[:, t]) * advantage_t
        advantages[:, t] = advantage_t
        returns[:, t] = advantage_t + values[:, t]
    
    # 优势标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns

# ======================== 2. 序列级别似然计算（论文版，无采样） ========================
def compute_sequence_log_prob_paper(logits, actions, mask=None):
    # 1. 计算所有动作的对数概率（softmax + log，无采样）
    log_probs_all = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, num_actions]
    
    # 2. 取出实际动作对应的对数概率（核心：直接索引，无采样）
    # 先扩展actions维度，方便gather
    actions_expanded = actions.unsqueeze(-1)  # [batch_size, seq_len, 1]
    step_log_probs = torch.gather(log_probs_all, dim=-1, index=actions_expanded).squeeze(-1)  # [batch_size, seq_len]
    
    # 3. 应用mask（忽略填充部分）
    if mask is not None:
        step_log_probs = step_log_probs * mask
    
    # 4. 序列级别似然：所有时间步对数概率求和
    seq_log_prob = step_log_probs.sum(dim=1)  # [batch_size]
    
    return seq_log_prob, step_log_probs

# ======================== 3. GSPO损失计算（简化版） ========================
def compute_gspo_loss_simple(old_step_log_probs, new_step_log_probs, advantages, old_values, new_values, clip_epsilon=0.2):
    """
    简化版GSPO损失（序列级别的PPO损失）
    输入维度均为: [batch_size, seq_len]
    """
    # 策略损失（PPO裁剪）
    log_ratio = new_step_log_probs - old_step_log_probs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 价值损失（裁剪的MSE）
    value_clipped = old_values + torch.clamp(new_values - old_values, -clip_epsilon, clip_epsilon)
    v_loss1 = F.mse_loss(new_values, advantages + old_values, reduction='none')
    v_loss2 = F.mse_loss(value_clipped, advantages + old_values, reduction='none')
    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
    
    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss

# ======================== 测试示例 ========================
if __name__ == "__main__":
    # 模拟输入
    batch_size, seq_len, num_actions = 4, 10, 5
    logits = torch.randn(batch_size, seq_len, num_actions)  # 模型输出logits
    actions = torch.randint(0, num_actions, (batch_size, seq_len))  # 实际动作
    mask = torch.ones(batch_size, seq_len)  # 掩码
    
    # 1. 计算序列似然（论文版，无采样）
    seq_log_prob, step_log_probs = compute_sequence_log_prob_paper(logits, actions, mask)
    print("序列级别对数似然:", seq_log_prob)
    print("单步对数似然形状:", step_log_probs.shape)
    
    # 2. 模拟优势计算
    rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    dones = torch.zeros(batch_size, seq_len)
    last_value = torch.randn(batch_size, 1)
    advantages, returns = compute_gae_advantages(rewards, values, dones, last_value)
    
    # 3. 模拟损失计算
    old_step_log_probs = step_log_probs  # 旧策略似然
    new_step_log_probs = step_log_probs * 0.98  # 新策略似然（模拟更新）
    old_values = values
    new_values = values * 1.02
    total_loss, p_loss, v_loss = compute_gspo_loss_simple(old_step_log_probs, new_step_log_probs, advantages, old_values, new_values)
    print("总损失:", total_loss.item())
```
