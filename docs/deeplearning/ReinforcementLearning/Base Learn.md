:::info 过估计
Q-learning在 OOD 数据上经常出现过估计的情况，即给出过分高的Q值。

核心原因是 Max 操作引入的系统性正偏差

Q-learning的TD目标是：

$$y = r + \gamma \max_{a'} Q(s', a')$$

假设Q值估计中存在无偏噪声（即对每个动作$a'$的误差$\epsilon_{a'}$满足$\mathbb{E}[\epsilon_{a'}]=0$），但误差有方差（训练中的采样噪声、函数逼近误差、网络随机初始化等）。
那么真实期望满足：

$$\mathbb{E}\Bigl[\max_{a'} \bigl(Q^*(s', a') + \epsilon_{a'}\bigr)\Bigr] \;\; > \;\; \max_{a'} Q^*(s', a')$$

这是统计学上经典的“最大化偏差”（maximization bias）：多个随机变量的最大值的期望，总是大于它们各自期望的最大值。正误差会被 max 选中，而负误差会被“抛弃”，导致整体目标$y$系统性地偏高。

- ID 数据：如果样本足够多、覆盖充分，网络对每个动作的估计方差较小，偏差可以被后续更新部分抵消（尤其在tabular Q-learning理论收敛情况下：表格形式的Q-learning在无限长时间下，以概率1收敛到最优动作价值函数）。
- OOD数据：状态/动作严重偏离训练分布，网络处于高不确定性区域（extrapolation），每个$Q(s', a')$的误差$\epsilon_{a'}$方差极大。此时max操作把“最乐观（最大）的那个噪声”挑出来，偏差被几何级数放大（因为是$\gamma$倍的未来值，$\gamma<1$但仍会累积）。
:::

:::tip
稍微总结一下RL的发展历程
- 所有RL基本都基于Bellman方程，最开始就是动态规划，分为策略迭代和价值迭代，前者算所有价值再更新，后者一边算价值一边贪心
- 之后有一些Actor critic等自适应评价方法、策略梯度以及TD error等，这些都是model based的，情况简单方便建模，后面发展出了基于的policy的方法
- 基于表格的Q learning出现的时间更晚，因为其需要TD、动作价值Q(s,a)、off policy同时成熟，在此之前都是V(s)和on policy，Q learning更难一点。
- 现在就是基于深度网络的方法
	- 基于AC的on-policy方法：PPO、A2C，critic一般使用value function即可
	- 基于AC的off-policy方法：SAC、TD3等，critic一般使用Q function
	- 纯Q learning：DQN、CQL，无policy
	- 基于Q learing+critic的方法：XQL，无policy，使用Value function替换max Q(a', s')
- 离线RL方法：CQL、BCQ、TD3+BC、ILQL
:::

:::info 为什么LLM RL中如何使用Q learning
已经证明Q learning在高维连续、离散空间学习不稳定。

LLM的动作空间是整个词表，维度非常大，基于最大化的 Q learning会迅速坍缩。自回归也不是马尔可夫的。
当然传统的Q learning需要取整个动作空间的最大，但是LLM中已经SFT了，实际上动作空间应该是及其有限的，所以理论上LLM RL是可以做off policy的。

至于DPO等这种偏好对齐的离线方法就不太一样了，还有类似Decision Transformer或者classifier guidance free的曲线救国RL方式。

LLM RL Off policy 也有一些相关探索，有空再看。
:::

:::info VLA RL Q learning
1. action是低维连续
2. 决策是马尔可夫的，单步贪心不会坍缩
3. 状态空间平滑，Q函数好泛化，相近的状态Q值也相近。
:::

## DQN

:::info
这些都是Value Based的方法，主要用于离散控制，**只学习状态或动作的价值，决策时隐式地选择价值最高的动作**。

模型输出的不是单独的action，而是所有可能action的score.

探索方式一般为ϵ-greedy，以1-ϵ的概率选择最优动作，ϵ的概率随机均匀选择，ϵ参数需要精细的schedule

Boltzmann Exploration （Softmax探索），Q值越高，则采样概率越大

UCB（Upper Condense Bound），额外考虑某个动作尝试了多少次，不是像greedy一样随机尝试

深度特化的NoisyNet，增加一个可学习的噪声，自主决定探索

:::

首次结合深度学习和强化学习，实现从原始高维输入学习控制策略。

经验回放
- rl数据往往具有时间关联性，直接训练可能会过拟合
- 使用经验回放满足神经网络i.i.d假设
- 提高样本效率，同一样本被多次训练
- 平滑数据分布，混合不同时期数据

目标网络
- 标准Q学习目标依赖于正在更新的参数，目标为及时奖励+折扣未来q值，后者用到了参数本身
- 使用一个在线模型每步都进行更新，计算当前q值
- 使用一个目标网络计算目标q值
- 每隔一定步数，从在线网络更新目标网络（硬更新）
- 每步使用ema平滑更新（软更新）

## Double DQN

Q learning目标的 **max** 操作会导致**系统性高估**：

- 神经网络输出的 Q 值总是带有**噪声（estimation error）**。
- **max** 操作会倾向于挑选那些**噪声恰好为正（被高估）的动作**。
- 结果：即使某个动作的真实 Q 值不高，但只要它的噪声大，就会被选为 max，导致目标 y 被持续高估。
- 高估会像滚雪球一样传播（因为下一个目标又基于这个高估的 Q 值），最终让策略偏向“看起来好但实际很差”的动作。

:::info
这种目标函数中存在要优化的参数的情况叫做自举（bootstrapped）
:::

**Double Q-Learning**（表格形式）早就证明：**把“选择动作”和“评估动作价值”用两个不同的 Q 表分开**，就能大幅减轻过估计。Double DQN 把这个思想直接搬到深度强化学习里。
- 在线网络，用于动作选择，选择下一个状态的最优动作
- 目标网络，用于评估，评估选出来的动作

## Rainbow DQN

集大成制作，融合了后面的相关技术
 - Double DQN
 - Prioritized Experience Replay（优先经验回放，PER），根据td error回放
 - Dueling Network Architecture（对偶网络架构），将Q分解为状态价值和动作优势
 - Multi-step Learning（多步学习，n-step returns），使用多步累计回报代替TD Error
 - Distributional Q-learning（分布 Q 学习，C51），把回归转换为分类问题
 - Noisy Nets（噪声网络）

## DDPG (Deep Deterministic Policy Gradient)

DDPG专门为连续动作设计，采用Actor Critic的确定性策略，有四个网络构成
- 目标actor，计算下一个状态的目标动作
- 在线actor，计算当前状态的动作
- 目标critic，评估目标动作的价值
- 在线critic，评估当前状态的价值

:::tip
critic使用两个网络类似与double dqn，因为其目标函数本身是自举的，TD(0)

actor的目标为最大化value值，和后续的策略梯度不太一样。

目标actor是给critic生产未来动作的。
:::

**训练**
- 不同于Double DQN直接更新，DDPG使用ema软更新
- 通过Ornstein-Uhlenbeck 过程在动作上添加噪声实现探索

:::tip
感性理解为带弹簧拉力的随机游走

现实中的一种随机现象是，虽然数值会波动，但总是倾向于回到一个长期平均值

$$dX_t = \theta (\mu - X_t) dt + \sigma dW_t$$

其中各参数的含义如下：
- $\theta$回归速度，决定了系统偏离均值后，多快能“拉回来”。$\theta$ 越大，回复力越强。
- $\mu$长期均值过程最终趋向的平衡中心。
- $\sigma$波动率系统的随机噪声强度。
- $dW_t$维纳过程标准布朗运动带来的随机干扰。

在强化学习中，常见的探索方式是添加高斯白噪声。但在**连续控制任务**中，OU 过程具有独特的优势：

- **时间相关性（Temporal Correlation）**： 这是 OU 过程的核心。白噪声在每个时间步是完全独立随机的，这会导致动作高频震荡（比如机器人关节一秒钟内来回抖动几百次），这种噪声往往会被物理系统抵消，无法实现有效的位移。
- **惯性探索**： OU 过程的噪声在短时间内倾向于维持同一方向。如果上一个时刻噪声是正的，下一时刻大概率还是正的。这模拟了具有“惯性”的物理运动，使智能体能够沿着某个方向持续探索一段距离，从而更容易触发环境的奖励反馈。

加噪主要存在buffer中，优化是使用实际action.
:::

## TD3

依然是解决DDPG过估计的问题。
- 使用两个Q
- 评估时添加高斯噪声
- Actor延迟更新，Critic需要更新更频繁

## SAC

基于最大熵强化学习，最大化累积奖励和熵

$$\text{Objective} = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t))]$$

 熵衡量的是随机性。熵越高，动作越随机。

现代 SAC能够自动调节 $\alpha$。当发现某个地方没去过时，增加 $\alpha$ 去探索；当环境已经摸透时，降低 $\alpha$ 稳定下来。

相关改进有分布式q、TQC（分位数回归）

:::tip Soft Actor Critic
这里的 Soft 说的是在最大化期望累积奖励的基础上额外添加了最大化熵，这是一种 soft 目标，不严格追求最优确定性动作。

同时相当于添加了一个正则化，让动作分布更均匀，并且天然鼓励探索。

另一方面是迭代的soft，选取动作不是直接选择最大，而是类似与softmax的概率形式。通过KL三度让参数尽量靠近该分布。
:::

```python
class SAC:
    def __init__(self, state_dim, action_dim, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )
        
        self.alpha = 0.2          # 温度参数（可学习或固定）
        self.gamma = 0.99
        self.tau = 0.005          # soft update 系数（EMA）
        self.device = device

    # ------------------- Soft Update (EMA) -------------------
    def soft_update(self):
        """目标网络的 soft update（Polyak averaging）"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------- Actor 更新（核心：softmax-like） -------------------
    def update_actor(self, states):
        actions, log_probs = self.actor.sample(states)
        
        # 当前策略下的 Q 值（用较小的那个，Double Q 防止高估）
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Actor loss = E[ α * logπ(a|s) - Q(s,a) ]
        # 这等价于最小化 KL( π || softmax(Q/α) )
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()

    # ------------------- Critic 更新（Soft Bellman Backup） -------------------
    def update_critic(self, states, actions, rewards, next_states, dones, replay_buffer=None):
        with torch.no_grad():
            # 下一个动作来自当前 actor（off-policy）
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Soft Q target：带熵的 Bellman backup
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # soft Bellman equation
            target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)
        
        # 当前 Q 值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # MSE loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()

    # ------------------- 完整训练步骤 -------------------
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch   # 从 Replay Buffer 采样
        
        # 1. 更新 Critic（Soft Q）
        critic_loss = self.update_critic(states, actions, rewards, next_states, dones)
        
        # 2. 更新 Actor（朝着 softmax(Q) 方向更新）
        actor_loss = self.update_actor(states)
        
        # 3. Soft Update 目标网络（EMA）
        self.soft_update()
        
        return {"critic_loss": critic_loss, "actor_loss": actor_loss}
```

## CQL

离线RL，之前的Q learning对于没见过的数据通过会过分高估，CQL在标准Q learning基础上添加正则项
- 压低所有可能动作的Q
- 拉高数据中已有动作的Q

CQL 的损失函数可以简化理解为：$$\text{Loss}_{CQL} = \text{Standard Q Loss} + \alpha \cdot \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \text{data}}[Q(s, a)] \right]$$

- 前一项（log-sum-exp）：在所有动作空间内寻找 Q 值的“峰值”并把它按下去。
- 后一项（Expectation over data）：确保数据集里出现的动作不被误伤得太深。
- $\alpha$（系数）：控制“保守程度”。$\alpha$ 越大，智能体越不敢尝试未知的动作。

论文证明了CQL学习的Q是真实Q的下界，即一定小于等于真实Q。

副作用就是悲观偏差 (Pessimism Bias)，过于胆小，数据集内部的Q也不会很高。

## IQL (Implicit Q-Learning)

IQL 完全不需要对策略之外的动作进行采样，只用了两个简单的数学工具：预期回归 (Expectile Regression) 和 优势权重 (Advantage Weighting)。

通过学习一个状态价值函数 $V(s)$，使其逼近数据集内 Q 值的高分位点。
- 普通均值回归： 学习的是 $s$ 状态下所有动作的平均表现。
- 预期回归 (Expectile Regression)： 引入一个参数 $\tau$ (通常取 0.7-0.9)。它会让 $V(s)$ 更多地去拟合那些表现较好的样本。
- 直观理解： 它在不看 OOD 动作的前提下，通过对数据集内的样本“选优”，隐式地知道了“在这个状态下，最好的表现大概是多少”。

预期回归损失为：

$$L_2^\tau (u) = \begin{cases} \tau \cdot u^2 & \text{if } u > 0 \\ (1 - \tau) \cdot u^2 & \text{if } u \leq 0 \end{cases}$$

- $u = Q(s, a) - V(s)$：代表数据集里的动作价值与当前状态价值的偏差。
- $\tau \in [0.5, 1)$：这是一个超参数（IQL 论文中通常取 0.7 或 0.9）。

不涉及 OOD 动作的 Q 学习IQL 更新 Q 函数时，目标值不再使用 $max_{a'} Q(s', a')$，而是直接使用学到的 $V(s')$：

$$y = r + \gamma V(s')$$

这样，Q 函数的更新过程完全不涉及对任何新动作的查询，从根源上杜绝了 Q 值爆炸的可能性。

最后采用一种类似“模仿学习”的方式，观察数据集里的动作 $a$。如果这个动作的 $Q(s, a)$ 明显高于平均水平 $V(s)$（即优势 $A > 0$），就让 Actor 狠狠地学它。如果表现平平，就轻微学习。

$$\pi(a|s) \propto \exp(\beta(Q(s, a) - V(s)))$$

## XQL

基于极值理论的RL，无显式建模策略（无actor）或熵，直接估计最大熵RL中的软最优值函数，支持在线和离线

传统深度RL（SAC、TD3、DQN）在连续空间中有两大问题
- 最大Q值难以估计，连续动作无法穷举
- OOD数据Q值过估计

XQL假设bellman误差服从Gumbel极值分布
- **软最优值 $V^*(s) = \log\mathbb{E}_\beta[e^{\beta Q(s,a)}] \approx \text{LogSumExp}_a Q(s,a)$**（$\beta$为温度系数），就是使用critic来估计一个Q最大值
- 无需对动作采样，直接用**Gumbel回归（极大似然）**拟合该分布。

标准Bellman目标：$y = r + \gamma V^*(s')$

XQL的核心损失（基于Gumbel极大似然）：

$$\mathcal{L}_{\text{XQL}} = \mathbb{E}\left[ e^{\beta (y - Q(s,a))} \right]$$

```python
# 网络：只有 Q 和 V，没有 Policy 网络！
q_net = QNetwork()    # 输入 s, a → 输出 Q(s,a)
v_net = VNetwork()    # 输入 s    → 输出 V(s)
v_target = VNetwork() # 目标 V 网络

# ---------------------------
# 1. 构造 XQL 的 Q-target（最关键！）
# ---------------------------
# 从回放池取数据：s, a, r, s_next, done
Q = q_net(s, a)                      # 当前 Q(s,a)
V_next = v_target(s_next).detach()   # 下一状态价值 V(s')
Q_target = r + gamma * V_next * (1 - done)  # 【XQL核心】没有动作！

# 训练 Q 网络
q_loss = F.mse_loss(Q, Q_target)

# ---------------------------
# 2. 训练 V 网络（Gumbel/LINEX 损失）
# ---------------------------
Q_sa = q_net(s, a).detach()
V_s = v_net(s)

# XQL 核心损失：直接拟合软最大值，不需要采样动作！
beta = 0.1
adv = (Q_sa - V_s) / beta
v_loss = (exp(adv) - adv - 1).mean()

# ---------------------------
# 3. 推理动作：直接从 Q 导出，不需要 Policy
# ---------------------------
def select_action(s):
    # 直接用 Gumbel-softmax / 采样，不需要 max
    a = sample_from_Q(q_net, s)
    return a
```
