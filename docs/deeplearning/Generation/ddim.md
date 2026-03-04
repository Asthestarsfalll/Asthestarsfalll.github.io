## 前向推导

DDPM 的前向过程是一个**严格的马尔可夫链**：

前向过程的每一步 $x_t$ 仅由前一步 $x_{t-1}$ 生成，满足：

$$q(x_t|x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{\alpha_t}x_{t-1},\ (1-\alpha_t)\mathbf{I}\right)$$

其中 $\alpha_t=1-\beta_t$，$\beta_t$ 是预设的噪声调度系数，且 $\beta_1\ll\beta_T$。

对任意时间步 $t$，通过链式展开马尔可夫链的联合分布 $q(x_{1:T}|x_0)=\prod_{s=1}^T q(x_s|x_{s-1})$，可以推导出 $x_t$ 与原始图像 $x_0$ 的边缘分布：

$$q(x_t|x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar{\alpha}_t}x_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right)$$

其中 $\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$ 是累积系数，对应的加噪公式为：

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\cdot\epsilon_t,\quad \epsilon_t\sim\mathcal{N}(0,\mathbf{I})$$

**马尔可夫过程的核心约束**

DDPM 的前向过程满足 **无后效性**：$x_t$ 的分布仅依赖 $x_{t-1}$，与更早的 $x_{0:t-2}$ 无关；且联合分布 $q(x_{1:T}|x_0)$ 完全由相邻条件分布的乘积决定。

DDIM 的核心洞察是：**扩散模型的训练目标仅依赖边缘分布 $q(x_t|x_0)$，与联合分布 $q(x_{1:T}|x_0)$ 的形式无关**。
具体来说：
- DDPM 的损失函数是 **噪声预测损失**：$\mathcal{L}=\mathbb{E}_{x_0,\epsilon_t,t}\left[\|\epsilon_t-\epsilon_\theta(x_t,t)\|^2\right]$，该损失仅涉及 $x_0$ 和 $x_t$ 的关系，即边缘分布 $q(x_t|x_0)$。
- 因此，只要保证 **边缘分布 $q(x_t|x_0)$ 与 DDPM 一致**，无论联合分布 $q(x_{1:T}|x_0)$ 是否满足马尔可夫性，训练好的噪声预测网络 $\epsilon_\theta$ 都可以直接复用。

在保留边缘分布的前提下，我们可以构造**任意形式的联合分布**，从而得到非马尔可夫的前向过程，具体推导如下：

### 前向过程

DDIM 不使用连续的时间步 $1,2,…,T$，而是定义一个 **稀疏的跳步序列**：

$$\tau = \{\tau_S, \tau_{S-1}, …, \tau_1\},\quad \tau_S=T,\ \tau_1=1,\ S\ll T$$

例如 $T=1000, S=50$ 时，时间步序列可以是 $\tau=\{1000,980,960,…,20,0\}$。

对跳步序列中的任意两个相邻时间步 $\tau_i$ 和 $\tau_{i-1}$（$\tau_i>\tau_{i-1}$），我们直接定义 $x_{\tau_{i-1}}$ 与 $x_{\tau_i}$ 的关系，**无需经过中间时间步**。
根据边缘分布一致性约束，我们有：

$$
\begin{cases}
x_{\tau_i} = \sqrt{\bar{\alpha}_{\tau_i}}x_0 + \sqrt{1-\bar{\alpha}_{\tau_i}}\cdot\epsilon_{\tau_i} \\
x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}x_0 + \sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\cdot\epsilon_{\tau_{i-1}}
\end{cases}
$$

消去 $x_0$，可以得到 $x_{\tau_{i-1}}$ 与 $x_{\tau_i}$ 的直接映射关系：

$$
x_{\tau_{i-1}} = \sqrt{\frac{\bar{\alpha}_{\tau_{i-1}}}{\bar{\alpha}_{\tau_i}}}x_{\tau_i} + \sqrt{\bar{\alpha}_{\tau_{i-1}}(1-\bar{\alpha}_{\tau_i}) - \bar{\alpha}_{\tau_i}(1-\bar{\alpha}_{\tau_{i-1}})}\cdot\epsilon_{\tau_i} + \sqrt{1-\bar{\alpha}_{\tau_{i-1}}}\cdot(\epsilon_{\tau_{i-1}}-\sqrt{\frac{1-\bar{\alpha}_{\tau_i}}{1-\bar{\alpha}_{\tau_{i-1}}}}\epsilon_{\tau_i})
$$

:::tip
本质上是由于DDPM实际上是一个受限的马尔可夫过程，展开之后体现了非马尔可夫的性质。
:::

## 采样过程

与DDPM的随机采样不同，DDIM支持**确定性采样**（无随机噪声注入）和**半随机采样**（可控随机性）。

反向采样核心公式：给定$x_t$和噪声预测网络输出$\epsilon_\theta(x_t,t)$，直接推导$x_{t-\tau}$（$\tau$为步长间隔，可大于1）的表达式：

$$x_{t-\tau}=\sqrt{\bar{\alpha}_{t-\tau}}x_0+\sqrt{1-\bar{\alpha}_{t-\tau}}\epsilon_\theta(x_t,t)$$

 代入$x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$，可得最终迭代公式： $$
      \begin{align}
      x_{t-\tau}&=\sqrt{\frac{\bar{\alpha}_{t-\tau}}{\bar{\alpha}_t}}x_t + \sqrt{\bar{\alpha}_{t-\tau}(1-\frac{\bar{\alpha}_{t-\tau}}{\bar{\alpha}_t})}\cdot(-\epsilon_\theta(x_t,t)) \\
      &\quad + \sqrt{1-\bar{\alpha}_{t-\tau}-\frac{\bar{\alpha}_{t-\tau}}{\bar{\alpha}_t}(1-\bar{\alpha}_t)}\cdot\eta\epsilon_\theta(x_t,t)
      \end{align}
      $$

  其中$\eta$为**随机性系数**：
- $\eta=0$：**确定性采样**，无额外噪声注入，路径固定；
- $\eta=1$：等价于DDPM采样（完全随机）；
- $0<\eta<1$：半随机采样，平衡效率与多样性。

以**任意步长采样**（步长间隔$\tau=T/S$，$S$为目标采样步数）为例，流程如下：
1.  **参数初始化**
    1. 设定总扩散步数$T$、目标采样步数$S$，计算步长间隔$\tau=T/S$；
    2. 确定时间步序列$\{t_0=T,t_1=T-\tau,\dots,t_S=0\}$；
    3. 采样初始纯噪声$x_{t_0}\sim\mathcal{N}(0,\mathbf{I})$；
    4. 设定随机性系数$\eta$（通常取0或0.5）。
2.  **迭代逆向去噪**（从$i=0$到$i=S-1$）
    1.  取当前时间步$t=t_i$，下一时间步$s=t_{i+1}=t_i-\tau$；
    2.  将$x_t$和时间步$t$输入噪声预测网络，得到预测噪声$\epsilon=\epsilon_\theta(x_t,t)$；
    3.  计算系数：

        $$
        \begin{align}
        \tilde{\alpha}_t&=\bar{\alpha}_t,\quad\tilde{\alpha}_s=\bar{\alpha}_s \\
        \sigma_{t,s}&=\eta\cdot\sqrt{\frac{(1-\tilde{\alpha}_t/\tilde{\alpha}_s)\tilde{\alpha}_s}{1-\tilde{\alpha}_t}} \\
        \tilde{\mu}&=\sqrt{\frac{\tilde{\alpha}_s}{\tilde{\alpha}_t}}x_t - \sqrt{\tilde{\alpha}_s(1-\frac{\tilde{\alpha}_s}{\tilde{\alpha}_t})}\epsilon
        \end{align}
        $$

    4.  采样随机噪声$z\sim\mathcal{N}(0,\mathbf{I})$（若$\eta=0$则$z=0$）；
    5.  更新样本：$x_s=\tilde{\mu}+\sigma_{t,s}\cdot z$；
    6.  令$t=s$，进入下一轮迭代。
3.  **输出结果**
    迭代至$t=0$时，得到$x_0$，即为最终生成样本。
