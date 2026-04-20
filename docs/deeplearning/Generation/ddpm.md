
Ref: https://zhuanlan.zhihu.com/p/12591930520

## 扩散过程

正向过程施加小范围的噪声

$$
x_{t+1}:=x_t+n_t, n\sim N(0, \sigma^2)
$$

### DDMP 采样器的目标

DDMP（去噪扩散模型）的核心思想是**逆向操作**：

- 正向过程：从真实数据 $x_0$​ 出发，一步步加高斯噪声，得到 $x_1​, x_2​,…,x_t$​。
- 逆向过程：从加了很多噪声的 $x_t$​ 出发，一步步 “去噪”，最终还原出真实数据 $x_0$​。

理想的 DDMP 采样器，就是在每一步 $t$，根据当前的含噪数据 $x_t=z$​，输出一个能生成前一步数据 $x_{t−1}$​ 的条件分布采样器：

$$p(x_{t−1}​∣x_t​=z)$$

:::info
这个条件分布是逆向过程的核心。

如果直接去学习这个条件分布 $p(x_{t−1}​∣x_t​)$，我们需要为每一个 $x_t$​ 都训练一个生成模型，这在计算上是巨大的负担。

**当噪声足够小时，逆向条件分布近似高斯** $p(x_{t−1​}∣x_t​)\approx N(x_{t-1};\mu,\sigma^2)$，这是整个DDPM的核心。

这意味着，我们不需要**学习整个复杂的分布**，只需要学习这个高斯分布的**均值 μ** 就够了，因为高斯分布的形状完全由均值和方差决定，而这里的方差 $σ^2$ 是我们在正向过程中已知的超参数。

因此问题变成了：给在时间 $t$ 和条件 $x_t$，学习$p_{x_{t-1}|x_t}$ 的均值即可。学习均值比较简单，可以用回归方法。
:::

:::info
对任意分布 $p(x_{t−1}​)$，如果我们通过加极小方差高斯噪声得到 $x_t$​，那么逆向的条件分布 $p(x_{t−1}​∣x_t​)$ 会**近似服从**高斯分布，并且其方差与正向过程的噪声方差几乎相同。

当$p(x_{t-1})$服从高斯分布时，逆向条件分布才会严格服从高斯分布。
:::

### 训练过程

一步一步正向加噪，公式为

$$
x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t} \epsilon \tag1
$$

其中

$$
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

### 噪声方差的确定

考虑一个加噪序列，时间均匀间隔为 $\Delta t=\frac{1}{T}$，每一步都加上一个均值为0，方差为$\sigma^2$的噪声，那个任意一步的总噪声和（仍是高斯分布）的方差就是各分布方差和$N \sigma^2$。

:::info
但我们的目标是噪声方差与时间步数$N$无关，因此将噪声添加一个归一化项$\sigma\sqrt \Delta t$，这样任意时间步总噪声就变为$\frac{t}{\Delta t}\sigma^2\Delta t=\sigma^2 t$，只与扩散时间（而不是扩散步数）有关。

这就是连续时间的随机微分方程（SDE），表面扩散速度和时间的平方根成正比，这也是布朗运动的典型特征。

**DDPM 的前向加噪过程，本质上就是在模拟一个受限的布朗运动。**
:::

### DDPM的成立

前文说到只有当噪声足够小时，DDPM的核心依据才成立，根据上文噪声方差的确定，即$\Delta t$足够小，因为DDPM也是使用了1000个时间步。

证明过程看论文。

:::info
直接把时间步 $\Delta t$ 缩小，虽然每一步引入的噪声会变小，但需要的总步数 $T=\frac{1}{\Delta t}$​ 会随之增加。如果每一步的误差没有以足够快的速度衰减，最终的**累计误差**可能会变得很大，因此原证明的严谨性需要补充。

可以使用用 KL 散度来量化 “单步噪声添加” 带来的分布误差，近似高斯分布与真实后验分布之间的 KL 散度误差是 $O(\sigma^4)$
每步噪声的方差 $σ^2=σ_q^2​\Delta t$，因此 $σ^4=σ_q^4​(\Delta t)^2$

为什么会存在这个 $\mathcal{O}(\Delta t^2)$ 的误差？

当我们用高斯分布近似 $q(x_{t-1}|x_t)$ 时，我们实际上只匹配了分布的前二阶矩（均值和方差）。

- **真实后验：** 包含更高阶的累积量（Cumulants），如偏度（Skewness）和峰度（Kurtosis）。
- **量级评估：** 这些高阶项的大小与得分函数（Score Function）的导数（即 Hessian 矩阵）有关。如果数据的分布非常复杂（曲率很大），那么高斯近似的误差就会显著增加。
因此有其他工作如improved ddpm，同时学习方差预测。

总误差为 $T⋅O(\sigma^4)=\frac 1 {\Delta t}​⋅O((\Delta t)^2)=O(\Delta t)$
当 $\Delta t→0$ 时，总误差也趋于 0，这就保证了整个离散化过程的收敛性，也证明了 DDPM 的正确性。
:::

## 直接预测x0

由式 $(1)$ 可知，$x_0$和噪声$\epsilon$是线性关系，因此可以直接预测。

并且由于每一步的噪声是同分布的（不独立， 正向过程是$i.i.d$），与其估计每一步的噪声，不如直接等效地估计所有先验噪声的平均，并且方差更小。

$$
E[x_{t-\Delta t}-x_t|x_t]=\frac{\Delta t}{t}E[(x_0-x_t)|x_t]
$$

## 采样过程

**均值计算**：利用预测噪声$\epsilon_\theta(x_t,t)$推导反向分布均值，公式如下：

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)$$

该式核心是从当前带噪样本$x_t$中减去网络预测的对应噪声分量，再通过$\alpha_t$相关系数校正，得到去噪后的均值估计。

**单步采样更新**：在均值基础上加入高斯噪声实现随机采样，公式为：

$$x_{t-1}=\mu_\theta(x_t,t)+\sigma_t\cdot z,z\sim\mathcal{N}(0,\mathbf{I})$$

其中$\sigma_t=\sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\cdot\beta_t}$，且当$t=1$时，$z=0$（避免最终样本引入额外噪声）。

## 噪声推导过程

### 符号定义

- $x_0 \sim q(x_0)$：真实数据分布
- $x_1,\dots,x_T$：隐变量，逐步加噪
- $T$：总扩散步数
- $\beta_t \in (0,1)$：第 $t$ 步噪声强度，通常递增
- $\alpha_t = 1-\beta_t$
- $\bar\alpha_t = \prod_{i=1}^t \alpha_i$
- $\epsilon_t \sim \mathcal{N}(0,I)$：标准高斯噪声

### 前向过程（加噪）：从 $x_0 \to x_T$

**一步加噪**

条件分布为**高斯**：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\ \sqrt{\alpha_t}x_{t-1},\ \beta_t I\bigr)$$

重参数，这z高斯分布上：

$$x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1}$$

### 递推合并：直接从 $x_0$ 得到 $x_t$（关键推导）

对递推式展开：

$$\begin{aligned}
x_1 &= \sqrt{\alpha_1}x_0 + \sqrt{\beta_1}\epsilon_0 \\
x_2 &= \sqrt{\alpha_2}x_1 + \sqrt{\beta_2}\epsilon_1
= \sqrt{\alpha_1\alpha_2}x_0 + \sqrt{\alpha_2\beta_1}\epsilon_0 + \sqrt{\beta_2}\epsilon_1 \\
&\vdots \\
x_t &= \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
\end{aligned}$$
其中 $\epsilon \sim \mathcal{N}(0,I)$，且**所有高斯噪声可合并为一个**。

### 最终前向闭式公式
$$
q(x_t \mid x_0) = \mathcal{N}\bigl(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\bigr)
$$
$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

---

# 三、反向过程（去噪）：从 $x_T \to x_0$
目标：学习 $p_\theta(x_{t-1}\mid x_t)$ 逼近真实后验 $q(x_{t-1}\mid x_t,x_0)$。

## 1. 真实后验 $q(x_{t-1}\mid x_t,x_0)$（高斯）
用贝叶斯 + 高斯乘积推导：
$$q(x_{t-1}\mid x_t,x_0) = \mathcal{N}\bigl(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\bigr)$$

### 推导均值 $\tilde\mu_t(x_t,x_0)$
由前向闭式：
$$x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\epsilon\right)$$
代入后验均值：
$$\tilde\mu_t(x_t,x_0) = \frac{1}{\sqrt{\alpha_t}}
\left(
x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon
\right)$$

---

# 四、DDPM 训练目标（核心损失）
模型不预测 $x_0$，而是**预测噪声 $\epsilon$**：
$$\epsilon_\theta(x_t,t) \approx \epsilon$$

## 训练损失
$$\boxed{
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\bigl\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\ t)\bigr\|^2
}$$

---

# 五、采样（生成）过程
从纯噪声 $x_T\sim\mathcal{N}(0,I)$ 开始迭代：
$$\boxed{
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}
\left(
x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
\right)
+ \sigma_t z,\quad z\sim\mathcal{N}(0,I)
}$$
其中 $\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t}$（或简化 $\sigma_t=\sqrt{\beta_t}$）。

---

## 总结（最常用公式速记）
1. **前向加噪**
$$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$$
2. **训练目标**
$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t,t)\|^2$$
3. **反向采样**
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr) + \sigma_t z$$

如果你需要，我可以把**每一步高斯乘积、贝叶斯展开、方差合并**写成更详细的逐行推导。
