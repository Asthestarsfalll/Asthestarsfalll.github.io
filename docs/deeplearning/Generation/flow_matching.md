## 流匹配 (Flow Matching)

在传统的 CNF 中，需要通过求解常微分方程（ODE）来建立数据与噪声的映射：

$$\frac{dx}{dt} = v_t(x), \quad x(0) = z \sim p_0, \quad x(1) \sim p_{data}$$

流匹配的目标是学习一个向量场 $v_t(x)$，使得这个向量场定义的流能将简单分布精确地推向数据分布。

### 直接拟合概率路径

扩散模型通过预测“噪声”来间接训练，而流匹配直接通过概率路径（Probability Path）进行监督。

最常用的是条件流匹配（Conditional Flow Matching）。给定起始点 $x_0$ 和终点 $x_1$，我们可以定义一条最简单的直线路径：

$$x_t = (1-t)x_0 + t x_1$$

这条路径对应的速度（向量场）就是：

$$v_t(x_t) = x_1 - x_0$$

流匹配的训练目标就是让神经网络 $v_\theta(x_t, t)$ 去逼近这个简单的速度向量。

:::tip
学习一个线性关系，而ddpm相当于在一个圆弧上，见[此](https://zhuanlan.zhihu.com/p/11228697012)
:::

## 对比

| **特性**   | **DDPM**     | **DDIM**   | **CNF (传统)**  | **Flow Matching**   |
| -------- | ------------ | ---------- | ------------- | ------------------- |
| **轨迹类型** | 随机 (SDE)     | 确定性 (ODE)  | 确定性 (ODE)     | 确定性 (ODE)           |
| **训练开销** | 中 (预测噪声)     | 同 DDPM     | **极高** (需解回传) | **低** (类似回归)        |
| **采样速度** | 慢 (50-1000步) | 快 (20-50步) | 慢 (由于刚性方程)    | **极快** (直轨迹，可1-10步) |
| **数学基础** | 分数匹配 (Score) | 分数匹配       | 变量代换定理        | 条件流匹配               |
| **路径优化** | 弯曲轨迹         | 较弯曲        | 任意            | **最优传输 (直线)**       |

### 代码对比

```python
# 假设 x1 是真实图像，x0 是标准高斯噪声
# t 是 [0, 1] 之间的随机时间点

# --- DDPM (扩散模型) ---
def ddpm_loss(x1, t):
    noise = torch.randn_like(x1)
    # 按照预定义的 alpha 进度表混合，路径是弯曲的
    xt = sqrt_alpha_cum[t] * x1 + sqrt_one_minus_alpha_cum[t] * noise
    predicted_noise = model(xt, t)
    return mse_loss(predicted_noise, noise) # 预测噪声

# --- Flow Matching (流匹配) ---
def flow_matching_loss(x1, t):
    x0 = torch.randn_like(x1)
    # 概率路径是完美的直线：xt = (1-t)*x0 + t*x1
    xt = (1 - t) * x0 + t * x1
    # 目标速度向量就是直线的斜率：v = x1 - x0
    target_velocity = x1 - x0
    predicted_velocity = model(xt, t)
    return mse_loss(predicted_velocity, target_velocity) # 预测速度
```

### 技术演进

**CNF (连续归一化流)** 要通过解 $dx/dt = v(x,t)$ 来生成数据。
- **问题：** 训练时需要对 ODE 求解器求导，显存爆炸，速度极慢。
- FM 不需要解 ODE 也能训练 $v(x,t)$。只要给定起点 $x_0$ 和终点 $x_1$，我可以强行规定轨迹是直的，然后让模型去学这个速度。

**DDPM** 要通过逐步加噪和去噪来生成数据。
- **问题：** 本质上它是在拟合“得分函数（Score Function）”，路径是随机且弯曲的。
- **FM 的改进：** 它兼容了扩散模型的训练稳定性，但把路径“拉直”了。这使得它在同样的参数量下，生成质量更高，速度更快。
