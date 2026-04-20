
- 目标分布：$P(x)$
- 近似分布：$Q(x)$
- KL 散度（离散形式）：

$$D_{KL}(P \parallel Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = \mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]$$

- 定义概率比：

$$r(x) = \frac{Q(x)}{P(x)} \quad\Rightarrow\quad \log\frac{P(x)}{Q(x)} = -\log r(x)$$

---

## K1 估计（朴素蒙特卡洛估计器）

**公式**：

$$K_1(x) = \log\frac{P(x)}{Q(x)} = -\log r(x)$$

样本平均估计：

$$\hat{D}_{KL}^{K1} = \frac{1}{N}\sum_{i=1}^N K_1(x_i) = \frac{1}{N}\sum_{i=1}^N \left[-\log r(x_i)\right]$$

**特点**：
- **无偏**：$\mathbb{E}[K_1(x)] = D_{KL}(P \parallel Q)$
- **高方差**：$\log$ 函数对极端值敏感，样本波动大
-  **可能为负**：单个样本的 $\log\frac{P(x)}{Q(x)}$ 可正可负，但期望非负

## K2 估计（二阶矩近似估计器）

**公式**：

$$K_2(x) = \frac{1}{2}\left(\log\frac{Q(x)}{P(x)}\right)^2 = \frac{1}{2}\left(\log r(x)\right)^2$$

样本平均估计：

$$\hat{D}_{KL}^{K2} = \frac{1}{N}\sum_{i=1}^N K_2(x_i) = \frac{1}{2N}\sum_{i=1}^N \left(\log r(x_i)\right)^2$$

**推导与关系**：
1.  对 $\log r(x)$ 在 $r=1$ 处做泰勒展开：

    $$-\log r = (1 - r) + \frac{(1 - r)^2}{2} + \frac{(1 - r)^3}{3} + \cdots$$

2.  取二阶近似：

    $$-\log r \approx \frac{(1 - r)^2}{2}$$

3.  进一步用 $\log r \approx r-1$（一阶泰勒）替换，可得：

    $$-\log r \approx \frac{1}{2}(\log r)^2$$

4.  因此：

    $$D_{KL}(P \parallel Q) = \mathbb{E}[-\log r] \approx \mathbb{E}\left[\frac{1}{2}(\log r)^2\right]$$

这就是 K2 估计的来源，其期望对应 **F-散度** 族中的一种近似。

**特点**：
- **有偏**：$\mathbb{E}[K_2(x)] \neq D_{KL}(P \parallel Q)$，是近似值
- **低方差**：平方项平滑了极端波动
- **恒不为负**：平方保证 $K_2(x) \ge 0$，与 KL 散度非负性一致

## K3 估计

> 这是你图里还没展示完的部分，我按 RL 常用形式补全

**公式**：

$$K_3(x) = \frac{1}{2}\left(\frac{Q(x)}{P(x)} - 1\right)^2 = \frac{1}{2}\left(r(x) - 1\right)^2$$

样本平均估计：

$$\hat{D}_{KL}^{K3} = \frac{1}{N}\sum_{i=1}^N K_3(x_i) = \frac{1}{2N}\sum_{i=1}^N \left(r(x_i) - 1\right)^2$$

**推导**：
1.  从 KL 定义出发：

    $$D_{KL}(P \parallel Q) = \mathbb{E}_{x\sim P}\left[-\log r(x)\right]$$

2.  对 $-\log r(x)$ 在 $r=1$ 处做二阶泰勒展开：

    $$-\log r = (1 - r) + \frac{(1 - r)^2}{2} + o\left((r-1)^2\right)$$

3.  当策略更新很小时，$\mathbb{E}[1 - r] \approx 0$（因为 $\mathbb{E}[r] = \mathbb{E}[Q/P] = 1$），一阶项可忽略：

    $$-\log r \approx \frac{(1 - r)^2}{2} = \frac{(r - 1)^2}{2}$$

4.  因此：

    $$D_{KL}(P \parallel Q) \approx \mathbb{E}\left[\frac{(r - 1)^2}{2}\right]$$

**特点**：
- **有偏**：依赖小更新假设，更新越大偏差越大
- **低方差**：平方形式稳定，计算高效
- **恒不为负**：与 KL 散度非负性一致
