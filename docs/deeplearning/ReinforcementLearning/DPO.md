---
title: Direct Preference Optimization
description: 直接偏好优化
---

REF： https://zhuanlan.zhihu.com/p/671780768

DPO（Direct Preference Optimization）和 RLHF（Reinforcement Learning from Human Feedback）是两种不同的方法，DPO 是一种直接优化用户或专家偏好的方法。它通过收集用户对模型输出的偏好数据，直接优化模型的参数，使得模型输出更符合用户的偏好，而后者通过收集人类对模型行为的反馈，使用这些反馈来指导强化学习过程，从而优化模型的性能。

**优点**： DPO 不需要复杂的强化学习算法，直接使用偏好数据进行优化；（更少的计算资源）由于不需要模拟环境和大量的探索，计算资源需求相对较低。

**缺点**：需要大量的高质量偏好数据，收集这些数据可能成本较高；（适用范围有限）对于一些需要复杂策略和长时间决策的任务，DPO 可能不如 RLHF 有效。

## RLHF

RLHF 一般会分 2 步:

第一步是训练 reward model。训练数据是同一个 prompt 的 2 个回答，让人或 GPT4 标注哪个回答更好，reward model 会去优化如下的 loss：

$$
\max_{r_{\phi}}\left\{\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[\log\sigma(r_\phi(x,y_\text{win})-r_\phi(x,y_\text{lose}))]\right\}
$$

其中 $r_\phi$ 就是 reward model 用来给回答打分。$\mathcal{D}$ 是训练数据集，$x$ 是 prompt，$y_\text{win}$ 和 $y_\text{lose}$ 分别是好的回答和不好的回答。也就是说，要尽可能让好的回答的得分比不好的回答高，拉大他们之间的差别。

第二步是用 RL 算法来提升模型的得分。使用的 loss 是：

$$
\max_{\pi_\theta}\left\{\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[r_\phi(x,y)]-\beta\mathbb{D}_{\text{KL}}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]\right\}
$$

其中 $\pi_\theta$ 是我们在训练的 LLM，$\pi_\text{ref}$ 是训练的初始值。这个 loss 意思是希望 LLM 输出的回答的评分能尽可能高，同时 $\pi_\theta$ 不要偏离 $\pi_\text{ref}$ 太多，保证它还能正常做回答，不要训成一个评分很高但是回答乱码的东西。

DPO 的作者们意识到，后面的这个式子是有显式解的。因为：

$$
\begin{aligned}\max_{\pi_\theta}&\left\{\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[r_\phi(x,y)] -\beta\mathbb{D}_{\text{KL}}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]\right\}\\&=\max_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}]\\&=\min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - \frac{1}{\beta} r_\phi(x,y)]\\&=\min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)e^{r_\phi(x,y)/\beta}}]\end{aligned}
$$

:::tip

先将 KL 散度公式展开，吸收至期望中，再提取一个 $-\beta$，和优化目标无关去除即可，即变为最小化，最后合并入 $\log$ 中。

:::

如果我们归一化一下分母，即取 $Z(x)=\sum_y\pi_\text{ref}(y|x)e^{r_\phi(x,y)/\beta}$，也就可以构造出一个新的概率分布：

$$
\pi^*(y|x) = \pi_\text{ref}(y|x)e^{r_\phi(x,y)/\beta}/Z(x)
$$

那么上式变成了：

$$
\begin{aligned}\min_{\pi_\theta}&\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)e^{r_\phi(x,y)/\beta}}]\\ &=\min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi^*(y|x)}-\log Z(x)]\\ &=\min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi^*(y|x)}]\\ &=\min_{\pi_\theta}\mathbb{E}_{x\sim \mathcal{D}}\mathbb{D}_\text{KL}(\pi_\theta(y|x)||\pi^*(y|x)) \end{aligned}
$$

由于 KL 散度在 2 个分布相等时取最小值，我们得到了这样的结论：RLHF 训练希望得到的最优的概率分布就是 $\pi^*$*。

另一个角度来说，由 $\pi^*$ 的公式，我们相当于是得到了 $r_\phi$ 和 $\pi^*$ 的关系，那么是否我们可以把训练 $r_\phi$ 转化成直接去训练 $\pi^*$ 呢？

简单转换一下 $\pi^*$ 的定义式，可以得到：

$$
r_{\phi}(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta \log Z(x)
$$

带入最上面优化 $r_\phi$ 的 loss，也就有了：

$$
\max_{\pi^*}\left\{\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[\log\sigma(\beta\log\frac{\pi^*(y_\text{win}|x)}{\pi_\text{ref}(y_\text{win}|x)} - \beta\log\frac{\pi^*(y_\text{lose}|x)}{\pi_\text{ref}(y_\text{lose}|x)})]\right\}
$$

或者说，我们可以直接用这个 loss 去求 $\pi_\theta$：

$$
\max_{\pi_\theta}\left\{\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[\log\sigma(\beta\log\frac{\pi_\theta(y_\text{win}|x)}{\pi_\text{ref}(y_\text{win}|x)} - \beta\log\frac{\pi_\theta(y_\text{lose}|x)}{\pi_\text{ref}(y_\text{lose}|x)})]\right\}
$$

这就是 DPO 的 loss。DPO 通过以上的公式转换把 RLHF 无损地转化为了 SFT，在训练的时候不再需要同时跑 4 个模型（reward model, ref model, critic, actor），而是只用跑 actor 和 ref 2 个模型，甚至由于不再在线采数据，ref model 的输出可以预先存下来，训练的时候重复使用。
