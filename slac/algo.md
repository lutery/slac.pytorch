# 什么是负的熵

当我们在SAC算法的代码中看到 `log_pi` 被描述为"动作的对数概率（负的熵）"时，这里涉及到统计学和信息论中的几个关键概念。

## 熵的定义

**熵**（Entropy）是信息论中表示随机变量不确定性的度量。对于一个离散随机变量X，其熵定义为：

\[ H(X) = -\sum_{x} p(x) \log p(x) \]

其中p(x)是随机变量X取值为x的概率。

## 负的熵与对数概率的关系

在代码中，`log_pi` 是策略π的对数概率（log probability），表示为log(π(a|s))，即在状态s下选择动作a的概率的对数。

**负的熵**可以表示为：

\[ -H(X) = \sum_{x} p(x) \log p(x) \]

当我们计算策略π的熵期望时：
\[ H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)] \]

所以，`log_pi` 的负值的期望就是策略的熵：
\[ H(\pi) = -\mathbb{E}[\log \pi(a|s)] \]

这就是为什么在代码中，`log_pi` 被称为"负的熵"，因为：
- `log_pi` = log(π(a|s))
- `-log_pi` = -log(π(a|s))
- 而熵的期望 H(π) = -E[log(π(a|s))]

## 在SAC中的应用

在SAC算法中：
1. 当计算`entropy = -log_pi.detach().mean()`时，这实际上是在计算策略的熵。
2. 当使用`next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi`时，减去`self.alpha * log_pi`相当于加上`self.alpha * 单个样本的熵贡献`，鼓励策略更具随机性。

负的熵越小，意味着熵越大，策略越随机（探索性越高）。SAC通过调整温度参数`alpha`来控制熵正则化的强度，从而平衡探索与利用之间的权衡。