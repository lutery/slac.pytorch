import torch
from torch import nn

from slac.network.initializer import initialize_weight
from slac.utils import build_mlp, reparameterize


class GaussianPolicy(torch.jit.ScriptModule):
    """
    Policy parameterized as diagonal gaussian distribution.
    """

    def __init__(self, action_shape, num_sequences, feature_dim, hidden_units=(256, 256)):
        """
        param action_shape: 动作空间维度
        param num_sequences: 序列长度 这里的序列长度就是SlacObservation中的环境序列长度，表明总共采集了多少个特征用于训练
        param feature_dim: 特征维度，就是观察经过特征提取后的维度
        param hidden_units: 隐藏层单元数 用于构建决定动作均值和方差的MLP层数
        """
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        # 这里的意思是说动作策略网络和潜在模型的卷积层是共享的，所以这里没有卷积特征提取器 ？ todo
        self.net = build_mlp(
            input_dim=num_sequences * feature_dim + (num_sequences - 1) * action_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature_action):
        means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, feature_action):
        '''
        param feature_action: 传入观察特征_动作组合 （到当前观察执行的动作)
        得到预测的动作均值和方差（经过缩放到合理范围）
        '''
        mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
        # 这里对预测的动作添加噪声，同时限制了动作的取值范围到合理的范围
        action, log_pi = reparameterize(mean, log_std.clamp(-20, 2))
        return action, log_pi
    
'''
log_pi是对数概率密度，它和熵和方差的关系是什么样子的？


# `log_pi`与熵和方差的关系

`log_pi`是动作的对数概率密度，它与熵和方差有着密切的关系：

## 1. `log_pi`与熵的关系

熵是随机变量不确定性的度量，定义为概率分布的负对数期望：

$$H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$$

因此：
- **熵** = $-\mathbb{E}[\log \pi(a|s)]$ = $-\mathbb{E}[\text{log\_pi}]$
- 当策略的`log_pi`值越小（更负），策略的熵越大
- 当策略的`log_pi`值越大（更接近0），策略的熵越小

在SAC算法中：
- 当计算`entropy = -log_pi.detach().mean()`时，直接计算了策略的熵
- 熵越大意味着策略越随机（探索性越强）
- 熵越小意味着策略越确定（利用性越强）

## 2. `log_pi`与方差的关系

对于高斯策略，方差直接影响`log_pi`的值：

### 对数概率密度公式

对于高斯分布 $\mathcal{N}(\mu, \sigma^2)$，其对数概率密度为：

$$\log \pi(a|s) = -\frac{(a-\mu)^2}{2\sigma^2} - \log(\sigma) - \frac{1}{2}\log(2\pi)$$

从这个公式可以看出：
- 方差 $\sigma^2$ 越大，$\log \pi(a|s)$ 通常会越小（更负）
  1. $-\frac{(a-\mu)^2}{2\sigma^2}$ 项在方差大时增大（更接近0）
  2. 但 $-\log(\sigma)$ 项在方差大时减小（更负）
  3. 通常第二项效应更显著，因此高方差往往导致更低的对数概率密度

### 在SAC中的体现

当策略的`log_std`（对数标准差）增大时：
1. 生成的动作分布更分散（方差更大）
2. `log_pi`值通常变得更小（更负）
3. 策略的熵增加，探索性增强
4. 策略从决定性行为向随机行为转变

## 3. 在SAC算法中的作用机制

在SAC中，这三者相互作用：
1. 策略网络预测`mean`和`log_std`（对数标准差）
2. `reparameterize`函数通过`log_std`控制采样方差，并计算对应的`log_pi`
3. `log_pi`用于计算策略的熵，影响SAC的"温度参数"`alpha`
4. 策略通过优化`-torch.mean(torch.min(q1, q2) - self.alpha * log_pi)`来平衡奖励最大化和熵最大化

## 总结

- **方差**控制策略的探索程度和动作的分散性
- **`log_pi`**是给定策略参数下特定动作的对数概率密度
- **熵**是`log_pi`的负期望，表示整个动作分布的随机性或不确定性

SAC算法通过这三者之间的关系，在探索与利用之间找到平衡，学习到既能获得高奖励又保持适当探索性的策略。

'''


class TwinnedQNetwork(torch.jit.ScriptModule):
    """
    Twinned Q networks.
    用于评价网络
    """

    def __init__(
        self,
        action_shape, # 动作空间维度
        z1_dim, # z1的维度 todo
        z2_dim, # z2的维度 todo
        hidden_units=(256, 256), # 隐藏层单元数
    ):
        super(TwinnedQNetwork, self).__init__()
        
        # 构建了两个net，输入的是动作和z1和z2的特征
        # 并进行了参数初始化
        # 两个net分别预测两个q值
        self.net1 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)
        self.net2 = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        return self.net1(x), self.net2(x)
