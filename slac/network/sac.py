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
