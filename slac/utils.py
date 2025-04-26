import math

import torch
from torch import nn


def create_feature_actions(feature_, action_):
    '''
    feature shape: (N, num_sequences + 1, feature_dim)
    action shape: (N, num_sequences, action_dim)
    '''
    N = feature_.size(0) # batch size
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1) # 去掉最后一个序列 shape: (N, num_sequences * feature_dim)
    n_f = feature_[:, 1:].view(N, -1) # 去掉第一个序列 shape: (N, num_sequences * feature_dim)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1) # 去掉最后一个动作 shape: (N, (num_sequences - 1) * action_dim)
    n_a = action_[:, 1:].view(N, -1) # 去掉第一个动作 shape: (N, (num_sequences - 1) * action_dim)
    # Concatenate feature and action.
    fa = torch.cat([f, a], dim=-1) # todo 为啥能拼接 调试看看 这里相当于
    n_fa = torch.cat([n_f, n_a], dim=-1)
    return fa, n_fa


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    '''
    构建mlp层
    :param input_dim: 输入维度
    :param output_dim: 输出维度
    :param hidden_units: 隐藏层单元数
    :param hidden_activation: 隐藏层激活函数
    :param output_activation: 输出层激活函数
    '''
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std, noise):
    '''
    params log_std: 方差
    params noise: 噪声
    根据 SLAC（Stochastic Latent Actor-Critic）算法的背景，calculate_gaussian_log_prob 函数的作用是计算多维高斯分布的对数概率密度函数（log-probability density function, log-pdf）。具体来说，它计算给定噪声 noise 在高斯分布下的对数概率
    '''
    '''
    -0.5 * noise.pow(2):

    计算标准正态分布的平方项（-0.5 * (x - μ)^2 / σ^2），这里假设均值为 0，标准差为 1。
    - log_std:

    对概率密度函数的标准差部分取对数。
    .sum(dim=-1, keepdim=True):

    对最后一个维度求和，表示计算多维高斯分布的联合对数概率。
    - 0.5 * math.log(2 * math.pi) * log_std.size(-1):

    计算高斯分布的归一化常数部分（1 / sqrt(2πσ^2)），并扩展到多维

    返回的是 noise 在高斯分布下的对数概率密度值
    todo 跟踪后续概率密度在哪里使用
    '''
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    '''
    param log_std: 方差
    param noise: 噪声
    param action: 添加噪声后的动作
    '''
    # 得到noise 在高斯分布下的对数概率密度值
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    '''
    **Answer:**
`calculate_log_pi` 先用 `calculate_gaussian_log_prob` 计算噪声在高斯分布下的对数概率，然后减去
\(\log\bigl(1 - \text{action}^2 + 1e-6\bigr)\) 的求和，用于修正动作通过 `tanh` 变换后带来的分布变化。它最终返回添加噪声后的动作在策略分布下的对数概率，经常用于策略梯度方法里更新策略网络时所需的 \(\log \pi(a|s)\)。
    '''
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean, log_std):
    '''
    对预测的均值添加噪声
    '''
    noise = torch.randn_like(mean)
    # 因为tah会对预测的均值的值进行更改，缩放到合适的动作空间，那么同时方差也要做适配，所以进行calculate_log_pi计算tanh对应的log方差
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    '''
    计算先验分布和后验分布之间的KL散度
    todo 查看这种方式的计算数学公式与之对应
    '''
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
