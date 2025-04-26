import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from slac.network.initializer import initialize_weight
from slac.utils import build_mlp, calculate_kl_divergence


class FixedGaussian(torch.jit.ScriptModule):
    """
    Fixed diagonal gaussian distribution.
    todo 作用是啥？
    """

    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    @torch.jit.script_method
    def forward(self, x):
        # 这里构建的是一个均值（全0）和方差（全1）？todo
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(torch.jit.ScriptModule):
    """
    Diagonal gaussian distribution with state dependent variances.
    用于计算第一个时间步的后验分布，均值和方差
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        # 构建mlp层
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, x):
        if x.ndim == 3:
            # todo 为什么维度会是3
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        # 将预测的x分为均值和标准差
        mean, std = torch.chunk(x, 2, dim=-1)
        # 这里是为了防止标准差为0，保持为大于0 todo
        std = F.softplus(std) + 1e-5
        return mean, std


class Decoder(torch.jit.ScriptModule):
    """
    Decoder.
    这里应该是环境特征解码器
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        self.std = std

    @torch.jit.script_method
    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)


class Encoder(torch.jit.ScriptModule):
    """
    Encoder.
    从这里来看，这里应该是环境特征采集器
    """

    def __init__(self, input_dim=3, output_dim=256):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, x):
        B, S, C, H, W = x.size()
        # 看md
        x = x.view(B * S, C, H, W)
        x = self.net(x)
        #  这里就是将提取后的特征在分开为batch、时间序列、特征
        x = x.view(B, S, -1)
        return x


class LatentModel(torch.jit.ScriptModule):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    构建的是一个潜在模型，todo 作用是啥？
    """

    def __init__(
        self,
        state_shape, # 观察空间shape
        action_shape, # 动作空间shape
        feature_dim=256, # 特征维度
        z1_dim=32, # z1的维度
        z2_dim=256, # z2的维度
        hidden_units=(256, 256), # 隐藏层单元数
    ):
        super(LatentModel, self).__init__()
        # p(z1(0)) = N(0, I) todo 作用 得到的是一个固定的均值和方差
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0)) todo 作用 得到的是一个预测的均值和方差
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t)) todo 作用 预测的也是均值和方差
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t)) todo 作用 预测的也是均值和方差
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0)) todo 作用
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        #  z2_posterior_init 被定义为 z2_prior_init 说明简单的看成后验分布与先验分布相同
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t)) # todo
        self.z1_posterior = Gaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1)) 
        # todo 作用
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )

        # feat(t) = Encoder(x(t))
        self.encoder = Encoder(state_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            z1_dim + z2_dim,
            state_shape[0],
            std=np.sqrt(0.1),
        )
        self.apply(initialize_weight)

    @torch.jit.script_method
    def sample_prior(self, actions_, z2_post_):
        '''
        sample_prior 函数是 SLAC（Stochastic Latent Actor-Critic）算法中潜在变量模型的重要部分，负责计算潜在变量 z1 的先验分布。
        这与 sample_posterior 函数形成对比 - 后者计算后验分布，而前者计算先验分布。

        actions_: 动作序列张量，形状为 (B, S, A)
        B: 批次大小
        S: 序列长度
        A: 动作空间维度
        z2_post_: 从后验分布中采样得到的 z2 序列，形状为 (B, S+1, z2_dim)

        先验仅传入动作，后验有传入环境特征
        '''

        # p(z1(0)) = N(0, I) 计算初始潜在变量 z1(0) 的先验分布
        # p(z1(0)) = N(0, I): 表示初始时刻 z1(0) 的先验分布是均值为0，标准差为1的标准正态分布
        # self.z1_prior_init: 是一个 FixedGaussian 实例，它总是返回固定的均值（0）和标准差（1）
        # actions_[:, 0]: 第一个时间步的动作，这里实际上只用于确定批次大小，不影响输出值
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # 计算后续时间步 z1(t) 的先验分布
        # p(z1(t) | z2(t-1), a(t-1)) 表示 z1(t) 的先验分布是在给定上一时刻的 z2(t-1) 和动作 a(t-1) 的条件下计算的 因为在存储的时候，当前的观察和所执行的动作是存在一个时间差存储的 todo 注意
        # z2_post_[:, : actions_.size(1)]: 从 z2_post_ 中选择前 S 个时间步（与动作序列长度相匹配），对应 z2(0:S-1),相当于结合了上一个状态和 
        # [torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1)](http://vscodecontentref/19): 将 z2(t-1) 和 a(t-1) 在特征维度上连接
        # self.z1_prior: 一个 Gaussian 网络，计算条件先验分布的参数
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        # 合并初始和后续时间步的分布
        # z1_mean_init.unsqueeze(1): 在第二个维度（时间维度）上添加一个维度，使得形状变为 (B, 1, z1_dim)
        # torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1): 将初始 z1(0) 和后续 z1(1:S) 的均值在时间维度上连接
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        # torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1): 同理，连接标准差
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        # 返回一个元组，包含:
        # z1_mean_: 先验分布的均值，形状为 (B, S+1, z1_dim)
        # z1_std_: 先验分布的标准差，形状为 (B, S+1, z1_dim)
        return (z1_mean_, z1_std_)
    
    '''
    与 sample_posterior 的对比
    计算目的不同:

    sample_prior: 计算潜在变量 z1 的先验分布 p(z1)，不依赖于当前观测
    sample_posterior: 计算潜在变量 z1 和 z2 的后验分布 q(z1,z2|x)，依赖于观测
    输入参数不同:

    sample_prior 使用动作序列和从后验采样得到的 z2 序列
    sample_posterior 使用观测特征序列和动作序列
    输出不同:

    sample_prior 只返回 z1 的分布参数（均值和标准差）
    sample_posterior 返回 z1 的分布参数，以及采样得到的 z1 和 z2

    在 SLAC 中的作用
    函数在 SLAC 算法中的主要作用是:

    变分推断的一部分:

    在变分自编码器（VAE）框架中，需要同时有先验分布p(z)和后验分布q(z|x)
    两者之间的 KL 散度是变分下界（ELBO）的重要组成部分
    潜在动态建模:

    SLAC 的一个关键创新是学习潜在空间中的动态，即状态如何随时间和动作变化
    sample_prior 捕获了这种动态关系：p(z1(t) | z2(t-1), a(t-1))
    与后验的对齐:

    通过最小化先验和后验之间的 KL 散度，模型学习到更准确的状态表示和转移动态
    在 calculate_loss 中计算这个 KL 散度损失: loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_)


    '''

    # TorchScript 装饰器，用于将函数编译为 TorchScript，提高执行效率
    @torch.jit.script_method
    def sample_posterior(self, features_, actions_):
        '''
        sample_posterior 函数是 SLAC（Stochastic Latent Actor-Critic）算法中潜在变量模型的核心部分，负责从观测特征和动作序列中推断潜在状态变量。它实现了变分推断的后验采样过程。下面我将详细解析每一行代码
        param features_: 观察特征  shape (B, S+1, N)
        param actions_: 动作  shape (B, S, A)
        '''
        ## 初始时刻潜在变量计算
        # p(z1(0)) = N(0, I) 
        # features_[:, 0]：表示选择所有的batch的第一个时间步的特征 取所有批次的初始观测特征
        # self.z1_posterior_init: 一个神经网络，计算初始 z1 的后验分布参数
        # 专门用于计算序列的第一个潜在特征的后验分布，预测执行的动作
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        # 通过重参数化技巧采样得到初始的 z1 ，主要目的是为了采样得到的动作可以求导
        # 使用重参数化技巧采样 z1，公式: z = μ + σ * ε，其中 ε ~ N(0, 1)
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        ## 计算初始潜在变量 z2(0)
        # p(z2(0) | z1(0)) 计算在给定初始时刻的第一个潜在变量 z1(0) 的条件下，初始时刻的第二个潜在变量 z2(0) 的分布参数（均值和标准差）
        # 在 sample_posterior 函数中，它接收前面采样得到的 z1（即 z1(0)）作为输入 z1 是初始时刻 (z1(0))，由 z1_posterior_init 计算得到
        # z2_mean 和 z2_std 是 (q(z2(0) | z1(0))) 的均值和标准差。 todo 用来做啥？
        # self.z2_posterior_init: 与 z2_prior_init 相同）
        z2_mean, z2_std = self.z2_posterior_init(z1)
        # z2 是通过重参数化技巧从该分布中采样得到的初始潜在变量 (z2(0))。
        # 生成潜在变量序列的起点。
        # 通过 z1(0) 和 z2(0) 的初始化，后续时刻的潜在变量 (z1(t)) 和 (z2(t)) 可以逐步生成。
        # 同样使用重参数化技巧采样 z2(0)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        # 创建四个列表来存储序列中每个时间步的潜在变量及其分布参数
        # 先添加初始时刻 t=0 的值
        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std] # z1 初始时刻预测的均值和方差
        z1_ = [z1] # 
        z2_ = [z2] # 两个重采样的动作吧 todo

        # actions_.size(1)：表示时间序列长度
        # actions_.size(1) + 1应该是为了起始索引从1开始也能遍历时间序列的长度
        # 从 1 开始是因为已经处理了 t=0 的初始状态
        # 因为在采样时state时多了一个时间步（在存储的时候就多了），所以这里就算从1开始，也能对得上
        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            # features_[:, t]：获取当前时刻的所有观察特征
            # z2：上一个时间步的噪声动作？todo
            # actions_[:, t - 1]：实际的上一个时刻的动作
            # 将当前观测特征、上一时刻的 z2 和上一时刻的动作连接起来 计算得到的是什么？
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            # 使用重参数化技巧采样得到 z1(t)
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            # z1：z1_posterior预测、采样得到的动作均值？todo
            # z2：上一时刻计算的？
            # actions_[:, t - 1]：实际的上一个时刻的动作
            # self.z2_posterior计算得到？
            # 当前 z1(t)、上一时刻的 z2(t-1) 和上一时刻的动作 a(t-1) 连接起来
            # 计算 z2(t) 的后验分布（与 z2_prior 相同）
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            # 使用重参数化技巧采样得到 z2(t)
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            # 将计算得到存储到列表中
            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        # 将每一时刻计算得到的？？按照时间维度堆叠起来
        # torch.stack: 沿着新的维度（dim=1，即时间维度）将列表中的张量堆叠起来
        # 将四个变量作为元组返回：
        # z1_mean_: z1 的后验均值，形状 (B, S+1, z1_dim)
        # z1_std_: z1 的后验标准差，形状 (B, S+1, z1_dim)
        # z1_: 采样的 z1 值，形状 (B, S+1, z1_dim)
        # z2_: 采样的 z2 值，形状 (B, S+1, z2_dim)
        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    @torch.jit.script_method
    def calculate_loss(self, state_, action_, reward_, done_):
        '''
        params state_: 观察  shape (B, S+1, C, H, W)
        params action_: 动作  shape (B, S, A) A是一个动作维度大小，可能是一个多维的动作空间
        params reward_: 奖励  shape (B, S, 1)
        params done_: 结束标识  shape (B, S, 1)
        计算潜在模型的损失函数
        这里的state_是一个序列，action_是一个序列，reward_是一个序列，done_是一个序列
        '''
        # Calculate the sequence of features. 提取环境特征
        # feature shape (B, S+1, n)
        feature_ = self.encoder(state_)

        # Sample from latent variable model.
        # 以下计算的均值和方差都不是动作，而是一种潜在特征变量的分布
        # z1_mean_post_: z1 的后验均值，形状 (B, S+1, z1_dim)
        # z1_std_post_: z1 的后验标准差，形状 (B, S+1, z1_dim)
        # z1_: 采样的 z1 值，形状 (B, S+1, z1_dim)
        # z2_: 采样的 z2 值，形状 (B, S+1, z2_dim)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        # z1_mean_pri_: 先验分布的均值，形状为 (B, S+1, z1_dim)
        # z1_std_pri_: 先验分布的标准差，形状为 (B, S+1, z1_dim)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        # 这三行代码是在计算图像预测的负对数似然损失（negative log-likelihood loss），这是 SLAC 模型的一部分重构损失
        # 这三行代码共同实现了变分自编码器（VAE）中的重构损失。在 SLAC 模型中，这部分损失鼓励解码器从潜在变量 z 中重构出原始观察状态 state_。最小化这个损失意味着：
        # 潜在表示 z 包含了足够的信息来重建原始观察
        # 解码器能够准确地将潜在表示映射回原始观察空间
        # 这个损失与 KL 散度损失一起构成了 SLAC 变分下界（ELBO）的主要组成部分。
        z_ = torch.cat([z1_, z2_], dim=-1)
        # 对采集的特征分布还原为环境状态均质和方差
        state_mean_, state_std_ = self.decoder(z_)
        # 标准化的预测误差（也称为 z-score 或标准分数）
        # state_：真实观测到的环境状态（图像）
        # state_mean_：由解码器预测的状态均值
        # state_std_：由解码器预测的状态标准差
        # 1e-8：数值稳定性的小常数，防止除以零
        # state_noise_ 表示每个像素值的预测误差，以标准差为单位
        # 也就是这里就是解码后的状态和真实状态之间的差异 ？todo
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        # 这一行计算了高斯分布的对数概率密度函数（log-pdf）
        # -0.5 * state_noise_.pow(2)：标准正态分布的平方项 $-\frac{(x-\mu)^2}{2\sigma^2}$
        # -state_std_.log()：标准差的对数项 $-\ln(\sigma)$
        # -0.5 * math.log(2 * math.pi)：正态分布的归一化常数 $-\frac{1}{2}\ln(2\pi)$
        # 这三项加起来就是数据点在高斯分布下的对数概率密度 todo 了解高斯概率密度
        # 这计算了观测在预测分布下的对数似然。对于高斯分布，对数似然值越大，表示观测与预测分布越匹配 todo 对数似然值
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        # 这计算了观测在预测分布下的对数似然。对于高斯分布，对数似然值越大，表示观测与预测分布越匹配。
        # 计算总损失
        # 将对数似然取负并求平均，得到负对数似然损失
        # mean(dim=0)：在批次维度上计算平均值
        # 对所有剩余维度（通道、高、宽）求和
        # 取负号是因为训练时是最小化损失，而我们希望最大化似然
        loss_image = -log_likelihood_.mean(dim=0).sum()
        # 一个负对数似然（negative log-likelihood）损失，用于衡量解码器重构的状态与真实观测状态之间的差异：
        # 取负号是因为我们要最小化损失，而最大化似然。因此：
        # 损失值越小：表示对数似然越大，即重构质量越好
        # 损失值越大：表示对数似然越小，即重构质量越差

        # Prediction loss of rewards.
        # 生成用于奖励预测的特征向量
        # z_[:, :-1]: 当前时间步 t 的潜在状态表示 (由 z1_ 和 z2_ 连接而成)
        # action_: 当前时间步 t 的动作
        # z_[:, 1:]: 下一时间步 t+1 的潜在状态表示
        # 这个特征向量包含了从 t 到 t+1 的状态转移信息和执行的动作，用于预测在这个转移中获得的奖励
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        #  预测奖励的均值和标准差
        # x.view(B * S, X): 将输入重塑为二维张量，便于通过神经网络处理
        # self.reward: 一个 Gaussian 网络，输出奖励的预测均值和标准差
        # 这体现了 SLAC 模型对奖励的概率建模方法 - 奖励不是确定性值，而是通过高斯分布描述的随机变量
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        # 恢复序列形状
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        # 类似于状态重构损失，这里计算了奖励预测的标准化误差
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        # 与状态重构损失相同，这里计算了奖励的对数概率密度
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
#         函数返回三个损失值：
# loss_kld: KL 散度损失，确保后验分布与先验分布接近
# loss_image: 状态重构损失，确保潜在表示能重构原始观察
# loss_reward: 奖励预测损失，确保潜在表示能预测状态转移的奖励
        return loss_kld, loss_image, loss_reward
