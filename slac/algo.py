import os

import numpy as np
import torch
from torch.optim import Adam

from slac.buffer import ReplayBuffer
from slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
from slac.utils import create_feature_actions, grad_false, soft_update


class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape, # 观察空间
        action_shape, # 动作空间
        action_repeat, # 动作重复次数
        device, # 设备
        seed, # 随机种子 
        gamma=0.99, # 折扣因子
        batch_size_sac=256, # SAC的批量大小
        batch_size_latent=32, # Latent的批量大小
        buffer_size=10 ** 5, # 缓冲区大小
        num_sequences=8, # 序列长度
        lr_sac=3e-4, # SAC的学习率
        lr_latent=1e-4, # Latent的学习率
        feature_dim=256, # 特征维度
        z1_dim=32, # z1的维度 todo
        z2_dim=256, # z2的维度 todo
        hidden_units=(256, 256), # 隐藏层单元数 todo
        tau=5e-3, # 软更新的参数
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, action_shape, device)

        # Networks.
        # 动作策略网络
        self.actor = GaussianPolicy(action_shape, num_sequences, feature_dim, hidden_units).to(device)
        # 这里应该是sac的评价网络模型
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units).to(device)
        # 这里是潜在模型，应该就是slac的特有，环境特征的采集和解码应该就是这里进行吧 todo
        self.latent = LatentModel(state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units).to(device)
        # 将self.critic的参数完全拷贝到self.critic_target
        soft_update(self.critic_target, self.critic, 1.0)
        # 设置目标网络的参数不需要更新 梯度设置为false
        grad_false(self.critic_target)

        # Target entropy is -|A|. 目标熵 todo为什么是这么算目标熵
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0. 
        # todo 作用，设置为0.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            # 这里是干嘛todo
            self.alpha = self.log_alpha.exp()

        # Optimizers. 创建优化器
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.learning_steps_sac = 0 # sac的训练次数，要保存
        self.learning_steps_latent = 0 # 潜在空间的训练次数，要保存
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        # JIT compile to speed up. todo 作用
        fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        # 通过跟踪模型的前向计算路径来生成 TorchScript，具体作用是？todo
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))

    def preprocess(self, ob):
        '''
        对环境进行预处理

        ob.state 不是tensor，应该是numpy
        由于ob时SlacObservation，所以.state拿到的是一个序列状态
        '''
        # 这里也进行了一次归一化，因为这里传入的不是缓冲区的数据
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        with torch.no_grad():
            # 对环境进行特征提取
            feature = self.latent.encoder(state).view(1, -1)
        # 拿到每个观察对应的动作
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device)
        # 观察特征，动作
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        '''
        param ob: SlacObservation
        根据环境观察得到动作
        '''
        # 预处理，得到（观察特征_动作） todo 这个动作是什么动作
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            # 根据特征得到预测的动作（添加了噪声）
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_random):
        '''
        对环境进行一次动作执行

        ob：这里传入的是SlacObservation
        '''
        t += 1

        # 得到待执行的动作
        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        state, reward, done, _ = env.step(action)
        # 如果t到到了最大步数，则mask为False
        # 否则mask为实际的done
        mask = False if t == env._max_episode_steps else done
        ob.append(state, action)
        # 存储数据
        self.buffer.append(action, reward, mask, state, done)

        if done:
            # 如果游戏结束，则
            # 重置环境
            # 重置SlacObservation观察
            # 重置buff
            # 在重置时，会将重置的观察保存到缓冲区中 注意
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return t

    def update_latent(self, writer):
        '''
        更新潜在空间模型
        用于模拟环境的潜在动态、动作解码器、奖励解码器
        '''
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()

        if self.learning_steps_latent % 1000 == 0:
            # 这里是记录训练的损失
            writer.add_scalar("loss/kld", loss_kld.item(), self.learning_steps_latent)
            writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        # state_ shape (batch_size, num_sequences + 1, *state_shape) t
        # action_ shape (batch_size, num_sequences, *action_shape)
        # reward_ shape (batch_size, 1)
        # done_ shape (batch_size, 1)
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, action_):
        '''
        预处理sac训练的数据
        # state_ shape (batch_size, num_sequences + 1, *state_shape) t
        # action_ shape (batch_size, num_sequences, *action_shape)

        prepare_batch 是 SLAC（Stochastic Latent Actor-Critic）算法中的关键函数，用于将原始观察和动作序列转换为适合 SAC（Soft Actor-Critic）算法训练的格式。它处理时序数据并提取潜在表示，构建强化学习所需的状态-动作对
        '''
        with torch.no_grad():
            # f(1:t+1) 对观察进行特征提取
            # feature shape (batch_size, num_sequences + 1, feature_dim)
            feature_ = self.latent.encoder(state_)
            # z(1:t+1) 对观察进行潜在空间采样，并cat先验特征和后验特征
            # 形状为 (batch_size, num_sequences + 1, z1_dim + z2_dim)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1)

        # z(t), z(t+1) 倒数第二个和最后一个时间步的潜在空间特征
        # 这两个状态对是强化学习中的关键元素，用于计算当前状态的Q值和下一状态的目标Q值 todo 查看shape
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t) 最后一个时间步的动作 todo
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t)) 这里就是在拼接当前状态和上一个动作的特征 创建特征-动作对
        # feature_action: 用于策略网络输入，包含时间 1:t 的观察特征和时间 1:t-1 的动作
        # next_feature_action: 用于计算下一时刻的动作，包含时间 2:t+1 的观察特征和时间 2:t 的动作
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        # todo 补充shape
        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        '''
        param z: 当前时刻的潜在空间特征
        param next_z: 下一个时刻的潜在空间特征
        param action: 当前时刻的动作
        param next_feature_action: 下一个时刻的特征-动作对
        param reward: 当前时刻的奖励
        param done: 当前时刻的结束标识
        '''
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            # 下一个时刻的潜在状态特征和动作预测下一个状体的Q值
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            # 使用sac的q值来计算目标q值
            # self.alpha * log_pi 是熵正则化项
            # log_pi 是动作的对数概率（负的熵）
            # self.alpha 是温度参数，控制探索与利用的平衡
            # 减去这一项意味着鼓励策略具有高熵（更多样化的行为）
            # 这是SAC的核心公式之一，实现了"最大熵强化学习"原则 todo 查看sac的其他的代码
            # 传统Q值仅考虑奖励，而SAC的目标函数加入了最大化策略熵的项
            # 最终目标是找到既能获得高奖励又能保持高探索性的策略
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        # 计算目标q值
        target_q = reward + (1.0 - done) * self.gamma * next_q
        # 计算当前q值和目标q值的损失
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)

    def update_actor(self, z, feature_action, writer):
        '''
        param z: 当前时刻的潜在空间特征
        param feature_action: 当前时刻的特征-动作对
        '''
        # 根据当前时刻的潜在空间特征和特征-动作对，得到当前时刻的动作
        action, log_pi = self.actor.sample(feature_action)
        # 预测当前时刻的Q值
        q1, q2 = self.critic(z, action)
        # 利用最小化负q值来寻找使得q值更大的动作
        # 同时log_pi是动作的对数概率（负的熵），它越小
        # -log_pi.detach().mean()时，这实际上是在计算策略的熵。越大的熵意味着策略越随机
        # 这里的loss_actor是最小化q值和最大化熵的组合
        loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with torch.no_grad():
            entropy = -log_pi.detach().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/actor", loss_actor.item(), self.learning_steps_sac)
            writer.add_scalar("loss/alpha", loss_alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/alpha", self.alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/entropy", entropy.item(), self.learning_steps_sac)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
