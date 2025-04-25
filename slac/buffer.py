from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.

    序列缓冲区
    """

    def __init__(self, num_sequences=8):
        '''
        这里传入的是序列的长度，也就是进行特征提取、动作预测的序列长度
        '''
        self.num_sequences = num_sequences
        self._reset_episode = False # 表示是否已经被重置了
        self.state_ = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        '''
        清空缓冲区
        '''
        self._reset_episode = False
        self.state_.clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state):
        '''
        重置连续序列缓冲区
        '''
        assert not self._reset_episode
        self._reset_episode = True
        # 注意，重置的时候是没有清空state_的，也就是说这里允许达到最大步数接着下一轮的训练数据一起的情况？todo
        self.state_.append(state)

    def append(self, action, reward, done, next_state):
        '''
        param action: 到到next_state得到的动作
        param reward: 得到的奖励
        param done: 到到当前state的结束标识，这里有一个特殊处理，到达了最大步数也为False 注意
        param next_state: 下一个状态

        存储到缓存
        '''
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_.append(next_state)

    def get(self):
        '''
        将采集满的序列全部提取出来
        '''
        state_ = LazyFrames(self.state_)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class ReplayBuffer:
    """
    Replay Buffer.
    重放缓冲区
    存储按顺序存储
    采样随机不连续样本采样
    对观察归一化到0~1
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action_ = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        # 这里的buff是一个小连续序列的缓冲区
        self.buff = SequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        param action: 到到next_state得到的动作
        param reward: 到到当前state得到的奖励
        param done: 到到当前state的结束标识，这里有一个特殊处理，到达了最大步数也为False 注意
        param next_state: 下一个状态
        param episode_done: 结束标识，真正的模型提供的结束标识
        """
        # 将当前的样本填充到小段序列缓冲区，里面存储的连续数据
        self.buff.append(action, reward, done, next_state)

        if self.buff.is_full():
            # 如果buff 满了，就存储到重放缓冲区
            state_, action_, reward_, done_ = self.buff.get()
            self._append(state_, action_, reward_, done_)

        if episode_done:
            # 如果游戏环境真的结束了，则重置小段缓冲区的存储数据
            self.buff.reset()

    def _append(self, state_, action_, reward_, done_):
        self.state_[self._p] = state_
        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        对于latent采样则没有多返回一个元素，是多少就是多少
        对于第0维度来说是随机的，但是对于第1维度来说是连续的，采集batch_size个连续的环境观察

        return 
        state_ shape (batch_size, num_sequences + 1, *state_shape)
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        # 因为实际的state时LazyFrame，所以这里有点特殊处理
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        对于sac训练则再奖励和结束标识多返回一个维度，标识-1，因为在存储数据时
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        # todo 为什么这里要多返回一个索引为-1的元素
        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n
