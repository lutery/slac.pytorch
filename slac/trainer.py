import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        '''
        param state_shape: 环境shape
        param action_shape: 动作shape
        param num_sequences: 作用？todo 这个值默认是8
        '''
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        '''
        传入环境观察
        '''

        # 构建一个长度为num_sequences的队列
        self._state = deque(maxlen=self.num_sequences) # uint8
        self._action = deque(maxlen=self.num_sequences - 1) # 浮点数
        # 初始的值是0
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        # 保存观察，初始状态保存的是reset的观察，所以没有动作
        self._state.append(state)

    def append(self, state, action):
        '''
        params state: 环境观察，这里传入的相当于执行动作后的next_state
        params action: 到达state所执行的动作，
        '''
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Trainer:
    """
    Trainer for SLAC.
    SLAC训练器
    """

    def __init__(
        self,
        env, # 训练环境
        env_test, # 测试环境
        algo, # 算法器
        log_dir, # 日志路径
        seed=0, # 随机种子
        num_steps=3 * 10 ** 6, # 应该是训练的步数 实际使用时会除以action_repeat得到真实的步数
        initial_collection_steps=10 ** 4, # 初始化缓冲区的训练步数
        initial_learning_steps=10 ** 5, # 初始化潜在空间的训练步数
        num_sequences=8, # todo 
        eval_interval=10 ** 4, # todo
        num_eval_episodes=5, # todo 
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        # todo 这两个观察是什么作用？
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = self.env.action_repeat
        self.num_steps = num_steps # 代表要执行的步数（不包含action_repeat)
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        # 保存reset的state到ob中
        self.ob.reset_episode(state)
        # 保存state到缓冲区
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        # 在最开始训练时，缓冲区预热
        for step in range(1, self.initial_collection_steps + 1):
            t = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        # 首先更新潜变量模型（latent variable model），以便 SLAC 能够利用（学到的）潜在动态进行更好的学习。
        # 首先用预热好的缓冲区，训练潜在空间动作规律，以便辅助模型训练，todo 思考是否可以迁移到其他强化学习方法
        # 有点像dreamer
        '''
        选中的这句话：

        ```python
        Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        ```

        ### 含义：
        这句话的意思是：
        **"首先更新潜变量模型（latent variable model），以便 SLAC 能够利用（学到的）潜在动态进行更好的学习。"**

        ---

        ### 背景：
        在 SLAC（Stochastic Latent Actor-Critic）算法中，潜变量模型（latent variable model）是核心部分，用于学习环境的潜在动态（latent dynamics）。潜变量模型通过对环境状态和动作的建模，生成一个潜在空间的表示，从而帮助策略网络和价值网络更高效地学习。

        ---

        ### 具体作用：
        1. **潜变量模型的学习**：
        - 潜变量模型需要先训练，以捕获环境的潜在动态（如状态转移规律）。
        - 通过学习潜在动态，SLAC 可以在潜在空间中进行推理和预测，而不依赖于高维的原始状态空间。

        2. **为后续学习提供基础**：
        - 在训练初期，潜变量模型的更新优先于策略网络（SAC 部分）的更新。
        - 这是因为策略网络依赖于潜变量模型生成的潜在表示。如果潜变量模型没有学好，策略网络的学习效果会受到影响。

        ---

        ### 总结：
        这句话强调了在 SLAC 算法中，潜变量模型的优先更新是为了确保其能够准确地捕获环境的潜在动态，从而为后续的策略学习提供可靠的基础。
        '''
        # 如果可持续话训练，这里应该是要跳过吧？todo
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        # 训练，初始轮数：self.initial_collection_steps + 1
        # 结束轮数：self.num_steps // self.action_repeat + 1
        # 这里的self.num_steps // self.action_repeat + 1 是因为每次执行动作都要乘以action_repeat
        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            t = self.algo.step(self.env, self.ob, t, False)

            # Update the algorithm.
            # 每次执行一步后更新latent模型和sac模型
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
                self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))

        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, step_env):
        mean_return = 0.0

        for i in range(self.num_eval_episodes):
            state = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                state, reward, done, _ = self.env_test.step(action)
                self.ob_test.append(state, action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
