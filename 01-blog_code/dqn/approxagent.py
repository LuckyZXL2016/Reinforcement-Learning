# /usr/local/bin/python3.7
# -*- coding:utf-8 -*-

from random import random, choice
from gym import Env, spaces
import sys
import gym
import numpy as np
from approximator import Approximator
import torch

sys.path.append('../Gridworld2')
from gridworld2 import *

sys.path.append('../core')
from core import Transition, Experience, Agent


class ApproxQAgent(Agent):
    '''使用近似的价值函数实现的Q学习的个体
    '''

    def __init__(self, env: Env = None,
                 trans_capacity=20000,
                 hidden_dim: int = 16):
        if env is None:
            raise Exception("agent should have an environment")
        super(ApproxQAgent, self).__init__(env, trans_capacity)
        self.input_dim, self.output_dim = 1, 1

        # 适应不同的状态和行为空间类型
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = 1
        elif isinstance(env.observation_space, spaces.Box):
            self.input_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            self.output_dim = env.action_space.n
        elif isinstance(env.action_space, spaces.Box):
            self.output_dim = env.action_space.shape[0]
        # print("{},{}".format(self.input_dim, self.output_dim))

        # 隐藏层神经元数目
        self.hidden_dim = hidden_dim
        # 关键在下面两句，声明了两个近似价值函数
        # 变量Q是一个计算价值，产生loss的近似函数（网络），
        # 该网络参数在一定时间段内不更新参数
        self.Q = Approximator(dim_input=self.input_dim,
                              dim_output=self.output_dim,
                              dim_hidden=self.hidden_dim)
        # 变量PQ是一个生成策略的近似函数，该函数（网络）的参数频繁更新
        # 更新参数的网络
        self.PQ = self.Q.clone()
        return

    def _learning_from_memory(self, gamma, batch_size, learning_rate, epochs):
        # 随机获取记忆里的Transmition
        trans_pieces = self.sample(batch_size)
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        X_batch = states_0
        # 调用的时approximator的__call__方法
        y_batch = self.Q(states_0)

        # 使用了Batch，代码是矩阵运算
        # np.max => axis=1时取出最大的一列；axis=0时取出最大的一行
        # ～ True = -2;  ~ False = -1
        Q_target = reward_1 + gamma * np.max(self.Q(states_1), axis=1) * (~ is_done)
        y_batch[np.arange(len(X_batch)), actions_0] = Q_target
        # loss is a torch Variable with size of 1
        loss = self.PQ.fit(x=X_batch,
                           y=y_batch,
                           learning_rate=learning_rate,
                           epochs=epochs)
        mean_loss = loss.sum().item() / batch_size
        self._update_Q_net()
        return mean_loss

    def learning(self, gamma=0.99,
                 learning_rate=1e-5,
                 max_episodes=1000,
                 batch_size=64,
                 min_epsilon=0.2,
                 epsilon_factor=0.1,
                 epochs=1):
        '''learning的主要工作是构建经历，当构建的经历足够时，同时启动基于经历的学习
        '''
        total_steps, step_in_episode, num_episode = 0, 0, 0
        target_episode = max_episodes * epsilon_factor
        while num_episode < max_episodes:
            epsilon = self._decayed_epsilon(cur_episode=num_episode,
                                            min_epsilon=min_epsilon,
                                            max_epsilon=1,
                                            target_episode=target_episode)
            self.state = self.env.reset()
            self.env.render()
            step_in_episode = 0
            loss, mean_loss = 0.00, 0.00
            is_done = False
            while not is_done:
                s0 = self.state
                a0 = self.performPolicy(s0, epsilon)
                # act方法封装了将Transition记录至Experience中的过程
                s1, r1, is_done, info, total_reward = self.act(a0)
                # self.env.render()
                step_in_episode += 1
                # 当经历里有足够大小的Transition时，开始启用基于经历的学习
                if self.total_trans > batch_size:
                    loss += self._learning_from_memory(gamma,
                                                       batch_size,
                                                       learning_rate,
                                                       epochs)
            mean_loss = loss / step_in_episode
            print("{0} epsilon:{1:3.2f}, loss:{2:.3f}".
                  format(self.experience.last, epsilon, mean_loss))
            # print(self.experience)
            total_steps += step_in_episode
            num_episode += 1
        return

    def _decayed_epsilon(self, cur_episode: int,
                         min_epsilon: float,
                         max_epsilon: float,
                         target_episode: int) -> float:
        '''获得一个在一定范围内的epsilon
        '''
        slope = (min_epsilon - max_epsilon) / (target_episode)
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    def _curPolicy(self, s, epsilon=None):
        '''依据更新策略的价值函数(网络)产生一个行为
                '''
        Q_s = self.PQ(s)
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Q_s))

    def performPolicy(self, s, epsilon=None):
        return self._curPolicy(s, epsilon)

    def _update_Q_net(self):
        '''将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
        '''
        self.Q = self.PQ.clone()


def testApproxQAgent():
    env = gym.make("MountainCar-v0")
    # env = gym.make("PuckWorld-v0")
    # env = SimpleGridWorld()

    # 保存训练的视频
    # directory = "/home/reinforce/monitor"
    # env = gym.wrappers.Monitor(env, directory, force=True)

    agent = ApproxQAgent(env,
                         trans_capacity=10000,  # 记忆容量（按状态转换数计）
                         hidden_dim=16)  # 隐藏神经元数量
    env.reset()
    print("Learning...")
    agent.learning(gamma=0.99,  # 衰减引子
                   learning_rate=1e-3,  # 学习率
                   batch_size=64,  # 集中学习的规模
                   max_episodes=2000,  # 最大训练Episode数量
                   min_epsilon=0.01,  # 最小Epsilon
                   epsilon_factor=0.3,  # 开始使用最小Epsilon时Episode的序号占最大
                   # Episodes序号之比，该比值越小，表示使用
                   # min_epsilon的episode越多
                   epochs=2  # 每个batch_size训练的次数
                   )


if __name__ == "__main__":
    testApproxQAgent()
