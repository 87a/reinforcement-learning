#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:example 5.1.py
@time:2022/03/13
"""
import random
import sys
from collections import defaultdict
from typing import Callable

import matplotlib
import numpy as np
import gym
import gym.spaces as spaces
from matplotlib import pyplot as plt

gamma = 1
cards = [10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def cmp(a, b):
    # a>b return 1.0, a<b return -1.0, a=b return 0.0
    return float(a > b) - float(a < b)


def draw_card():
    return np.random.choice(cards)


def draw_hand():
    return [draw_card(), draw_card()]


def is_usable(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    if is_usable(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    return sum_hand(hand) > 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    return sorted(hand) == [1, 10]


class BlackJack(gym.Env):
    def __init__(self):
        # 0:stick  1:hit
        self.action_space = spaces.Discrete(2)
        # 0: player's current sum  1: card dealer shows  2: whether A is usable
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )

    def reset(self):
        self.player = draw_hand()
        self.dealer = draw_hand()
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        if action:
            # if player continues to draw
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        else:
            # if player stop
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            # judge if natural
            if is_natural(self.player) and not is_natural(self.dealer):
                reward = 1.0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return sum_hand(self.player), self.dealer[0], is_usable(self.dealer)


def player_policy(state):
    """
    :param state: state now
    :return: action
    """
    player_score = state[0]
    return 0 if player_score >= 20 else 1


def first_visit_MC(policy: Callable, env: gym.Env, num_episodes: int):
    """
    :param policy:the policy player uses
    :param env:the question environment
    :param num_episodes:the number of episodes
    :return:v_\pi(s)
    """
    r_sum = defaultdict(float)
    r_count = defaultdict(float)
    r_v = defaultdict(float)

    for each_episode in range(num_episodes):
        print(f"Episode {each_episode}/{num_episodes}")
        sys.stdout.flush()
        # episode saves (state, action, reward)
        episode = []
        state = env.reset()

        done = False
        while not done:
            # use policy to get action
            action = policy(state)
            # get reward and next state
            next_state, reward, done, info = env.step(action)
            # save (state, action, reward)
            episode.append((state, action, reward))
            # update state
            state = next_state

        G = 0
        episode_len = len(episode)
        episode.reverse()
        for seq, data in enumerate(episode):
            state_visit = data[0]
            G = G * gamma + data[2]
            if seq != episode_len - 1:
                if data[0] in episode[seq + 1:][0]:
                    continue
            r_sum[state_visit] += G
            r_count[state_visit] += 1
            r_v[state_visit] = r_sum[state_visit] / r_count[state_visit]
    return r_v


def process_data_for_draw(v, ace):
    """
    v     : 状态价值函数
    ace   : 是否有可用A
    return: 返回处理好的三个坐标轴
    """

    # 生成网格点
    x_range = np.arange(12, 22)
    y_range = np.arange(1, 11)
    X, Y = np.meshgrid(x_range, y_range)

    # 根据是否有可用的A选择绘制不同的3D图
    if ace:
        Z = np.apply_along_axis(lambda _: v[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    else:
        Z = np.apply_along_axis(lambda _: v[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    return X, Y, Z


def plot_3D(X, Y, Z, xlabel, ylabel, zlabel, title):
    fig = plt.figure(figsize=(20, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.rainbow, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    ax.set_facecolor("white")
    fig.colorbar(surf)
    return fig


if __name__ == '__main__':
    env = BlackJack()
    num_episodes = 1000000
    v = first_visit_MC(policy=player_policy, env=env, num_episodes=num_episodes)
    X, Y, Z = process_data_for_draw(v, ace=True)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum", ylabel="Dealer Open Card", zlabel="Value", title="Usable Ace")
    fig.show()
    fig.savefig(f"D:\\RL\\images\\Usable_Ace(num_episodes={num_episodes}).jpg")
    X, Y, Z = process_data_for_draw(v, ace=False)
    fig = plot_3D(X, Y, Z, xlabel="Player Sum", ylabel="Dealer Open Card", zlabel="Value", title="No Usable Ace")
    fig.show()
    fig.savefig(f"D:\\RL\\images\\NO_Usable_Ace(num_episodes={num_episodes}).jpg")
