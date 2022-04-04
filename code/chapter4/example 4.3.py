#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:example 4.3.py
@time:2022/03/08
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(250)
S = [i for i in range(101)]
V = [0 for s in S]
# V = np.random.rand(101)
# V[-1] = 0
newV = [0 for s in S]
pi = [0 for s in S]
ph = 0.4


def isEnd(state: int) -> bool:
    assert 0 <= state <= 100
    if state == 0 or state == 100:
        return True
    return False


def get_actions(state: int) -> list:
    assert 0 <= state <= 100
    max = min(state, 100 - state)
    actions = [i for i in range(max + 1)]
    return actions


def get_reward(state: int) -> int:
    if state == 100:
        return 1
    return 0


def get_p_r_v(state: int):
    actions = get_actions(state)
    results = []
    # action, p,reward,v(s')
    for action in actions:
        result = []
        result.append((action, ph, get_reward(state + action), V[state + action]))
        result.append((action, 1 - ph, get_reward(state - action), V[state - action]))
        results.append(result)
    return results


def value_iteration(epochs: int = 100, threshold: float = 1e-20):
    for epoch in range(epochs):
        delta = 0
        for s in S[1:-1]:
            results = get_p_r_v(s)
            v = 0
            best_action = None
            best_actions = []
            for result in results:
                v_temp = result[0][1] * (result[0][2] + result[0][3]) + result[1][1] * (result[1][2] + result[1][3])
                if v_temp >= v:
                    v = v_temp
                    best_action = result[0][0]
                    best_actions.append(best_action)
            pi[s] = np.random.choice(best_actions)

            delta = max(delta, abs(v - V[s]))
            newV[s] = v
        for i in range(101):
            V[i] = newV[i]
        print(pi)
        if delta < threshold:
            plt.figure()
            plt.plot(np.linspace(0, 100, 101), pi)
            plt.figure()
            plt.plot(np.linspace(0, 100, 101), V)
            plt.show()
            break


if __name__ == '__main__':
    value_iteration()
