#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:example 4.1.py
@time:2022/03/08
"""
S = [(i, j) for i in range(1, 5) for j in range(1, 5)]
V = {s: 0 for s in S}
newV = {s: 0 for s in S}
actions = ['W', 'N', 'S', 'E']


def isEnd(state: tuple):
    if state == (1, 1) or state == (4, 4):
        return True

    return False


def get_next_state(state: tuple, action: str):
    assert len(state) == 2 and action in actions
    if action == "W":
        newstate = (state[0], max(state[1] - 1, 1))
    elif action == 'N':
        newstate = (max(state[0] - 1, 1), state[1])
    elif action == 'E':
        newstate = (state[0], min(state[1] + 1, 4))
    elif action == 'S':
        newstate = (min(state[0] + 1, 4), state[1])
    return newstate


def get_pi_p_r_v(state: tuple):
    results = []
    # pi, p, reward, v(s')
    for action in actions:
        next_state = get_next_state(state=state, action=action)
        r = -1
        results.append((0.25, 1, r, V[next_state]))
    return results


def value_evaluation(epochs: int = 10, thresh: float = 0.01):
    global V
    for epoch in range(epochs):
        print(f"epoch:{epoch}")
        delta = 0
        for s in S:
            if isEnd(s):
                continue
            results = get_pi_p_r_v(s)
            V_ = 0
            for result in results:
                V_ += result[0] * result[1] * (result[2] + result[3])
            newV[s] = round(V_, 2)
            delta = max(delta, abs(V[s] - V_))
        for key in V.keys():
            V[key] = newV[key]
        print_v(V)
        if delta < thresh:
            break



def print_v(V: dict):
    count = 0
    for key in V.keys():
        print('{0:7}'.format(V[key]), end='')
        count += 1
        if count % 4 == 0:
            print("\n")


if __name__ == '__main__':
    # print(S)
    # print(V)
    value_evaluation(epochs=1000, thresh=1e-2)
