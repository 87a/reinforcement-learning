#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:ex4.9.py
@time:2021/12/13
"""
import matplotlib
import matplotlib.pyplot as plt
from mdp import value_iteration
from gambler import GamblersProblem


def example():
    mdp = GamblersProblem(p_win=0.5)
    mdp.reset()
    capitals = []
    done = False
    while not done:
        bet = 1
        capital, reward, done, info = mdp.step(bet)
        capitals += [capital]

    plt.figure()
    plt.title("Example Episode")
    plt.xlabel("Timestep")
    plt.ylabel("Capital")
    plt.plot(capitals)
    plt.savefig("D:\\RL\\images\\Example Episode")


def plot_results(value, policy):
    if value is not None:
        plt.figure()
        plt.title("Value Function")
        plt.xlabel("Capital")
        plt.ylabel("Win Probability")
        plt.bar(range(len(value)), value)
        plt.savefig("D:\\RL\\images\\Value Function.png")

    if policy is not None:
        plt.figure()
        plt.title("Policy")
        plt.xlabel("Capital")
        plt.ylabel("Bet")
        plt.bar(range(len(policy)), policy + 1)
        plt.savefig("D:\\RL\\images\\Policy.png")


if __name__ == '__main__':
    matplotlib.rcParams['figure.figsize'] = [10, 5]
    mdp = GamblersProblem(p_win=0.4)
    value, policy = value_iteration(mdp)
    for i in range(len(policy)):
        print(i, policy[i] + 1)
    plot_results(value, policy)
