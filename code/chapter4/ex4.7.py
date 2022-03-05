#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:ex4.7.py
@time:2021/12/11
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from mdp import policy_iteration
from Jacks_car_rental import JacksCarRental, NonlinearJacksCarRental


def plot_year_and_hist():
    mdp = JacksCarRental()

    days = 365
    rewards = np.zeros(days, dtype=np.int)
    actions = np.zeros(days, dtype=np.int)
    states = np.zeros([days, len(mdp.observation_space.nvec)], dtype=np.int)

    mdp.reset()
    for day in range(days):
        action = mdp.action_space.sample()
        state, reward, done, info = mdp.step(action)
        rewards[day] = reward
        actions[day] = action
        states[day] = state

    matplotlib.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    plt.title("Jack's Random Year")
    plt.ylabel("Number of Cars")
    plt.xlabel("Days")
    z0 = plt.bar(range(days), states[:, 0], label="Location 1")
    z1 = plt.bar(range(days), states[:, 1], label="Location 2")
    plt.legend()
    plt.savefig("D:\\RL\\images\\Jack's Random Year.png")
    plt.close()
    plt.figure()
    plt.title("Histogram of Revenue vs Transfer")
    plt.ylabel("Revenue")
    plt.xlabel("Transfer (1 -> 2)")
    z = plt.hist2d(actions - mdp.max_transfer, rewards, bins=[mdp.action_space.n - 1, 6],
                   range=[[-mdp.max_transfer, +mdp.max_transfer], [-60, +60]])
    plt.colorbar()
    plt.savefig("D:\\RL\\images\\Histogram of Revenue vs Transfer.png")
    plt.close()


def performance_benchmark():
    mdp = JacksCarRental(max_poisson=30)
    start = time.perf_counter()
    mdp.step(mdp.action_space.sample())
    end = time.perf_counter()
    print(end - start)

    mdp = JacksCarRental(max_poisson=20)
    start = time.perf_counter()
    mdp.step(mdp.action_space.sample())
    end = time.perf_counter()
    print(end - start)

    mdp = JacksCarRental(max_poisson=10)
    start = time.perf_counter()
    mdp.step(mdp.action_space.sample())
    end = time.perf_counter()
    print(end - start)


def plot_results(mdp, value, policy):
    if policy is not None:
        plt.figure()
        plt.title("Policy")
        plt.ylabel("Cars at Location 1")
        plt.xlabel("Cars at Location 2")
        plt.imshow(policy - mdp.max_transfer, origin='lower',
                   vmin=-mdp.max_transfer, vmax=+mdp.max_transfer)
        plt.colorbar()
        # plt.savefig("D:/RL/images/figure4.2(1).png")
        plt.savefig("D:/RL/images/ex4.7(1).png")
    if value is not None:
        plt.figure()
        plt.title("Value Function")
        plt.ylabel("Cars at Location 1")
        plt.xlabel("Cars at Location 2")
        plt.imshow(value, origin='lower')
        plt.colorbar()
        # plt.savefig("D:/RL/images/figure4.2(2).png")
        plt.savefig("D:/RL/images/ex4.7(2).png")


def plot_figure_4_2():
    mdp = JacksCarRental()
    value, policy = policy_iteration(mdp)
    plot_results(mdp, value, policy)


def ex_4_7():
    mdp = NonlinearJacksCarRental()
    value, policy = policy_iteration(mdp)
    plot_results(mdp, value, policy)


if __name__ == '__main__':
    np.random.seed(0)
    # performance_benchmark()
    # plot_figure_4_2()
    ex_4_7()
