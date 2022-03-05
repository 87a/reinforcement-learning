#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:ex2.5.py
@time:2021/11/09
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # tqdm is a progress bar
from estimators import SampleAverageEstimator, WeightedEstimator
from testbed import K_armed_testbed

np.random.seed(250)


def plot_performance(estimator_names, rewards, action_optimality):
    for i, estimator_name in enumerate(estimator_names):
        average_run_rewards = np.average(rewards[i], axis=0)
        plt.plot(average_run_rewards, label=estimator_name)

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()

    for i, estimator_name in enumerate(estimator_names):
        average_run_optimality = np.average(action_optimality[i], axis=0)
        plt.plot(average_run_optimality, label=estimator_name)

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.show()


if __name__ == '__main__':
    # hyper-parameters
    K = 10
    N_STEPS = 10000
    N_RUNS = 2000
    N_ESTIMATORS = 2

    # initialize reward and optimal selection matrix
    rewards = np.full((N_ESTIMATORS, N_RUNS, N_STEPS), fill_value=0.)
    optimal_selections = np.full((N_ESTIMATORS, N_RUNS, N_STEPS), fill_value=0.)

    for run_i in tqdm(range(N_RUNS)):

        testbed = K_armed_testbed(k_actions=K)  # initialize testbed

        action_value_estimates = np.full(K, fill_value=0.0)  # initialize action-value estimate matrix
        # initialize estimators
        sample_average_estimator = SampleAverageEstimator(action_value_estimates.copy(), epsilon=0.1)
        weighted_estimator = WeightedEstimator(action_value_estimates.copy(), epsilon=0.1, alpha=0.1)

        # put estimators in a list
        estimators = [sample_average_estimator, weighted_estimator]

        for step_i in range(N_STEPS):
            for estimator_i, estimator in enumerate(estimators):
                action_selected = estimator.select_action()  # select a action
                is_optimal = testbed.is_optimal_action(action_selected)  # know if it's optimal
                reward = testbed.sample_action(action_selected)  # get the reward
                estimator.update_estimates(action_selected, reward)  # update action-value

                rewards[estimator_i][run_i][step_i] = reward  # update reward
                optimal_selections[estimator_i][run_i][step_i] = is_optimal  # update optimal selections

            testbed.random_walk_action_values()  # random walk of testbed

    plot_performance(["Ɛ=0.1", "Ɛ=0.1 α=0.1"], np.array(rewards), np.array(optimal_selections))
    print(rewards)
    print(optimal_selections)
