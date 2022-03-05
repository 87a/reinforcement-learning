#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:testbed.py
@time:2021/11/09
"""
import numpy as np


class K_armed_testbed:
    # initialize
    # k is the number of arms
    def __init__(self, k_actions):
        self.k = k_actions
        self.action_values = np.full(self.k, fill_value=0.0)

    def random_walk_action_values(self):
        increment = np.random.normal(loc=0, scale=0.01, size=self.k)
        self.action_values += increment

    def sample_action(self, action_i):
        #  get the reward
        return np.random.normal(loc=self.action_values[action_i], scale=1, size=1)[0]

    def get_optimal_action(self):
        return np.argmax(self.action_values)

    def get_optimal_action_value(self):
        return self.action_values[self.get_optimal_action()]

    def is_optimal_action(self, action_i):
        return float(self.get_optimal_action_value() == self.action_values[action_i])

    def __str__(self):
        return "\t".join(["A%d: %.2f" % (action_i, self.action_values[action_i]) for action_i in range(self.k)])


if __name__ == '__main__':
    pass
