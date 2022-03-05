#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:mdp.py
@time:2021/12/10
"""
import abc
import gym
import numpy as np


class MDP(gym.Env, abc.ABC):
    gamma = None
    terminals = []

    def transitions(self, state, action):
        raise NotImplementedError()

    def step(self, action):
        transitions, rewards, ps, terminals = self.transitions(self.state, action)

        idx = np.random.choice(ps.size, p=ps)
        self.state = transitions[idx]
        return transitions[idx], rewards[idx], terminals[idx], {}

    def render(self, mode='ansi'):
        print(self.state)


def extract_policy(mdp, value, tolerance=1e-5):
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete

    policy = np.zeros(shape=mdp.observation_space.nvec, dtype=np.int)
    # nvec: vector of counts of each categorical variable
    for state in np.ndindex(*mdp.observation_space.nvec):
        # np.ndindex:An N-dimensional iterator object to index arrays.
        if state in mdp.terminals:
            policy[state] = 0
            continue
        expected_returns = []
        for action in np.arange(mdp.action_space.n):
            transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)
            returns = rewards + mdp.gamma * value[tuple(transitions.T)]
            terminal = np.where(terminals == 1)
            if terminal[0].size > 0:
                value[tuple(transitions[terminal].T)] = rewards[terminal]
                returns[terminal] = rewards[terminal]

            expected_returns += [np.sum(ps * returns)]

        policy[state] = np.argmax(expected_returns)

    return policy


def policy_iteration(mdp):
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete

    old_policy = -1
    value = np.zeros(shape=mdp.observation_space.nvec)
    policy = np.random.randint(mdp.action_space.n, size=value.shape)
    while (policy != old_policy).any():

        old_policy = np.copy(policy)
        for state in np.ndindex(*mdp.observation_space.nvec):

            if state in mdp.terminals:
                continue

            action = policy[state]
            transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)

            returns = rewards + mdp.gamma * value[tuple(transitions.T)]
            terminal = np.where(terminals == 1)
            if terminal[0].size > 0:
                value[tuple(transitions[terminal].T)] = rewards[terminal]
                returns[terminal] = rewards[terminal]

            expected_return = np.sum(ps * returns)
            value[state] = expected_return

        policy = extract_policy(mdp, value)

    return value, policy


def value_iteration(mdp, tolerance=1e-6):
    """ Find and return the optimal value function and optimal policy for the MDP.

    This function uses the following version of the value iteration algorithm:
        0. Set value function for all states to 0.
        1. For every state update the value function to:
              value[state] = optimal_reward + discount_factor * optimal_payoff
           Where optimal_payoff is the maximum of expected payoffs when taking
           every possible action from the current state.
        2. Goto 1. unless the maximal value function update is less than tolerance.
        3. Extract the final optimal policy from the optimal value function.
    """
    assert type(mdp.action_space) == gym.spaces.Discrete
    assert type(mdp.observation_space) == gym.spaces.MultiDiscrete

    old_value = 2 * tolerance
    value = np.zeros(shape=mdp.observation_space.nvec)
    while (np.abs(value - old_value) > tolerance).any():

        old_value = np.copy(value)
        for state in np.ndindex(*mdp.observation_space.nvec):
            # Calculate the update for value function for every non-terminal state
            if state in mdp.terminals:
                continue

            expected_returns = []
            for action in np.arange(mdp.action_space.n):
                # print(action)
                # Get all transitions and their associated rewards and probabilities, given state and action
                transitions, rewards, ps, terminals = mdp.transitions(np.array(state), action)

                # Returns non-terminal transitions
                returns = rewards + mdp.gamma * value[tuple(transitions.T)]
                # Returns from terminal transitions
                terminal = np.where(terminals == 1)
                if terminal[0].size > 0:
                    value[tuple(transitions[terminal].T)] = rewards[terminal]
                    returns[terminal] = rewards[terminal]

                # Expected returns
                expected_returns += [np.sum(ps * returns)]

            value[state] = np.max(expected_returns)

    # Extraction of the optimal policy from the optimal value function
    policy = extract_policy(mdp, value)
    return value, policy

