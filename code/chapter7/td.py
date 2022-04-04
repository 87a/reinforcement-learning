#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:td.py
@time:2022/04/02
"""
import gym
import numpy as np
from walk import RandomWalkEnv


def nstep_on_policy_return(v, done, states, rewards):
    """
    n-step td for ex7.2
    Args:
        v:values
        done:whether the process is done
        states:array of states
        rewards:array of rewards

    Returns:value

    """
    assert len(states) - 1 == len(rewards)

    if not rewards:
        return 0 if done else v[states[0]]

    sub_return = nstep_on_policy_return(v, done, states[1:], rewards[1:])
    return rewards[0] + sub_return


def td_on_policy_prediction(env: gym.Env, policy, n: int, num_episodes: int, alpha: float = 0.5, tderr: bool = False):
    """
    n-step TD algorithm for on-policy value prediction per Chapter 7.1. Value function updates are
     calculated by summing TD errors per Exercise 7.2 (tderr=True) or with (7.2) (tderr=False).
    Args:
        env:environment created by gym
        policy:policy
        n:step
        num_episodes:num of episodes
        alpha:hyper parameter
        tderr:whether use td error or not

    Returns:

    """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    n_state, n_action = env.observation_space.n, env.action_space.n
    assert policy.shape == (n_state, n_action)

    # Initialization of value function
    v = np.ones([n_state], dtype=np.float) * 0.5

    history = []
    for episode in range(num_episodes):
        # Reset the environment and initialize n-step rewards and states
        state = env.reset()
        nstep_states = [state]
        nstep_rewards = []

        dv = np.zeros_like(v)

        done = False

        while nstep_rewards or not done:
            if not done:
                action = np.random.choice(n_action, p=policy[state])
                state, reward, done, info = env.step(action)

                nstep_rewards.append(reward)
                nstep_states.append(state)

                if len(nstep_rewards) < n:
                    continue
                assert len(nstep_states) - 1 == len(nstep_rewards) == n

            v_target = nstep_on_policy_return(v, done, nstep_states, nstep_rewards)

            if tderr:
                dv[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])
            else:
                v[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])

            del nstep_rewards[0]
            del nstep_states[0]

            v += dv
        history += [np.copy(v)]
    return history


if __name__ == '__main__':
    env = gym.make('RandomWalk-v0')
    policy = np.ones([env.observation_space.n, env.action_space.n], dtype=np.float)
    size = env.observation_space.n
    env.seed(7)
    n = 4
    # history0 = td_on_policy_prediction(env, policy, n=n, num_episodes=10,
                                       # alpha=0.1, tderr=False)
    history1 = td_on_policy_prediction(env, policy, n=n, num_episodes=10,
                                       alpha=0.1, tderr=True)
    # print(history0)
    print(history1)
