#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:test.py
@time:2021/11/12
"""
import numpy as np
import scipy.stats

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    requests_lambda = np.array([3, 4])
    returns_lambda = np.array([3, 2])
    dist_requests = scipy.stats.poisson(requests_lambda)
    dist_returns = scipy.stats.poisson(returns_lambda)
    p_requests = dist_requests.pmf(5)
    p_returns = dist_returns.pmf(4)
    n_requests_returns = np.indices([15] * 4).reshape([4, -1]).T
    print(n_requests_returns)
