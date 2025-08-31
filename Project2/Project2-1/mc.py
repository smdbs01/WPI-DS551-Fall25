#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""

import numpy as np
import random
from collections import defaultdict

# -------------------------------------------------------------------------
"""
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
"""
# -------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    current_sum, _, _ = observation
    action = 0 if current_sum >= 20 else 1

    #                          #
    ############################
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #

    for _ in range(n_episodes):
        # Sample a trajectory
        episode = []
        state, _ = env.reset()
        first_visits = set()  # indexes of first-visit states in the episode
        visited = set()  # state
        n = 0

        while True:
            action = policy(state)
            new_state, r, terminated, truncated, _ = env.step(action)

            episode.append((state, action, r))
            if state not in visited:
                first_visits.add(n)
                visited.add(state)

            state = new_state

            n += 1

            if terminated or truncated:
                break

        G = 0.0
        for t in range(n - 1, -1, -1):
            s, _, r = episode[t]
            G = gamma * G + r
            if t in first_visits:
                # add to return sum
                returns_sum[s] += G
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]

    #                          #
    ############################

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state:
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    if np.random.random() < epsilon:
        return np.random.randint(nA)

    action = np.argmax(Q[state])

    #                          #
    ############################
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

    eps = epsilon

    for _ in range(n_episodes):
        # Sample a trajectory
        episode = []
        state, _ = env.reset()
        first_visits = set()  # indexes of first-visit states in the episode
        visited = set()  # state
        n = 0

        while True:
            action = epsilon_greedy(Q, state, 2, eps)  # env.action_space.n
            new_state, r, terminated, truncated, _ = env.step(action)

            episode.append((state, action, r))
            if (state, action) not in visited:
                first_visits.add(n)
                visited.add((state, action))

            state = new_state

            n += 1

            if terminated or truncated:
                break

        G = 0.0
        for t in range(n - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            if t in first_visits:
                sa = (s, a)
                # add to return sum
                returns_sum[sa] += G
                returns_count[sa] += 1
                Q[s][a] = returns_sum[sa] / returns_count[sa]

        eps = max(eps - 0.1 / n_episodes, 0)

    #                          #
    ############################

    return Q
