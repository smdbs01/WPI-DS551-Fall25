#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from typing import Any, Optional
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from replay_buffer import PrioritizedExperienceReplayBuffer

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.total_frames = args.total_frames

        self.batch_size = args.batch_size
        self.replay_start_size = args.replay_start_size
        self.train_freq = args.train_freq
        self.target_update_freq = args.target_update_freq

        self.q = DQN().to(device)
        self.target_q = DQN().to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optimizer = optim.RMSprop(
            self.q.parameters(), lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01
        )

        self.gamma = args.gamma

        self.device = device
        self.buffer = PrioritizedExperienceReplayBuffer(
            capacity=args.replay_buffer_size, device=self.device
        )

        self.epsilon = args.eps_start
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.frame_number = 0
        self.learn_step_counter = 0

        self.state = None
        self.current_episode_reward = 0
        self.episode_rewards = []

        if args.test_dqn:
            # you can load your model here
            print("loading trained model")
            ###########################
            # YOUR IMPLEMENTATION HERE #
            try:
                self.q.load_state_dict(torch.load("models/dqn_model.pth"))
                self.target_q.load_state_dict(torch.load("models/dqn_model.pth"))
            except Exception as e:
                print("Failed to load model:", e)

            self.epsilon = 0

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.q.eval()
        self.target_q.eval()
        self.current_episode_reward = 0
        ###########################

    def update_epsilon(self):
        if self.frame_number < self.replay_start_size:
            return
        if self.frame_number < self.eps_decay:
            self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * (
                self.frame_number / self.eps_decay
            )
        else:
            self.epsilon = self.eps_end

    def _epsilon_greedy(self, q_values: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        return int(torch.argmax(q_values).item())

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test and self.frame_number < self.replay_start_size:
            return np.random.randint(self.env.action_space.n)

        obs = np.transpose(observation, (2, 0, 1))
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(not test):
            q_values = self.q(obs)
            action = self._epsilon_greedy(q_values)

        ###########################
        return action

    def push(self, s, a, r, next_s, done):
        """You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        experience = (s, a, r, next_s, done)
        self.buffer.add(
            torch.Tensor([self.buffer.max_priority], device=self.device), experience
        )
        ###########################

    def replay_buffer(self) -> Optional[tuple[list[Any], list[int], torch.Tensor]]:
        """You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.buffer) < self.batch_size:
            return None
        ###########################
        return self.buffer.sample(self.batch_size)

    def _preprocess_batch(self, batch: list[tuple]):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(
            [torch.FloatTensor(np.transpose(s, (2, 0, 1))) for s in states]
        ).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(
            [torch.FloatTensor(np.transpose(s, (2, 0, 1))) for s in next_states]
        ).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def _compute_td_error(
        self, states, actions, rewards, next_states, dones
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the temporal difference (TD) error.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            TD errors, current Q values, target Q values
        """
        current_q_values = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (current_q_values - target_q_values).abs()

        return td_errors, current_q_values, target_q_values

    def train_step(self):
        batch_data = self.replay_buffer()
        if batch_data is None:
            return None

        batch, indices, is_weights = batch_data

        states, actions, rewards, next_states, dones = self._preprocess_batch(batch)

        td_errors, current_q_values, target_q_values = self._compute_td_error(
            states, actions, rewards, next_states, dones
        )

        loss = (
            is_weights
            * F.smooth_l1_loss(
                current_q_values, target_q_values.detach(), reduction="none"
            )
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)

        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors.detach())

        return loss.item()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        print("Training...")

        episode = 0
        best_mean_reward = -float("inf")

        self.state = self.env.reset()
        self.current_episode_reward = 0

        while self.frame_number < self.total_frames:
            action = self.make_action(self.state, test=False)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.push(self.state, action, reward, next_state, done)

            self.state = next_state
            self.current_episode_reward += reward
            self.frame_number += 1

            self.update_epsilon()

            if self.frame_number == self.replay_start_size:
                print("Replay buffer filled, start training...")

            if (
                self.frame_number > self.replay_start_size
                and self.frame_number % self.train_freq == 0
            ):
                loss = self.train_step()

                self.learn_step_counter += 1

                if self.learn_step_counter % self.target_update_freq == 0:
                    self.target_q.load_state_dict(self.q.state_dict())

            if done:
                self.episode_rewards.append(self.current_episode_reward)

                if episode % 100 == 0 and episode > 0:
                    mean_reward = (
                        np.mean(self.episode_rewards[-100:])
                        if len(self.episode_rewards) >= 100
                        else np.mean(self.episode_rewards)
                    )
                    print(
                        f"Episode: {episode}, Frame: {self.frame_number}, Mean Reward: {mean_reward:.2f}, Epsilon: {self.epsilon:.4f}"
                    )

                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        torch.save(self.q.state_dict(), "models/dqn_model.pth")
                        print(
                            f"New best mean reward: {best_mean_reward:.2f}. Model saved."
                        )

                self.state = self.env.reset()
                self.current_episode_reward = 0
                episode += 1

        torch.save(self.q.state_dict(), "models/dqn_final_model.pth")
        print("Final model saved.")

        ###########################
