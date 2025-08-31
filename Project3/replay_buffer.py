from typing import Any
import torch
import numpy as np
import random


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedExperienceReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.001,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 0.01
        self.device = device

    def _get_priority(self, error: torch.Tensor) -> float:
        return (error.abs().item() + self.epsilon) ** self.alpha

    def add(self, error: torch.Tensor, sample: tuple) -> None:
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.max_priority = max(self.max_priority, p)

    def sample(self, n: int) -> tuple[list[Any], list[int], torch.Tensor]:
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            while data is None:
                s = random.uniform(a, b)
                idx, p, data = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        is_weights = torch.from_numpy(is_weights).float().to(self.device)

        return batch, idxs, is_weights

    def update_priorities(self, idxs: list[int], errors: torch.Tensor) -> None:
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return self.tree.n_entries
