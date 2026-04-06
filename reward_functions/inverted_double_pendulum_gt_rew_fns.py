"""
Ground-truth reward for Gymnasium InvertedDoublePendulum-v5.

Matches inverted_double_pendulum_v5.py _get_rew: alive bonus minus distance and velocity penalties.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_inverted_double_pendulum_observation import (
    MujocoInvertedDoublePendulumObservation,
)


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class InvertedDoublePendulumRewardFunction(RewardFunction):
    def __init__(self, healthy_reward=10.0):
        self._healthy_reward = healthy_reward

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoInvertedDoublePendulumObservation), (
            "next_obs must be MujocoInvertedDoublePendulumObservation."
        )
        assert isinstance(prev_obs, MujocoInvertedDoublePendulumObservation), (
            "prev_obs must be MujocoInvertedDoublePendulumObservation."
        )
        tip = next_obs.tip_site_pos
        x = float(tip[0])
        y_tip = float(tip[2])
        v1 = float(next_obs.qvel[1])
        v2 = float(next_obs.qvel[2])
        terminated = y_tip <= 1.0
        dist_penalty = 0.01 * x * x + (y_tip - 2.0) ** 2
        vel_penalty = 1e-3 * v1 * v1 + 5e-3 * v2 * v2
        alive_bonus = self._healthy_reward * int(not terminated)
        reward = alive_bonus - dist_penalty - vel_penalty
        return float(reward)
