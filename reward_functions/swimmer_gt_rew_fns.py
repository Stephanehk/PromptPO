"""
Ground-truth reward for Gymnasium Swimmer-v5: forward x-velocity minus ctrl cost.

Matches swimmer_v5.py _get_rew.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_swimmer_observation import MujocoSwimmerObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class SwimmerRewardFunction(RewardFunction):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoSwimmerObservation), (
            "next_obs must be MujocoSwimmerObservation."
        )
        assert isinstance(prev_obs, MujocoSwimmerObservation), (
            "prev_obs must be MujocoSwimmerObservation."
        )
        x_velocity = (next_obs.qpos[0] - prev_obs.qpos[0]) / next_obs.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(np.asarray(action)))
        reward = forward_reward - ctrl_cost
        return float(reward)
