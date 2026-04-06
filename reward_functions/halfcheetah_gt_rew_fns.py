"""
Ground-truth reward for Gymnasium HalfCheetah-v5 matching the default env step reward.

reward = forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action**2).
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_halfcheetah_observation import MujocoHalfCheetahObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class HalfcheetahRewardFunction(RewardFunction):
    """HalfCheetah-v5 default: forward speed minus quadratic control cost."""

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

    def set_specific_reward(self, _reward_id):
        """API compatibility with other direct_policy_learning rollouts (unused)."""
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoHalfCheetahObservation), (
            "next_obs must be MujocoHalfCheetahObservation."
        )
        assert isinstance(prev_obs, MujocoHalfCheetahObservation), (
            "prev_obs must be MujocoHalfCheetahObservation."
        )
        x_velocity = (next_obs.qpos[0] - prev_obs.qpos[0]) / next_obs.dt
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(np.asarray(action)))
        reward = forward_reward - ctrl_cost
        return float(reward)

