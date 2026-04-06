"""
Ground-truth reward for Gymnasium Humanoid-v5.

reward = forward + healthy - ctrl - contact (humanoid_v5.py _get_rew).
Ctrl uses sum of squares of data.ctrl; contact uses sum of squares of cfrc_ext (clipped).
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_humanoid_observation import MujocoHumanoidObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class HumanoidRewardFunction(RewardFunction):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        contact_cost_range=(-np.inf, 10.0),
        healthy_reward=5.0,
        healthy_z_range=(1.0, 2.0),
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoHumanoidObservation), (
            "next_obs must be MujocoHumanoidObservation."
        )
        assert isinstance(prev_obs, MujocoHumanoidObservation), (
            "prev_obs must be MujocoHumanoidObservation."
        )
        x_velocity = (
            next_obs.mass_center_xy[0] - prev_obs.mass_center_xy[0]
        ) / next_obs.dt
        forward_reward = self._forward_reward_weight * x_velocity

        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < next_obs.qpos[2] < max_z
        healthy_reward = float(is_healthy) * self._healthy_reward

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(next_obs.ctrl))

        raw_contact = self._contact_cost_weight * np.sum(
            np.square(np.asarray(next_obs.cfrc_ext, dtype=np.float64))
        )
        lo, hi = self._contact_cost_range
        contact_cost = float(np.clip(raw_contact, lo, hi))

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        return float(reward)
