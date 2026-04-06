"""
Ground-truth reward for Gymnasium Ant-v5 matching the default env step reward.

reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
(Gymnasium Ant-v5 step / _get_rew). See ant_v5.py in Gymnasium.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_ant_observation import MujocoAntObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class AntRewardFunction(RewardFunction):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range

    def set_specific_reward(self, _reward_id):
        """API compatibility with other direct_policy_learning rollouts (unused)."""
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoAntObservation), (
            "next_obs must be MujocoAntObservation."
        )
        assert isinstance(prev_obs, MujocoAntObservation), (
            "prev_obs must be MujocoAntObservation."
        )
        x_velocity = (
            next_obs.main_body_xy[0] - prev_obs.main_body_xy[0]
        ) / next_obs.dt
        forward_reward = self._forward_reward_weight * x_velocity

        state = next_obs.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        healthy_reward = float(is_healthy) * self._healthy_reward

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(np.asarray(action)))

        lo, hi = self._contact_force_range
        clipped = np.clip(np.asarray(next_obs.cfrc_ext, dtype=np.float64), lo, hi)
        contact_cost = self._contact_cost_weight * np.sum(np.square(clipped))

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        return float(reward)
