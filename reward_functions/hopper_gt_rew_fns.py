"""
Ground-truth reward for Gymnasium Hopper-v5 (forward + healthy - ctrl).

Matches hopper_v5.py _get_rew / is_healthy defaults.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_hopper_observation import MujocoHopperObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


def _hopper_is_healthy(qpos, qvel, healthy_state_range, healthy_z_range, healthy_angle_range):
    """Mirror HopperEnv.is_healthy with default ranges."""
    state = np.concatenate([qpos, qvel])
    z, angle = qpos[1], qpos[2]
    rest = state[2:]
    min_state, max_state = healthy_state_range
    healthy_state = np.all(np.logical_and(min_state < rest, rest < max_state))
    min_z, max_z = healthy_z_range
    healthy_z = min_z < z < max_z
    min_angle, max_angle = healthy_angle_range
    healthy_angle = min_angle < angle < max_angle
    return bool(healthy_state and healthy_z and healthy_angle)


class HopperRewardFunction(RewardFunction):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoHopperObservation), (
            "next_obs must be MujocoHopperObservation."
        )
        assert isinstance(prev_obs, MujocoHopperObservation), (
            "prev_obs must be MujocoHopperObservation."
        )
        x_velocity = (next_obs.qpos[0] - prev_obs.qpos[0]) / next_obs.dt
        forward_reward = self._forward_reward_weight * x_velocity

        is_healthy = _hopper_is_healthy(
            next_obs.qpos,
            next_obs.qvel,
            self._healthy_state_range,
            self._healthy_z_range,
            self._healthy_angle_range,
        )
        healthy_reward = float(is_healthy) * self._healthy_reward

        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(np.asarray(action)))
        reward = forward_reward + healthy_reward - ctrl_cost
        return float(reward)
