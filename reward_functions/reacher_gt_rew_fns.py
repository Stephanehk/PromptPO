"""
Ground-truth reward for Gymnasium Reacher-v5.

reward = -w_dist * ||fingertip - target|| - w_ctrl * ||action||^2 (reacher_v5.py _get_rew).
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.mujoco_reacher_observation import MujocoReacherObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class ReacherRewardFunction(RewardFunction):
    def __init__(
        self,
        reward_dist_weight=1.0,
        reward_control_weight=1.0,
    ):
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MujocoReacherObservation), (
            "next_obs must be MujocoReacherObservation."
        )
        assert isinstance(prev_obs, MujocoReacherObservation), (
            "prev_obs must be MujocoReacherObservation."
        )
        vec = np.asarray(next_obs.fingertip_minus_target, dtype=np.float64)
        reward_dist = -np.linalg.norm(vec) * self._reward_dist_weight
        reward_ctrl = -np.square(np.asarray(action)).sum() * self._reward_control_weight
        reward = reward_dist + reward_ctrl
        return float(reward)
