"""
Ground-truth reward for Gymnasium-Robotics Point Maze (sparse only).

Matches gymnasium_robotics MazeEnv.compute_reward for reward_type="sparse" (maze_v4.py):
1.0 if ||achieved_goal - desired_goal|| <= 0.45 else 0.0 per step.

Public docs sometimes mention 0.5 m; the library uses 0.45 for sparse, termination,
and info["success"].
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.point_maze_large_v3_observation import (
    PointMazeLargeV3Observation,
)


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class PointMazeLargeV3RewardFunction(RewardFunction):
    """
    Sparse reward for achieved vs desired goal.

    Assumptions:
    - prev_obs and next_obs are PointMazeLargeV3Observation instances.
    - Reward is computed from next_obs (post-transition), like env step reward.
    """

    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, PointMazeLargeV3Observation), (
            "next_obs must be PointMazeLargeV3Observation."
        )
        assert isinstance(prev_obs, PointMazeLargeV3Observation), (
            "prev_obs must be PointMazeLargeV3Observation."
        )
        achieved = np.asarray(next_obs.achieved_goal, dtype=np.float64)
        desired = np.asarray(next_obs.desired_goal, dtype=np.float64)
        distance = float(np.linalg.norm(achieved - desired))
        return 1.0 if distance <= 0.45 else 0.0
