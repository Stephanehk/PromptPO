"""
Ground-truth reward for Meta-World door-open-v3 (MT1).

Delegates to SawyerDoorEnvV3.evaluate_state so the scalar matches the env step reward
(reward_function_version v2 by default).
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from direct_policy_learning.observations.metaworld_door_open_v3_observation import (
    MetaWorldDoorOpenV3Observation,
)


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, obs):
        pass


class DoorOpenV3RewardFunction(RewardFunction):
    def set_specific_reward(self, _reward_id):
        pass

    def calculate_reward(self, prev_obs, action, next_obs):
        assert isinstance(next_obs, MetaWorldDoorOpenV3Observation), (
            "next_obs must be MetaWorldDoorOpenV3Observation."
        )
        inner = next_obs._env.unwrapped
        obs = np.asarray(next_obs.obs_vector, dtype=np.float64)
        act = np.asarray(action, dtype=np.float32)
        r, _info = inner.evaluate_state(obs, act)
        return float(r)
