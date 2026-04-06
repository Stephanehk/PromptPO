"""
Environments with no prior information leaked through prompts (layout/dynamics hidden).

NoiseWorld lives here; import `noise_world_env` to register the Gymnasium id.
"""

from direct_policy_learning.no_prior_envs.noise_world_env import (
    NoiseWorldEnv,
    register_noise_world,
)

__all__ = ["NoiseWorldEnv", "register_noise_world"]
