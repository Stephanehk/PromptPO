"""
Ground-truth reward for NoiseWorld.

Single source of truth: ``compute_noise_world_step_reward`` is used by the environment
``step`` and by ``NoiseWorldRewardFunction.calculate_reward`` so rollouts match env reward.

``calculate_reward`` maps normalized observations back to grid cells; it reconstructs the
goal and negative terminals from ``env_name`` via the same (n, seed) as
``noise_world_env_names`` (aligned with the ``board`` layout in the observation vector).

Assumptions:
- Grid indices are integers; ``next_pos`` is ``(row, col)``.
- ``bad_cells`` is a set of ``(row, col)``; ``goal`` is ``(row, col)``.
- set_specific_reward ids are ignored (API compatibility with other GT modules).
"""

from abc import ABCMeta, abstractmethod

from direct_policy_learning.observations.noise_world_observation import NoiseWorldObservation


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs, action, next_obs):
        pass


def compute_noise_world_step_reward(
    next_pos, n, goal, bad_cells, allow_goal_bonus=True
):
    """
    Scalar reward after moving to ``next_pos``.

    Inputs:
    - next_pos: (row, col) after the transition.
    - n: grid side length.
    - goal: (row, col) terminal goal.
    - bad_cells: set of (row, col) negative terminals.
    - allow_goal_bonus: if False, the large positive term for reaching ``goal`` is omitted
      (used when prerequisite milestones are not yet satisfied).

    Assumptions:
    - goal is not in bad_cells; at most one of goal / bad applies per step.
    """
    reward = -1.0
    if next_pos == goal:
        if allow_goal_bonus:
            reward += float(n**3)
    elif next_pos in bad_cells:
        reward -= float(n**5)
    return reward


def denormalize_agent_cell(obs, n):
    """
    Map ``NoiseWorldObservation`` agent coordinates to integer grid cell.

    Inputs:
    - obs: NoiseWorldObservation with agent_row and agent_col in [0, 1].
    - n: grid side length.

    Outputs:
    - (row, col) with entries in [0, n - 1].

    Assumptions:
    - obs has been filled from the environment (floats match env normalization).
    """
    assert isinstance(obs, NoiseWorldObservation), "prev_obs/next_obs must be NoiseWorldObservation."
    if n == 1:
        return 0, 0
    r = int(round(obs.agent_row * float(n - 1)))
    c = int(round(obs.agent_col * float(n - 1)))
    if r < 0:
        r = 0
    if r > n - 1:
        r = n - 1
    if c < 0:
        c = 0
    if c > n - 1:
        c = n - 1
    return r, c


class NoiseWorldRewardFunction(RewardFunction):
    """
    GT reward aligned with ``NoiseWorldEnv`` step reward.

    Inputs:
    - env_name: a ``noise_world_board_*`` id so (n, seed) match the rollout environment.

    Assumptions:
    - ``calculate_reward`` is only used with boards constructed from the same ``env_name``
      as this instance (same layout as ``NoiseWorldEnv(n, seed)`` from
      ``noise_world_env_names``).
    """

    def __init__(self, env_name):
        assert env_name is not None, "env_name is required for NoiseWorldRewardFunction."
        self._env_name = env_name
        self._layout_cache = None

    def set_specific_reward(self, _reward_id):
        pass

    def _layout(self):
        """
        Lazily build a reference environment to read goal and bad_cells for this board.

        Assumptions:
        - ``NoiseWorldEnv(n, seed)`` with n, seed from ``noise_world_env_names`` matches
          the rollout env instance for this env_name.
        """
        if self._layout_cache is None:
            from direct_policy_learning.env_utils.noise_world_env_names import (
                noise_world_ensure_full_standard_board_codes,
                noise_world_grid_n,
                noise_world_prerequisite_pair,
                noise_world_seed,
            )
            from direct_policy_learning.no_prior_envs.noise_world_env import NoiseWorldEnv

            n = noise_world_grid_n(self._env_name)
            seed = noise_world_seed(self._env_name)
            ref = NoiseWorldEnv(
                n=n,
                seed=seed,
                prerequisite_pair=noise_world_prerequisite_pair(self._env_name),
                ensure_all_standard_board_codes=noise_world_ensure_full_standard_board_codes(
                    self._env_name
                ),
            )
            self._layout_cache = {
                "n": n,
                "goal": ref.goal,
                "bad_cells": ref._bad_cells,
            }
        return self._layout_cache

    def calculate_reward(self, prev_obs, action, next_obs):
        """
        Match ``compute_noise_world_step_reward`` on the grid cell implied by ``next_obs``.

        Assumptions:
        - prev_obs and next_obs are NoiseWorldObservation instances; only next_obs is used
          for terminal bonuses (same as env, which keys off post-transition state).
        """
        from direct_policy_learning.env_utils.noise_world_env_names import (
            noise_world_prerequisite_pair,
        )

        layout = self._layout()
        n = layout["n"]
        next_pos = denormalize_agent_cell(next_obs, n)
        allow_goal_bonus = True
        if noise_world_prerequisite_pair(self._env_name):
            first_done = float(next_obs.prereq_first_done) >= 0.5
            second_done = float(next_obs.prereq_second_done) >= 0.5
            if next_pos == layout["goal"]:
                allow_goal_bonus = first_done and second_done
        return compute_noise_world_step_reward(
            next_pos,
            n,
            layout["goal"],
            layout["bad_cells"],
            allow_goal_bonus=allow_goal_bonus,
        )
