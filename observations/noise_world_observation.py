"""
Observation bundle for NoiseWorld: agent and goal positions plus a full discrete board layout.

The environment observation vector has length 4 + n * n, or 6 + n * n when the board uses
prerequisite milestones (two extra scalars at the end; see env_context for those boards):
- Four scalars: agent (row, col) and goal (row, col), each normalized to [0, 1].
- Then ``board``: n*n entries, row-major (cell (r, c) -> index r * n + c). Each entry is a
  non-negative integer label encoded as float64 on this object (float32 in the raw env vector).
  Label semantics are defined in ``NoiseWorldEnv`` (not in the prompt context files).
- Optional trailing: ``prereq_first_done``, ``prereq_second_done`` in {0, 1} when applicable.

Assumptions:
- Raw env observations match ``NoiseWorldEnv`` observation_space.
- ``fill_from_env`` infers ``n`` from the vector length so it works when ``env`` is a Gymnasium
  wrapper (e.g. ``OrderEnforcing``) that does not expose ``.n``; ``env`` is still required for API
  consistency with other observation classes.
"""

import math

import numpy as np


def _grid_n_from_noise_world_obs_len(length):
    """
    Return (grid side n, has_prerequisite_trailing) given raw vector length.

    Length is either 4 + n*n (standard) or 6 + n*n (prerequisite boards with two flags).

    Assumptions:
    - length is consistent with one of the two layouts above.
    """
    assert length >= 4, "NoiseWorld observation length must be at least 4."
    nn4 = length - 4
    n4 = math.isqrt(nn4)
    if n4 * n4 == nn4:
        return n4, False
    nn6 = length - 6
    n6 = math.isqrt(nn6)
    if n6 * n6 == nn6:
        return n6, True
    assert False, (
        "NoiseWorld observation length must be 4+n*n or 6+n*n; got length %s." % length
    )


class NoiseWorldObservation:
    """
    NoiseWorld observation for policies, Feedback, and rollout wrappers.

    Position fields (row/col as floats in [0, 1]):
    - agent_row, agent_col — normalized agent coordinates
    - goal_row, goal_col — normalized goal coordinates

    Layout (fixed for the env instance; same at every step):
    - board — 1d ndarray of length n*n, one integer label per cell (float-stored)

    Prerequisite boards only (two extra floats after ``board``):
    - prereq_first_done, prereq_second_done — each in {0.0, 1.0} when present

    Action space: Discrete(4) — indices 0..3 (see policy context for semantics).
    """

    def __init__(self):
        self.agent_row = None
        self.agent_col = None
        self.goal_row = None
        self.goal_col = None
        self.board = None
        self.prereq_first_done = 0.0
        self.prereq_second_done = 0.0

    def fill_from_env(self, env, obs):
        """
        Populate fields from the environment's observation array.

        Inputs:
        - env: Gym env instance (may be wrapped); not used to parse ``obs`` beyond API parity.
        - obs: ndarray of length 4 + n * n or 6 + n * n (prerequisite boards).

        Assumptions:
        - obs matches the concatenation produced by ``NoiseWorldEnv._obs_vector``.
        """
        v = np.asarray(obs, dtype=np.float64).reshape(-1)
        n, has_prereq = _grid_n_from_noise_world_obs_len(int(v.shape[0]))
        nn = n * n
        self.agent_row = float(v[0])
        self.agent_col = float(v[1])
        self.goal_row = float(v[2])
        self.goal_col = float(v[3])
        self.board = np.asarray(v[4 : 4 + nn], dtype=np.float64).copy()
        if has_prereq:
            self.prereq_first_done = float(v[-2])
            self.prereq_second_done = float(v[-1])
        else:
            self.prereq_first_done = 0.0
            self.prereq_second_done = 0.0
