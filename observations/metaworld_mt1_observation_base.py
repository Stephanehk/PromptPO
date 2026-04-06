"""
Shared observation wrapper for Meta-World MT1 (Sawyer v3) tasks.

Meta-World builds flat observations in SawyerXYZEnv._get_obs (39 elements by default):
current proprio + object features, previous-step copy, then goal position. See Farama
Metaworld sawyer_xyz_env.py and per-task env modules.

Ground-truth reward in this repo delegates to the live env's evaluate_state(obs, action),
so each observation keeps a reference to the Gymnasium env that produced it (same pattern
as needing simulator state for complex MuJoCo rewards).

Assumptions:
- env is the Gymnasium env returned by gym.make("Meta-World/MT1", ...) (possibly wrapped);
  env.unwrapped is the concrete Sawyer*EnvV3 with evaluate_state.
- obs_np is the vector returned by env reset/step for this timestep.

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MetaWorldMT1ObservationBase:
    """
    Base bundle: flat observation vector plus env handle for evaluate_state-based GT reward.

    Policies should only use obs_vector (and optional helpers you add in subclasses);
    _env is for rollout / reward code.
    """

    def __init__(self):
        self.obs_vector = None
        self._env = None

    def fill_from_env(self, env, obs_np):
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self._env = env
