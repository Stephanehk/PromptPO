"""
NoiseWorld: a stochastic grid navigation MDP for direct_policy_learning.

The observation includes normalized agent and goal positions plus a flattened integer ``board``
(length n*n, row-major): one code per cell for wall, goal, negative terminal, three
free-cell dynamics categories, and optionally two prerequisite milestone codes (see ``BOARD_*``
in noise_world_board_layout). Optional trailing scalars record prerequisite visit progress.
Prompt text must not name what each code means if the project chooses to hide semantics from the LLM.

Assumptions:
- n >= 1. Gymnasium reset/step API.
- Grid layout is fixed for the lifetime of an environment instance; use a different ``seed``
  in ``__init__`` for a new board.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from direct_policy_learning.reward_functions.noise_world_gt_rew_fns import (
    compute_noise_world_step_reward,
)
from direct_policy_learning.env_utils.noise_world_board_layout import (
    BOARD_MAX_CODE,
    BOARD_PREREQ_FIRST,
    BOARD_PREREQ_SECOND,
    CELL_TYPE_DETERMINISTIC,
    CELL_TYPE_STOCH_STAY,
    CELL_TYPE_STOCH_RESET,
    build_initial_layout,
    ensure_minimum_standard_board_codes,
    inject_prerequisite_pair,
)


ACTION_DELTA = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


class NoiseWorldEnv(gym.Env):
    """
    Grid world with stochastic dynamics. The observation vector is normalized positions
    (four scalars in [0, 1]) concatenated with ``board`` (n*n scalars, integer codes 0..BOARD_MAX_CODE
    as float32) in row-major order; with ``prerequisite_pair``, two extra scalars in {0,1} follow.

    Observation: Box with mixed bounds — first four dims [0, 1]; board cells [0, BOARD_MAX_CODE].

    Action: Discrete(4) — up, down, left, right.

    Step reward, termination, and horizon follow the project spec for NoiseWorld
    (per-step cost, bonuses/penalties on special events, max length n**2).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n=100,
        seed=None,
        prerequisite_pair=False,
        ensure_all_standard_board_codes=False,
    ):
        assert isinstance(n, int) and n >= 1, "n must be a positive integer."
        self.n = n
        self._rng = np.random.default_rng(seed)
        self._prerequisite_pair = bool(prerequisite_pair)
        self._ensure_all_standard_board_codes = bool(ensure_all_standard_board_codes)

        self.start = (0, 0)
        self.goal = (n - 1, n - 1)

        self._board_flat = None
        self._prereq_phase = 0

        base_board = 4 + n * n
        obs_dim = base_board + (2 if self._prerequisite_pair else 0)
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        high[4:4 + n * n] = float(BOARD_MAX_CODE)
        if self._prerequisite_pair:
            high[4 + n * n :] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.horizon = n**2

        self._region_id = None
        self._cell_type = None
        self._p_succ = None
        self._p_alt = None
        self._walls = None
        self._bad_cells = None
        self._pos = None
        self._steps = None

        self._build_layout()

    def _cell_params_at(self, r, c):
        rid = int(self._region_id[r, c])
        return int(self._cell_type[rid]), float(self._p_succ[rid]), float(self._p_alt[rid])

    def _build_layout(self):
        """
        Sample regions, walls, and negative terminals. Validates a start-to-goal path.

        Assumptions:
        - n >= 2 for nontrivial navigation; n == 1 is degenerate.
        """
        state = build_initial_layout(self.n, self._rng)
        if self._ensure_all_standard_board_codes:
            state = ensure_minimum_standard_board_codes(self.n, state, self._rng)
        self._region_id = state["region_id"]
        self._cell_type = state["cell_type"]
        self._p_succ = state["p_succ"]
        self._p_alt = state["p_alt"]
        self._walls = state["walls"]
        self._bad_cells = state["bad_cells"]
        self._board_flat = state["board_flat"]
        if self._prerequisite_pair:
            self._board_flat, _, _ = inject_prerequisite_pair(
                self.n,
                self._board_flat,
                self._walls,
                self._bad_cells,
                self.goal,
                self.start,
                self._rng,
            )

    def _norm(self, row, col):
        if self.n == 1:
            return 0.0, 0.0
        d = float(self.n - 1)
        return float(row) / d, float(col) / d

    def _obs_vector(self, row, col):
        gr, gc = self.goal
        ar, ac = self._norm(row, col)
        gnr, gnc = self._norm(gr, gc)
        base = np.array([ar, ac, gnr, gnc], dtype=np.float32)
        out = np.concatenate([base, self._board_flat])
        if self._prerequisite_pair:
            f1 = 1.0 if self._prereq_phase >= 1 else 0.0
            f2 = 1.0 if self._prereq_phase >= 2 else 0.0
            out = np.concatenate(
                [out, np.array([f1, f2], dtype=np.float32)]
            )
        return out

    def _transition_cell_params(self, r, c):
        """
        Stochastic parameters for the cell the agent is leaving (before moving).

        Prerequisite milestone cells use deterministic dynamics.

        Assumptions:
        - (r, c) is a valid in-grid cell.
        """
        idx = r * self.n + c
        code = float(self._board_flat[idx])
        if code == float(BOARD_PREREQ_FIRST) or code == float(BOARD_PREREQ_SECOND):
            return CELL_TYPE_DETERMINISTIC, 1.0, 1.0
        return self._cell_params_at(r, c)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._pos = self.start
        self._steps = 0
        self._prereq_phase = 0
        obs = self._obs_vector(self._pos[0], self._pos[1])
        info = {
            "pos": self._pos,
            "goal": self.goal,
            "n": self.n,
            "horizon": self.horizon,
        }
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action."
        assert self._pos is not None, "Call reset() before step()."

        n = self.n
        self._steps += 1
        r, c = self._pos

        dr, dc = ACTION_DELTA[int(action)]
        nr, nc = r + dr, c + dc
        blocked = (
            nr < 0
            or nr >= n
            or nc < 0
            or nc >= n
            or (nr, nc) in self._walls
        )

        if blocked:
            next_r, next_c = r, c
        else:
            ctype, p, p_prime = self._transition_cell_params(r, c)
            if ctype == CELL_TYPE_DETERMINISTIC:
                next_r, next_c = nr, nc
            elif ctype == CELL_TYPE_STOCH_STAY:
                if self._rng.random() < p:
                    next_r, next_c = nr, nc
                else:
                    next_r, next_c = r, c
            elif ctype == CELL_TYPE_STOCH_RESET:
                w_move = p
                w_reset = 1.0 - p_prime
                s = w_move + w_reset
                assert s > 1e-9, "p and p' must yield positive total transition weight."
                if self._rng.random() < w_move / s:
                    next_r, next_c = nr, nc
                else:
                    next_r, next_c = self.start[0], self.start[1]
            else:
                assert False, "Unknown cell type."

        self._pos = (next_r, next_c)
        if self._prerequisite_pair:
            idx = next_r * n + next_c
            code = float(self._board_flat[idx])
            if code == float(BOARD_PREREQ_FIRST) and self._prereq_phase == 0:
                self._prereq_phase = 1
            elif code == float(BOARD_PREREQ_SECOND) and self._prereq_phase == 1:
                self._prereq_phase = 2

        allow_goal_bonus = True
        if self._prerequisite_pair and self._pos == self.goal:
            allow_goal_bonus = self._prereq_phase >= 2

        reward = compute_noise_world_step_reward(
            self._pos,
            self.n,
            self.goal,
            self._bad_cells,
            allow_goal_bonus=allow_goal_bonus,
        )
        terminated = False
        truncated = False

        if self._pos == self.goal:
            terminated = True
        elif self._pos in self._bad_cells:
            terminated = True
        elif self._steps >= self.horizon:
            truncated = True

        obs = self._obs_vector(self._pos[0], self._pos[1])
        info = {
            "pos": self._pos,
            "goal": self.goal,
            "steps": self._steps,
        }
        return obs, float(reward), terminated, truncated, info


_NOISE_WORLD_REGISTERED = False


def register_noise_world():
    """
    Register NoiseWorld-v0 with Gymnasium if not already registered.

    Assumptions:
    - Safe to call multiple times.
    """
    global _NOISE_WORLD_REGISTERED
    if _NOISE_WORLD_REGISTERED:
        return
    gym.register(
        id="NoiseWorld-v0",
        entry_point="direct_policy_learning.no_prior_envs.noise_world_env:NoiseWorldEnv",
        max_episode_steps=None,
    )
    _NOISE_WORLD_REGISTERED = True


register_noise_world()
