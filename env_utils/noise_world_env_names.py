"""
NoiseWorld board ids for direct_policy_learning.

Each id fixes a pair (n, seed) so the board is reproducible across runs. Observations include
a flattened per-cell integer ``board``; prompt files omit the meaning of each code unless you add it.

Assumptions:
- `direct_policy_learning.no_prior_envs.noise_world_env` is importable (registers NoiseWorld-v0).
"""

NOISE_WORLD_ENV_NAMES = (
    "noise_world_board_0",
    "noise_world_board_1",
    "noise_world_board_2",
    "noise_world_board_3",
    "noise_world_board_4",
    "noise_world_board_5",
    "noise_world_board_6",
)

# Boards 4–6: prerequisite pair (see ``noise_world_prerequisite_pair``); same n as 1–3.
_PREREQ_ENV_NAMES = frozenset(
    {
        "noise_world_board_4",
        "noise_world_board_5",
        "noise_world_board_6",
    }
)

# Fixed (grid size, rng seed) per evaluation board. Boards 1–3,4–6 share n=10; board_0 is 5x5.
_N_SEED = {
    "noise_world_board_0": (5, 710_000),
    "noise_world_board_1": (10, 710_001),
    "noise_world_board_2": (10, 710_002),
    "noise_world_board_3": (10, 717_003),
    "noise_world_board_4": (10, 710_004),
    "noise_world_board_5": (10, 710_005),
    "noise_world_board_6": (10, 713_006),
}

_POLICY_CLASS_BASE = {
    "noise_world_board_0": "NoiseWorldBoard0",
    "noise_world_board_1": "NoiseWorldBoard1",
    "noise_world_board_2": "NoiseWorldBoard2",
    "noise_world_board_3": "NoiseWorldBoard3",
    "noise_world_board_4": "NoiseWorldBoard4",
    "noise_world_board_5": "NoiseWorldBoard5",
    "noise_world_board_6": "NoiseWorldBoard6",
}

_REWARD_CLASS = {
    "noise_world_board_0": "NoiseWorldRewardFunction",
    "noise_world_board_1": "NoiseWorldRewardFunction",
    "noise_world_board_2": "NoiseWorldRewardFunction",
    "noise_world_board_3": "NoiseWorldRewardFunction",
    "noise_world_board_4": "NoiseWorldRewardFunction",
    "noise_world_board_5": "NoiseWorldRewardFunction",
    "noise_world_board_6": "NoiseWorldRewardFunction",
}

_OBS_CLASS = {
    "noise_world_board_0": "NoiseWorldObservation",
    "noise_world_board_1": "NoiseWorldObservation",
    "noise_world_board_2": "NoiseWorldObservation",
    "noise_world_board_3": "NoiseWorldObservation",
    "noise_world_board_4": "NoiseWorldObservation",
    "noise_world_board_5": "NoiseWorldObservation",
    "noise_world_board_6": "NoiseWorldObservation",
}

_REWARD_FN_BASENAME = {
    "noise_world_board_0": "noise_world_gt_rew_fns.py",
    "noise_world_board_1": "noise_world_gt_rew_fns.py",
    "noise_world_board_2": "noise_world_gt_rew_fns.py",
    "noise_world_board_3": "noise_world_gt_rew_fns.py",
    "noise_world_board_4": "noise_world_gt_rew_fns.py",
    "noise_world_board_5": "noise_world_gt_rew_fns.py",
    "noise_world_board_6": "noise_world_gt_rew_fns.py",
}


def is_noise_world_env_name(env_name):
    """
    Return True if env_name is a registered NoiseWorld board id.

    Assumptions:
    - env_name is a string or None.
    """
    if env_name is None or not isinstance(env_name, str):
        return False
    return env_name in NOISE_WORLD_ENV_NAMES


def noise_world_grid_n(env_name):
    """
    Grid dimension n for this board id.

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _N_SEED, "invalid noise world env_name: %r" % (env_name,)
    return _N_SEED[env_name][0]


def noise_world_seed(env_name):
    """
    Layout seed for this board id.

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _N_SEED, "invalid noise world env_name: %r" % (env_name,)
    return _N_SEED[env_name][1]


def noise_world_policy_class_base(env_name):
    """
    Base name for generated policy classes (e.g. NoiseWorldBoard1 -> NoiseWorldBoard1Policy).

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _POLICY_CLASS_BASE, "invalid noise world env_name: %r" % (env_name,)
    return _POLICY_CLASS_BASE[env_name]


def noise_world_reward_function_name(env_name):
    """
    Class name in reward_functions/noise_world_gt_rew_fns.py for prompts.

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _REWARD_CLASS, "invalid noise world env_name: %r" % (env_name,)
    return _REWARD_CLASS[env_name]


def noise_world_feedback_obs_class_name(env_name):
    """
    Observation class name for Feedback prompts (import in starter code).

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _OBS_CLASS, "invalid noise world env_name: %r" % (env_name,)
    return _OBS_CLASS[env_name]


def noise_world_reward_fn_basename(env_name):
    """
    Basename of the ground-truth reward module under direct_policy_learning/reward_functions/.

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    assert env_name in _REWARD_FN_BASENAME, "invalid noise world env_name: %r" % (env_name,)
    return _REWARD_FN_BASENAME[env_name]


def noise_world_max_episode_steps(env_name):
    """
    Max timesteps per episode for an n×n board: n**2.

    Assumptions:
    - is_noise_world_env_name(env_name) is True.
    """
    n = noise_world_grid_n(env_name)
    return n**2


def noise_world_prerequisite_pair(env_name):
    """
    Return True if this board uses two milestone cell types and extended observations.

    When True, ``NoiseWorldEnv`` is constructed with ``prerequisite_pair=True`` (two extra
    observation scalars and goal bonus only after visiting milestones in order).

    Assumptions:
    - env_name is registered in ``_N_SEED``.
    """
    assert env_name in _N_SEED, "invalid noise world env_name: %r" % (env_name,)
    return env_name in _PREREQ_ENV_NAMES


def noise_world_ensure_full_standard_board_codes(env_name):
    """
    Return True if the env should run ``ensure_minimum_standard_board_codes`` after sampling.

    When True, the flattened observation board is guaranteed to contain at least one cell of
    each code 0..5 (standard types without prerequisite milestones).

    Assumptions:
    - env_name is registered in ``_N_SEED``.
    """
    assert env_name in _N_SEED, "invalid noise world env_name: %r" % (env_name,)
    return env_name == "noise_world_board_3"
