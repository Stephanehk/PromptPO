"""
Identifiers for Gymnasium-Robotics Point Maze v3 tasks used in direct_policy_learning.

env_name uses the pattern point_maze_<variant>_v3; the Gymnasium id is PointMaze_<Variant>-v3
(e.g. point_maze_large_v3 -> PointMaze_Large-v3; point_maze_large_diverse_g_v3 ->
PointMaze_Large_Diverse_G-v3). Rollouts pass a maze_map from point_maze_maps.py (custom for
Large, published LARGE_MAZE_DIVERSE_G for Large_Diverse_G).

Assumptions:
- gymnasium_robotics is installed and registered via gym.register_envs(gymnasium_robotics).
"""

POINT_MAZE_ENV_NAMES = (
    "point_maze_large_v3",
    "point_maze_large_diverse_g_v3",
)

_GYM_ID = {
    "point_maze_large_v3": "PointMaze_Large-v3",
    "point_maze_large_diverse_g_v3": "PointMaze_Large_Diverse_G-v3",
}

_POLICY_CLASS_BASE = {
    "point_maze_large_v3": "PointMazeLargeV3",
    "point_maze_large_diverse_g_v3": "PointMazeLargeDiverseGV3",
}

_REWARD_CLASS = {
    "point_maze_large_v3": "PointMazeLargeV3RewardFunction",
    "point_maze_large_diverse_g_v3": "PointMazeLargeDiverseGV3RewardFunction",
}

_OBS_CLASS = {
    "point_maze_large_v3": "PointMazeLargeV3Observation",
    "point_maze_large_diverse_g_v3": "PointMazeLargeDiverseGV3Observation",
}

_REWARD_FN_BASENAME = {
    "point_maze_large_v3": "point_maze_large_v3_gt_rew_fns.py",
    "point_maze_large_diverse_g_v3": "point_maze_large_diverse_g_v3_gt_rew_fns.py",
}

# Registered max_episode_steps for large Point Maze v3 tasks in gymnasium_robotics (800).
POINT_MAZE_LARGE_V3_MAX_EPISODE_STEPS = 800


def is_point_maze_env_name(env_name):
    """
    Return True if env_name is a supported Point Maze direct_policy_learning id.

    Assumptions:
    - env_name is a string or None.
    """
    if env_name is None or not isinstance(env_name, str):
        return False
    return env_name in POINT_MAZE_ENV_NAMES


def point_maze_gym_make_id(env_name):
    """
    Map env_name to Gymnasium registration id (e.g. point_maze_large_v3 -> PointMaze_Large-v3).

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    assert env_name in _GYM_ID, "invalid point maze env_name: %r" % (env_name,)
    return _GYM_ID[env_name]


def point_maze_policy_class_base(env_name):
    """
    Base name for generated policy classes (e.g. PointMazeLargeV3 -> PointMazeLargeV3Policy).

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    assert env_name in _POLICY_CLASS_BASE, "invalid point maze env_name: %r" % (env_name,)
    return _POLICY_CLASS_BASE[env_name]


def point_maze_reward_function_name(env_name):
    """
    Class name in reward_functions/*_gt_rew_fns.py for prompts.

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    assert env_name in _REWARD_CLASS, "invalid point maze env_name: %r" % (env_name,)
    return _REWARD_CLASS[env_name]


def point_maze_feedback_obs_class_name(env_name):
    """
    Observation class name for Feedback prompts (import in starter code).

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    assert env_name in _OBS_CLASS, "invalid point maze env_name: %r" % (env_name,)
    return _OBS_CLASS[env_name]


def point_maze_reward_fn_basename(env_name):
    """
    Basename of the ground-truth reward module under direct_policy_learning/reward_functions/.

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    assert env_name in _REWARD_FN_BASENAME, "invalid point maze env_name: %r" % (env_name,)
    return _REWARD_FN_BASENAME[env_name]


def point_maze_max_episode_steps(env_name):
    """
    Default max episode steps for the registered Gym id (used for rollout horizon).

    Assumptions:
    - is_point_maze_env_name(env_name) is True.
    """
    if env_name in (
        "point_maze_large_v3",
        "point_maze_large_diverse_g_v3",
    ):
        return POINT_MAZE_LARGE_V3_MAX_EPISODE_STEPS
    assert False, "invalid point maze env_name: %r" % (env_name,)
