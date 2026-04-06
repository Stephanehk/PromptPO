"""
Parse MuJoCo env identifiers for Gymnasium v5 tasks used in direct_policy_learning.

env_name uses the pattern <task>_v5; the Gymnasium id is <Task>-v5.
"""

MUJOCO_ENV_NAMES = (
    "ant_v5",
    "halfcheetah_v5",
    "hopper_v5",
    "humanoid_v5",
    "swimmer_v5",
    "reacher_v5",
    "inverted_double_pendulum_v5",
)

_MUJOCO_GYM_ID = {
    "ant_v5": "Ant-v5",
    "halfcheetah_v5": "HalfCheetah-v5",
    "hopper_v5": "Hopper-v5",
    "humanoid_v5": "Humanoid-v5",
    "swimmer_v5": "Swimmer-v5",
    "reacher_v5": "Reacher-v5",
    "inverted_double_pendulum_v5": "InvertedDoublePendulum-v5",
}

_MUJOCO_POLICY_CLASS_BASE = {
    "ant_v5": "AntV5",
    "halfcheetah_v5": "HalfcheetahV5",
    "hopper_v5": "HopperV5",
    "humanoid_v5": "HumanoidV5",
    "swimmer_v5": "SwimmerV5",
    "reacher_v5": "ReacherV5",
    "inverted_double_pendulum_v5": "InvertedDoublePendulumV5",
}

_MUJOCO_REWARD_CLASS = {
    "ant_v5": "AntRewardFunction",
    "halfcheetah_v5": "HalfcheetahRewardFunction",
    "hopper_v5": "HopperRewardFunction",
    "humanoid_v5": "HumanoidRewardFunction",
    "swimmer_v5": "SwimmerRewardFunction",
    "reacher_v5": "ReacherRewardFunction",
    "inverted_double_pendulum_v5": "InvertedDoublePendulumRewardFunction",
}

_MUJOCO_OBS_CLASS = {
    "ant_v5": "MujocoAntObservation",
    "halfcheetah_v5": "MujocoHalfCheetahObservation",
    "hopper_v5": "MujocoHopperObservation",
    "humanoid_v5": "MujocoHumanoidObservation",
    "swimmer_v5": "MujocoSwimmerObservation",
    "reacher_v5": "MujocoReacherObservation",
    "inverted_double_pendulum_v5": "MujocoInvertedDoublePendulumObservation",
}

_MUJOCO_REWARD_FN_BASENAME = {
    "ant_v5": "ant_gt_rew_fns.py",
    "halfcheetah_v5": "halfcheetah_gt_rew_fns.py",
    "hopper_v5": "hopper_gt_rew_fns.py",
    "humanoid_v5": "humanoid_gt_rew_fns.py",
    "swimmer_v5": "swimmer_gt_rew_fns.py",
    "reacher_v5": "reacher_gt_rew_fns.py",
    "inverted_double_pendulum_v5": "inverted_double_pendulum_gt_rew_fns.py",
}

# Only these Gymnasium MuJoCo v5 envs accept `terminate_when_unhealthy` in `gym.make` (their
# Env __init__ defines the kwarg). HalfCheetah, Swimmer, Reacher, etc. raise TypeError if it
# is passed. Same class of envs as Ant / Hopper / Humanoid / Walker2d in Gymnasium.
# When walker_v5 is added to this repo, append "walker_v5" here.
MUJOCO_ENV_NAMES_WITH_TERMINATE_WHEN_UNHEALTHY = (
    "ant_v5",
    "hopper_v5",
    "humanoid_v5",
)


def mujoco_supports_terminate_when_unhealthy(env_name):
    """
    Return True iff `gym.make(..., terminate_when_unhealthy=bool)` is valid for this env_name.

    Assumptions:
    - env_name is a supported mujoco id or will be checked by caller.
    """
    return env_name in MUJOCO_ENV_NAMES_WITH_TERMINATE_WHEN_UNHEALTHY


def is_mujoco_env_name(env_name):
    """
    Return True if env_name is a supported MuJoCo direct_policy_learning id.

    Assumptions:
    - env_name is a string.
    """
    if env_name is None or not isinstance(env_name, str):
        return False
    return env_name in MUJOCO_ENV_NAMES


def mujoco_gym_make_id(env_name):
    """
    Map env_name to Gymnasium registration id (e.g. ant_v5 -> Ant-v5).

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    assert env_name in _MUJOCO_GYM_ID, "invalid mujoco env_name: %r" % (env_name,)
    return _MUJOCO_GYM_ID[env_name]


def mujoco_variant(env_name):
    """
    Legacy coarse family for code paths that only distinguished ant vs halfcheetah.

    New envs return their short name: hopper, humanoid, swimmer, reacher.

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    if env_name == "ant_v5":
        return "ant"
    if env_name == "halfcheetah_v5":
        return "halfcheetah"
    if env_name == "hopper_v5":
        return "hopper"
    if env_name == "humanoid_v5":
        return "humanoid"
    if env_name == "swimmer_v5":
        return "swimmer"
    if env_name == "reacher_v5":
        return "reacher"
    if env_name == "inverted_double_pendulum_v5":
        return "inverted_double_pendulum"
    assert False, "invalid mujoco env_name: %r" % (env_name,)


def mujoco_policy_class_base(env_name):
    """
    Base name for generated policy classes (e.g. HopperV5 -> HopperV5Policy).

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    assert env_name in _MUJOCO_POLICY_CLASS_BASE, "invalid mujoco env_name: %r" % (env_name,)
    return _MUJOCO_POLICY_CLASS_BASE[env_name]


def mujoco_reward_function_name(env_name):
    """
    Class name in reward_functions/*_gt_rew_fns.py for prompts.

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    assert env_name in _MUJOCO_REWARD_CLASS, "invalid mujoco env_name: %r" % (env_name,)
    return _MUJOCO_REWARD_CLASS[env_name]


def mujoco_feedback_obs_class_name(env_name):
    """
    Observation class name for Feedback prompts (import in starter code).

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    assert env_name in _MUJOCO_OBS_CLASS, "invalid mujoco env_name: %r" % (env_name,)
    return _MUJOCO_OBS_CLASS[env_name]


def mujoco_reward_fn_basename(env_name):
    """
    Basename of the ground-truth reward module under direct_policy_learning/reward_functions/.

    Assumptions:
    - is_mujoco_env_name(env_name) is True.
    """
    assert env_name in _MUJOCO_REWARD_FN_BASENAME, "invalid mujoco env_name: %r" % (env_name,)
    return _MUJOCO_REWARD_FN_BASENAME[env_name]
