"""
Meta-World MT1 task identifiers for direct_policy_learning.

Internal env_name uses underscores (e.g. button_press_v3); Gymnasium MT1 uses hyphenated
task names (e.g. button-press-v3). Environments are constructed with:

  import metaworld  # registers Meta-World/* gym ids
  gym.make("Meta-World/MT1", env_name=<task>, ...)

Assumptions:
- The `metaworld` package is installed and imported before `gym.make` so registrations run.
"""

META_WORLD_ENV_NAMES = (
    "button_press_v3",
    "pick_place_v3",
    "door_open_v3",
    "drawer_open_v3",
)

_META_WORLD_MT1_TASK = {
    "button_press_v3": "button-press-v3",
    "pick_place_v3": "pick-place-v3",
    "door_open_v3": "door-open-v3",
    "drawer_open_v3": "drawer-open-v3",
}

_META_WORLD_POLICY_CLASS_BASE = {
    "button_press_v3": "ButtonPressV3",
    "pick_place_v3": "PickPlaceV3",
    "door_open_v3": "DoorOpenV3",
    "drawer_open_v3": "DrawerOpenV3",
}

_META_WORLD_REWARD_CLASS = {
    "button_press_v3": "ButtonPressV3RewardFunction",
    "pick_place_v3": "PickPlaceV3RewardFunction",
    "door_open_v3": "DoorOpenV3RewardFunction",
    "drawer_open_v3": "DrawerOpenV3RewardFunction",
}

_META_WORLD_OBS_CLASS = {
    "button_press_v3": "MetaWorldButtonPressV3Observation",
    "pick_place_v3": "MetaWorldPickPlaceV3Observation",
    "door_open_v3": "MetaWorldDoorOpenV3Observation",
    "drawer_open_v3": "MetaWorldDrawerOpenV3Observation",
}

_META_WORLD_REWARD_FN_BASENAME = {
    "button_press_v3": "button_press_v3_gt_rew_fns.py",
    "pick_place_v3": "pick_place_v3_gt_rew_fns.py",
    "door_open_v3": "door_open_v3_gt_rew_fns.py",
    "drawer_open_v3": "drawer_open_v3_gt_rew_fns.py",
}

META_WORLD_GYM_MT1_ID = "Meta-World/MT1"


def is_meta_world_env_name(env_name):
    """
    Return True if env_name is a supported Meta-World MT1 env id.

    Assumptions:
    - env_name is a string or None.
    """
    if env_name is None or not isinstance(env_name, str):
        return False
    return env_name in META_WORLD_ENV_NAMES


def meta_world_mt1_task_name(env_name):
    """
    Map env_name to the MT1 `env_name` string passed to gym.make("Meta-World/MT1", ...).

    Assumptions:
    - is_meta_world_env_name(env_name) is True.
    """
    assert env_name in _META_WORLD_MT1_TASK, "invalid meta_world env_name: %r" % (env_name,)
    return _META_WORLD_MT1_TASK[env_name]


def meta_world_policy_class_base(env_name):
    """
    Base name for generated policy classes (e.g. ButtonPressV3 -> ButtonPressV3Policy).

    Assumptions:
    - is_meta_world_env_name(env_name) is True.
    """
    assert env_name in _META_WORLD_POLICY_CLASS_BASE, "invalid meta_world env_name: %r" % (
        env_name,
    )
    return _META_WORLD_POLICY_CLASS_BASE[env_name]


def meta_world_reward_function_name(env_name):
    """
    Class name in reward_functions/*_gt_rew_fns.py for prompts.

    Assumptions:
    - is_meta_world_env_name(env_name) is True.
    """
    assert env_name in _META_WORLD_REWARD_CLASS, "invalid meta_world env_name: %r" % (
        env_name,
    )
    return _META_WORLD_REWARD_CLASS[env_name]


def meta_world_feedback_obs_class_name(env_name):
    """
    Observation class name for Feedback prompts (import in starter code).

    Assumptions:
    - is_meta_world_env_name(env_name) is True.
    """
    assert env_name in _META_WORLD_OBS_CLASS, "invalid meta_world env_name: %r" % (env_name,)
    return _META_WORLD_OBS_CLASS[env_name]


def meta_world_reward_fn_basename(env_name):
    """
    Basename of the ground-truth reward module under direct_policy_learning/reward_functions/.

    Assumptions:
    - is_meta_world_env_name(env_name) is True.
    """
    assert env_name in _META_WORLD_REWARD_FN_BASENAME, "invalid meta_world env_name: %r" % (
        env_name,
    )
    return _META_WORLD_REWARD_FN_BASENAME[env_name]
