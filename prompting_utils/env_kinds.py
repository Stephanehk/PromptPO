"""
Map ``env_name`` to class names, starter imports, and rollout horizons.

Used by prompt templates and by code that loads generated policy / Feedback classes.
"""

from direct_policy_learning.env_utils.meta_world_env_names import (
    is_meta_world_env_name,
    meta_world_feedback_obs_class_name,
    meta_world_policy_class_base,
    meta_world_reward_function_name,
)
from direct_policy_learning.env_utils.mujoco_env_names import (
    is_mujoco_env_name,
    mujoco_feedback_obs_class_name,
    mujoco_policy_class_base,
    mujoco_reward_function_name,
)
from direct_policy_learning.env_utils.point_maze_env_names import (
    is_point_maze_env_name,
    point_maze_feedback_obs_class_name,
    point_maze_max_episode_steps,
    point_maze_policy_class_base,
    point_maze_reward_function_name,
)
from direct_policy_learning.env_utils.noise_world_env_names import (
    is_noise_world_env_name,
    noise_world_feedback_obs_class_name,
    noise_world_max_episode_steps,
    noise_world_policy_class_base,
    noise_world_reward_function_name,
)


def policy_class_base_for_env(env_name):
    if is_mujoco_env_name(env_name):
        return mujoco_policy_class_base(env_name)
    if is_meta_world_env_name(env_name):
        return meta_world_policy_class_base(env_name)
    if is_point_maze_env_name(env_name):
        return point_maze_policy_class_base(env_name)
    if is_noise_world_env_name(env_name):
        return noise_world_policy_class_base(env_name)
    return env_name.title()


def reward_function_title_for_prompt(env_name):
    if is_mujoco_env_name(env_name):
        return mujoco_reward_function_name(env_name)
    if is_meta_world_env_name(env_name):
        return meta_world_reward_function_name(env_name)
    if is_point_maze_env_name(env_name):
        return point_maze_reward_function_name(env_name)
    if is_noise_world_env_name(env_name):
        return noise_world_reward_function_name(env_name)
    return "%sRewardFunction" % env_name.title()


def starter_code_for_env(env_name):
    if is_mujoco_env_name(env_name):
        obs_cls = mujoco_feedback_obs_class_name(env_name)
        return """from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import %s
""" % (
            obs_cls,
        )
    if is_meta_world_env_name(env_name):
        obs_cls = meta_world_feedback_obs_class_name(env_name)
        return """from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import %s
""" % (
            obs_cls,
        )
    if is_point_maze_env_name(env_name):
        obs_cls = point_maze_feedback_obs_class_name(env_name)
        return """from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import %s
""" % (
            obs_cls,
        )
    if is_noise_world_env_name(env_name):
        obs_cls = noise_world_feedback_obs_class_name(env_name)
        return """from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import %s
""" % (
            obs_cls,
        )
    starter_code_by_env = {
        "traffic": """from abc import ABCMeta
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation
""",
        "pandemic": """from abc import ABCMeta
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import PandemicObservation
""",
        "glucose": """from abc import ABCMeta
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation
""",
    }
    assert env_name in starter_code_by_env, "Missing starter code for env_name."
    return starter_code_by_env[env_name]


def get_horizon_for_env(env_name):
    if env_name == "reacher_v5":
        return 50
    if is_mujoco_env_name(env_name):
        return 1000
    if is_meta_world_env_name(env_name):
        return 500
    if is_point_maze_env_name(env_name):
        return point_maze_max_episode_steps(env_name)
    if is_noise_world_env_name(env_name):
        return noise_world_max_episode_steps(env_name)
    horizons = {
        "traffic": 500,
        "pandemic": 192,
        "glucose": 20 * 12 * 24,
    }
    assert env_name in horizons, (
        "env_name must be one of: traffic, pandemic, glucose, mujoco *_v5, "
        "Meta-World MT1, a supported point_maze_*_v3 id, or noise_world_board_*"
    )
    return horizons[env_name]
