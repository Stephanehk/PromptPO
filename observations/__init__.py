"""
Observation wrapper types for direct_policy_learning (MuJoCo and others).
"""

from direct_policy_learning.observations.mujoco_ant_observation import MujocoAntObservation
from direct_policy_learning.observations.mujoco_halfcheetah_observation import (
    MujocoHalfCheetahObservation,
)
from direct_policy_learning.observations.mujoco_hopper_observation import MujocoHopperObservation
from direct_policy_learning.observations.mujoco_humanoid_observation import (
    MujocoHumanoidObservation,
)
from direct_policy_learning.observations.mujoco_inverted_double_pendulum_observation import (
    MujocoInvertedDoublePendulumObservation,
)
from direct_policy_learning.observations.mujoco_reacher_observation import MujocoReacherObservation
from direct_policy_learning.observations.mujoco_swimmer_observation import MujocoSwimmerObservation
from direct_policy_learning.observations.metaworld_button_press_v3_observation import (
    MetaWorldButtonPressV3Observation,
)
from direct_policy_learning.observations.metaworld_door_open_v3_observation import (
    MetaWorldDoorOpenV3Observation,
)
from direct_policy_learning.observations.metaworld_drawer_open_v3_observation import (
    MetaWorldDrawerOpenV3Observation,
)
from direct_policy_learning.observations.metaworld_pick_place_v3_observation import (
    MetaWorldPickPlaceV3Observation,
)
from direct_policy_learning.observations.point_maze_large_v3_observation import (
    PointMazeLargeV3Observation,
)
from direct_policy_learning.observations.point_maze_large_diverse_g_v3_observation import (
    PointMazeLargeDiverseGV3Observation,
)
from direct_policy_learning.observations.noise_world_observation import NoiseWorldObservation

__all__ = [
    "MujocoAntObservation",
    "MujocoHalfCheetahObservation",
    "MujocoHopperObservation",
    "MujocoHumanoidObservation",
    "MujocoInvertedDoublePendulumObservation",
    "MujocoReacherObservation",
    "MujocoSwimmerObservation",
    "MetaWorldButtonPressV3Observation",
    "MetaWorldDoorOpenV3Observation",
    "MetaWorldDrawerOpenV3Observation",
    "MetaWorldPickPlaceV3Observation",
    "PointMazeLargeV3Observation",
    "PointMazeLargeDiverseGV3Observation",
    "NoiseWorldObservation",
]
