"""
Observation bundle for Meta-World MT1 task door-open-v3 (SawyerDoorEnvV3).

See metaworld/envs/sawyer_door_v3.py and metaworld_mt1_observation_base.py.
"""

from direct_policy_learning.observations.metaworld_mt1_observation_base import (
    MetaWorldMT1ObservationBase,
)


class MetaWorldDoorOpenV3Observation(MetaWorldMT1ObservationBase):
    """
    door-open-v3: grasp the handle and pull the door open to a target angle.

    Action space: Box(-1, 1, (4,), float32) — xyz hand displacement + gripper control
    (see env_contexts/door_open_v3_policy_context.txt).
    """
