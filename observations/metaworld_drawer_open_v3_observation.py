"""
Observation bundle for Meta-World MT1 task drawer-open-v3 (SawyerDrawerEnvV3).

See metaworld/envs/sawyer_drawer_v3.py and metaworld_mt1_observation_base.py.
"""

from direct_policy_learning.observations.metaworld_mt1_observation_base import (
    MetaWorldMT1ObservationBase,
)


class MetaWorldDrawerOpenV3Observation(MetaWorldMT1ObservationBase):
    """
    drawer-open-v3: grasp the handle and pull the drawer open to the target extension.

    Action space: Box(-1, 1, (4,), float32) — xyz hand displacement + gripper control
    (see env_contexts/drawer_open_v3_policy_context.txt).
    """
