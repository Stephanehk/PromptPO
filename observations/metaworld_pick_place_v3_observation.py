"""
Observation bundle for Meta-World MT1 task pick-place-v3 (SawyerPickPlaceEnvV3).

See metaworld/envs/sawyer_pick_place_v3.py and metaworld_mt1_observation_base.py.
"""

from direct_policy_learning.observations.metaworld_mt1_observation_base import (
    MetaWorldMT1ObservationBase,
)


class MetaWorldPickPlaceV3Observation(MetaWorldMT1ObservationBase):
    """
    pick-place-v3: pick up the puck and move it to a goal location.

    Action space: Box(-1, 1, (4,), float32) — xyz hand displacement + gripper control
    (see env_contexts/pick_place_v3_policy_context.txt).
    """
