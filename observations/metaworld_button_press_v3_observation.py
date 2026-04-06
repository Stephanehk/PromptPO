"""
Observation bundle for Meta-World MT1 task button-press-v3 (SawyerButtonPressEnvV3).

See metaworld/envs/sawyer_button_press_v3.py and metaworld_mt1_observation_base.py.
"""

from direct_policy_learning.observations.metaworld_mt1_observation_base import (
    MetaWorldMT1ObservationBase,
)


class MetaWorldButtonPressV3Observation(MetaWorldMT1ObservationBase):
    """
    button-press-v3: press the wall-mounted button with the gripper.

    Action space: Box(-1, 1, (4,), float32) — xyz hand displacement + gripper control
    (see env_contexts/button_press_v3_policy_context.txt).
    """
