"""
Structured observation for Gymnasium MuJoCo Swimmer-v5.

Field semantics follow Gymnasium Swimmer-v5 and swimmer.xml:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/swimmer_v5.py

MuJoCo generalized coordinates (mjData.qpos / qvel):
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

Default Swimmer-v5 uses exclude_current_positions_from_observation=True: _get_obs removes the
first two qpos components (planar x,y of the front tip) so the policy does not see absolute
position; forward reward still uses displacement of qpos[0] (tip x) from full mjData.

Assumptions:
- env is Swimmer-v5 unwrapped with .data.qpos, .data.qvel, .dt.
- obs_np equals env.unwrapped._get_obs().

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MujocoSwimmerObservation:
    """
    Swimmer-v5 observation bundle for policies and reward code.

    Action space (default): Box(-1, 1, (2,), float32) — torques at motor1_rot, motor2_rot
    (see Gymnasium action table and env_contexts/swimmer_v5_policy_context.txt).

    ---------------------------------------------------------------------------
    Full mjData.qpos (length model.nq, typically 5 for default swimmer.xml):
      [0] slider1 — x position of the swimming tip in the plane (m)
      [1] slider2 — y position of the tip (m)
      [2] free_body_rot — heading angle of the front segment (rad)
      [3] motor1_rot — angle at first internal hinge (rad)
      [4] motor2_rot — angle at second internal hinge (rad)

    Full mjData.qvel (length model.nv, typically 5): derivatives of the above (tip x/y speeds,
    then angular velocities).

    ---------------------------------------------------------------------------
    obs_vector — length 8 (default). Order: concat(qpos[2:], qvel) — i.e. drop tip x,y.

    Position block obs_vector[0:3] — corresponds to qpos[2:5]:
      [0] free_body_rot — heading angle of front link (rad)
      [1] motor1_rot — first rotor angle between links (rad)
      [2] motor2_rot — second rotor angle between links (rad)

    Velocity block obs_vector[3:8] — full qvel (5 elements):
      [3] x-velocity of tip (slider1) (m/s)
      [4] y-velocity of tip (slider2) (m/s)
      [5] angular velocity of free_body_rot (rad/s)
      [6] angular velocity of motor1_rot (rad/s)
      [7] angular velocity of motor2_rot (rad/s)

    Excluded from obs but present in self.qpos: tip x = qpos[0], tip y = qpos[1] (used for
    forward x-velocity in the environment reward).

    observation_structure — copy of env.unwrapped.observation_structure when present.
    """

    def __init__(self):
        self.obs_vector = None
        self.qpos = None
        self.qvel = None
        self.dt = None
        self.tip_xy = None
        self.observation_structure = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.tip_xy = np.asarray(self.qpos[0:2], dtype=np.float64).copy()
        struct = getattr(u, "observation_structure", None)
        self.observation_structure = dict(struct) if struct is not None else None
