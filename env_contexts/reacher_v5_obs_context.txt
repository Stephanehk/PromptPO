"""
Structured observation for Gymnasium MuJoCo Reacher-v5.

Field semantics follow Gymnasium Reacher-v5 and reacher.xml:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/reacher_v5.py

MuJoCo joint/geom conventions:
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

The environment builds a 10-dimensional observation from cos/sin of arm angles, target
position, angular velocities, and fingertip−target position (see _get_obs). Reward uses
the Euclidean distance after the physics step.

Assumptions:
- env is Reacher-v5 unwrapped with bodies named "fingertip" and "target" in the model.

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MujocoReacherObservation:
    """
    Reacher-v5 observation bundle for policies and reward code.

    Action space (default): Box(-1, 1, (2,), float32) — torques at joint0 and joint1
    (see env_contexts/reacher_v5_policy_context.txt).

    ---------------------------------------------------------------------------
    mjData.qpos (length 4 for default reacher.xml), joint order in model:
      [0] joint0 — first arm hinge angle (rad)
      [1] joint1 — second arm hinge angle (rad)
      [2] target_x — target position x (slide) (m)
      [3] target_y — target position y (slide) (m)

    mjData.qvel (length 4): first two are angular velocities of joint0, joint1; last two are
    target slide velocities (typically zero for the target).

    ---------------------------------------------------------------------------
    obs_vector — length 10. Gymnasium _get_obs order (see Reacher-v5 table):

      [0] cos(joint0)
      [1] cos(joint1)
      [2] sin(joint0)
      [3] sin(joint1)
      [4] target x — qpos[2] (m)
      [5] target y — qpos[3] (m)
      [6] angular velocity joint0 (rad/s)
      [7] angular velocity joint1 (rad/s)
      [8] (fingertip − target) x in world frame (m) — first 2 of 3D COM difference
      [9] (fingertip − target) y (m)

    The z-component of fingertip−target is always 0 in this 2D task and is omitted from obs.

    fingertip_minus_target — 3-vector copy of get_body_com("fingertip") − get_body_com("target")
    after the step; reward uses np.linalg.norm(vec).

    fingertip_com / target_com — body COM positions for debugging or auxiliary features.
    """

    def __init__(self):
        self.obs_vector = None
        self.dt = None
        self.qpos = None
        self.qvel = None
        self.fingertip_minus_target = None
        self.fingertip_com = None
        self.target_com = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        vec = u.get_body_com("fingertip") - u.get_body_com("target")
        self.fingertip_minus_target = np.asarray(vec, dtype=np.float64).copy()
        self.fingertip_com = np.asarray(u.get_body_com("fingertip"), dtype=np.float64).copy()
        self.target_com = np.asarray(u.get_body_com("target"), dtype=np.float64).copy()
