"""
Structured observation for Gymnasium MuJoCo Ant-v5.

Field semantics follow the Gymnasium Ant documentation and source:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v5.py

Default Ant-v5 uses exclude_current_positions_from_observation=True and
include_cfrc_ext_in_observation=True, so the policy sees a 105-dimensional vector:
qpos without torso x,y (13) + qvel (14) + flattened clipped-contact-related features (78).

Assumptions:
- env is a Gymnasium Ant-v5 (unwrapped from TimeLimit etc.) with .data.qpos, .data.qvel,
  .data.cfrc_ext, .dt, and get_body_com("torso").
- obs_np equals env.unwrapped observation for this timestep (same as _get_obs() output).

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MujocoAntObservation:
    """
    Ant-v5 observation bundle for policies and reward code.

    Action space (default): Box(-1, 1, (8,), float32) — torques at eight leg hinges
    (see Gymnasium Ant-v5 action table: hip_4/angle_4, hip_1/angle_1, ...).

    ---------------------------------------------------------------------------
    obs_vector — length 105 (default kwargs). Layout matches env._get_obs() order:
      concat(position, velocity, contact_force_flat).

    Position block obs_vector[0:13] — qpos with first two coordinates (torso x, y) removed:
      [0]  z-coordinate of the torso (centre) (m), root free joint
      [1]  w-orientation of the torso (quaternion), root free joint
      [2]  x-orientation of the torso (quaternion)
      [3]  y-orientation of the torso (quaternion)
      [4]  z-orientation of the torso (quaternion)
      [5]  angle between torso and first link on front left — hip_1 (rad)
      [6]  angle between the two links on the front left — ankle_1 (rad)
      [7]  angle between torso and first link on front right — hip_2 (rad)
      [8]  angle between the two links on the front right — ankle_2 (rad)
      [9]  angle between torso and first link on back left — hip_3 (rad)
      [10] angle between the two links on the back left — ankle_3 (rad)
      [11] angle between torso and first link on back right — hip_4 (rad)
      [12] angle between the two links on the back right — ankle_4 (rad)
    (Excluded from obs but still in full qpos: torso x, y — indices 0,1 of qpos.)

    Velocity block obs_vector[13:27] — full qvel (14 elements):
      [13] x linear velocity of the torso (m/s)
      [14] y linear velocity of the torso (m/s)
      [15] z linear velocity of the torso (m/s)
      [16] x angular velocity of the torso (rad/s)
      [17] y angular velocity of the torso (rad/s)
      [18] z angular velocity of the torso (rad/s)
      [19]–[26] angular velocities at the eight leg hinges (same leg order as position joints above)

    Contact block obs_vector[27:105] — contact_forces[1:].flatten() with world body excluded:
      13 bodies × 6 = 78 numbers. Per body k in 0..12, base = 27 + 6*k gives:
        [base+0..base+2] external force (fx, fy, fz) on that body’s COM (in world frame)
        [base+3..base+5] external torque (tx, ty, tz) on that body’s COM
      Body order (v5 ids 1..13 in env): torso, front_left_leg, aux_1, ankle_1, front_right_leg,
      aux_2, ankle_2, back_leg, aux_3, ankle_3, right_back_leg, aux_4, ankle_4
      (see Gymnasium Ant-v5 “body part” table; worldbody index 0 is omitted here).

    ---------------------------------------------------------------------------
    qpos — full generalized positions, length model.nq (15 for default ant.xml):
      [0] torso x, [1] torso y, [2] torso z, [3:7] torso quaternion (w,x,y,z), [7:15] eight hinge angles.

    qvel — full generalized velocities, length model.nv (14): first 6 are torso translational/
      angular velocity in generalized coordinates, then eight hinge velocities.

    cfrc_ext — shape (nbody, 6), raw env.data.cfrc_ext (used to compute contact cost like
      Ant-v5: clip then sum of squares). The contact segment of obs_vector uses the same
      clipped forces as env.contact_forces[1:].flatten(), not the raw tensor.

    main_body_xy — (2,) torso center-of-mass (x, y) in world frame from get_body_com("torso")[:2];
      forward reward uses x-displacement of this point per Ant-v5 step().
    """

    def __init__(self):
        self.obs_vector = None
        self.qpos = None
        self.qvel = None
        self.cfrc_ext = None
        self.dt = None
        self.main_body_xy = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        self.cfrc_ext = np.asarray(u.data.cfrc_ext, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.main_body_xy = np.asarray(u.get_body_com("torso")[:2], dtype=np.float64).copy()

    def state_vector(self):
        """
        Concatenate qpos and qvel in MuJoCo order (matches MujocoEnv.state_vector() for this model).

        Used for healthy checks consistent with Ant-v5 (z at index 2 of qpos).
        """
        assert self.qpos is not None and self.qvel is not None, "fill_from_env must run first."
        return np.concatenate([self.qpos, self.qvel])
