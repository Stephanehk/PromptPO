"""
Structured observation for Gymnasium MuJoCo HalfCheetah-v5.

Field semantics follow the Gymnasium HalfCheetah documentation and source:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/half_cheetah_v5.py

Default uses exclude_current_positions_from_observation=True, so the policy sees 17 numbers:
qpos without root x (8) + qvel (9).

Assumptions:
- env is HalfCheetah-v5 with .data.qpos, .data.qvel, .dt.
- obs_np matches env.unwrapped _get_obs() for this timestep.

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MujocoHalfCheetahObservation:
    """
    HalfCheetah-v5 observation bundle for policies and reward code.

    Action space (default): Box(-1, 1, (6,), float32) — torques on bthigh, bshin, bfoot,
    fthigh, fshin, ffoot (see Gymnasium action table).

    ---------------------------------------------------------------------------
    obs_vector — length 17 (default). Order: concat(position, velocity) after dropping root x.

    Position block obs_vector[0:8] — qpos with first coordinate (rootx / tip x) removed:
      [0] z-coordinate of the front tip (rootz slide) (m)
      [1] angle of the front tip (rooty hinge) (rad)
      [2] angle of the back thigh — bthigh (rad)
      [3] angle of the back shin — bshin (rad)
      [4] angle of the back foot — bfoot (rad)
      [5] angle of the front thigh — fthigh (rad)
      [6] angle of the front shin — fshin (rad)
      [7] angle of the front foot — ffoot (rad)
    (Excluded from obs: x-coordinate of the front tip — qpos[0], used for forward reward.)

    Velocity block obs_vector[8:17] — full qvel (9 elements):
      [8]  velocity of the x-coordinate of the front tip / rootx slide (m/s)
      [9]  velocity of the z-coordinate of the front tip (m/s)
      [10] angular velocity of the front tip — rooty (rad/s)
      [11] angular velocity of the back thigh — bthigh (rad/s)
      [12] angular velocity of the back shin — bshin (rad/s)
      [13] angular velocity of the back foot — bfoot (rad/s)
      [14] angular velocity of the front thigh — fthigh (rad/s)
      [15] angular velocity of the front shin — fshin (rad/s)
      [16] angular velocity of the front foot — ffoot (rad/s)

    ---------------------------------------------------------------------------
    qpos — full generalized positions (9 for default half_cheetah.xml): [0] root x, then
      z, angles for rooty and six leg joints as in MuJoCo model order.

    qvel — full generalized velocities (9 elements), matching qvel layout above.

    main_body_xy — [qpos[0], 0.0]; qpos[0] is root x for forward displacement / x_velocity
      in the true HalfCheetah-v5 reward.
    """

    def __init__(self):
        self.obs_vector = None
        self.qpos = None
        self.qvel = None
        self.dt = None
        self.main_body_xy = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.main_body_xy = np.array(
            [float(u.data.qpos[0]), 0.0],
            dtype=np.float64,
        )
