"""
Structured observation for Gymnasium MuJoCo Hopper-v5.

Field semantics follow Gymnasium Hopper-v5 and the underlying hopper.xml model:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v5.py

MuJoCo joint/DOF conventions (mjData.qpos, mjData.qvel) are documented in:
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

Default Hopper-v5 uses exclude_current_positions_from_observation=True: _get_obs drops qpos[0]
(rootx slide) and concatenates the remaining qpos with qvel (velocities clipped to [-10, 10]
in the observation only).

Assumptions:
- env is Gymnasium Hopper-v5 unwrapped (TimeLimit peeled) with .data.qpos, .data.qvel, .dt.
- obs_np equals env.unwrapped._get_obs() for this timestep.

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class MujocoHopperObservation:
    """
    Hopper-v5 observation bundle for policies and reward code.

    Action space (default): Box(-1, 1, (3,), float32) — torques at thigh, leg, foot hinges
    (see Gymnasium action table and env_contexts/hopper_v5_policy_context.txt).

    ---------------------------------------------------------------------------
    Full mjData.qpos (length model.nq, typically 6 for hopper.xml):
      [0] rootx — horizontal position of torso (slide along x) (m). Excluded from obs
          when exclude_current_positions_from_observation=True; used for x-velocity reward.
      [1] rootz — vertical (height) of torso (m). Maps to obs_vector[0] when rootx excluded.
      [2] rooty — pitch-like hinge angle of torso (rad). Maps to obs_vector[1].
      [3] thigh_joint — angle between torso and thigh (rad). Maps to obs_vector[2].
      [4] leg_joint — angle between thigh and leg (rad). Maps to obs_vector[3].
      [5] foot_joint — angle between leg and foot (rad). Maps to obs_vector[4].

    Full mjData.qvel (length model.nv, typically 6): time-derivatives of qpos in generalized
    coordinates (rootx, rootz slides; rooty, thigh, leg, foot hinges). The observation vector
    uses np.clip(qvel, -10, 10) for these six values (see _get_obs); see obs_velocity_block.

    ---------------------------------------------------------------------------
    obs_vector — length 11 (default). Order: concat(position_without_rootx, clipped_velocity).

    Position block obs_vector[0:5] — qpos with qpos[0] (rootx) removed:
      [0] z (height) of torso — same as qpos[1] / rootz (m)
      [1] angle of torso — rooty hinge (rad)
      [2] thigh_joint angle (rad)
      [3] leg_joint angle (rad)
      [4] foot_joint angle (rad)

    Velocity block obs_vector[5:11] — clip(qvel, -10, 10), same joint order as qvel:
      [5] x-velocity of rootx slide (m/s)
      [6] z-velocity of rootz slide (m/s)
      [7] angular velocity of rooty (rad/s)
      [8] angular velocity of thigh_joint (rad/s)
      [9] angular velocity of leg_joint (rad/s)
      [10] angular velocity of foot_joint (rad/s)

    For unclipped speeds and accelerations, use self.qvel (full mjData.qvel).

    ---------------------------------------------------------------------------
    qpos / qvel — full mjData copies (same as env.unwrapped.data.qpos / .qvel).
    root_x_position — qpos[0]; excluded from default observation but needed for forward motion.
    observation_structure — copy of env.unwrapped.observation_structure when present (skipped_qpos,
    qpos, qvel counts for tooling).
    """

    def __init__(self):
        self.obs_vector = None
        self.qpos = None
        self.qvel = None
        self.dt = None
        self.root_x_position = None
        self.observation_structure = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.root_x_position = float(self.qpos[0])
        struct = getattr(u, "observation_structure", None)
        self.observation_structure = dict(struct) if struct is not None else None

    def state_vector(self):
        """
        Concatenate qpos and qvel in MuJoCo order (matches MujocoEnv.state_vector()).

        Used for healthy-range checks consistent with Hopper-v5 (see Gymnasium is_healthy).
        """
        assert self.qpos is not None and self.qvel is not None, "fill_from_env must run first."
        return np.concatenate([self.qpos, self.qvel])
