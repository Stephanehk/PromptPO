"""
Structured observation for Gymnasium MuJoCo Humanoid-v5.

Field semantics follow Gymnasium Humanoid-v5 and humanoid.xml:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py

MuJoCo mjData fields (qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext, ctrl):
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

Default Humanoid-v5 concatenates (after optional exclusions) truncated qpos, qvel, flattened
cinert[1:], cvel[1:], qfrc_actuator[6:], cfrc_ext[1:] — see _get_obs in humanoid_v5.py.
World body (index 0) and free-joint root rows are omitted from cinert/cvel/cfrc_ext as in Gymnasium.

Assumptions:
- env is Humanoid-v5 unwrapped with standard default include_* flags.
- obs_np equals env.unwrapped._get_obs().

Lives under direct_policy_learning/observations/.
"""

import numpy as np


def _mass_center_xy(model, data):
    """
    Global center of mass in x,y (matches Gymnasium humanoid_v5.mass_center).

    Uses body masses and positions (mjData.xipos) — see MuJoCo body kinematics docs.
    """
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()


class MujocoHumanoidObservation:
    """
    Humanoid-v5 observation bundle for policies and reward code.

    Action space (default): Box(-0.4, 0.4, (17,), float32) — torques at abdomen, hips,
    knees, shoulders, elbows (see Gymnasium action table and env_contexts/humanoid_v5_policy_context.txt).

    ---------------------------------------------------------------------------
    obs_vector — length depends on model and flags (default ~348). Concatenation order in
    Gymnasium _get_obs (same as this.obs_vector):

      1) qpos with first two coordinates (torso x, y of free joint) removed when
         exclude_current_positions_from_observation=True — length (nq - 2).
      2) Full qvel (length nv).
      3) cinert[1:].flatten() — COM-based inertia diagnostics per body (10 floats per body
         after skipping world), see MuJoCo mjData.cinert.
      4) cvel[1:].flatten() — COM-based velocities per body (6 floats per body), mjData.cvel.
      5) qfrc_actuator[6:].flatten() — actuator forces on internal DOFs (skips first 6 tied to
         free joint), mjData.qfrc_actuator.
      6) cfrc_ext[1:].flatten() — external forces/torques on each body COM (6 per body), mjData.cfrc_ext.

    Slice references (when observation_structure is set on the env): obs_qpos_part, obs_qvel_part,
    obs_cinert_part, obs_cvel_part, obs_qfrc_actuator_part, obs_cfrc_ext_part align with these
    blocks in order.

    Full mjData.qpos (nq ≈ 24): free joint (torso position x,y,z + quaternion) then hinge joint
    angles (abdomen, hips, knees, shoulders, elbows) — see Gymnasium Humanoid observation table.

    Full mjData.qvel (nv ≈ 23): generalized velocities matching qpos.

    ---------------------------------------------------------------------------
    torso_xy — qpos[0:2], horizontal position of torso (excluded from default obs).
    mass_center_xy — global COM x,y from body masses (reward forward motion).
    ctrl — mjData.ctrl (control signals after action; Humanoid reward uses sum of squares of ctrl).
    cfrc_ext — full (nbody, 6) external wrench per body; contact cost uses sum of squares of full tensor.
    cinert_excl_world, cvel_excl_world — same slices as in _get_obs (bodies 1..nbody-1).
    qfrc_actuator_tail — qfrc_actuator[6:], matching obs block.
    observation_structure — copy of env.unwrapped.observation_structure (byte counts per block).
    """

    def __init__(self):
        self.obs_vector = None
        self.qpos = None
        self.qvel = None
        self.dt = None
        self.torso_xy = None
        self.mass_center_xy = None
        self.ctrl = None
        self.cfrc_ext = None
        self.cinert_excl_world = None
        self.cvel_excl_world = None
        self.qfrc_actuator_tail = None
        self.observation_structure = None
        self.obs_qpos_part = None
        self.obs_qvel_part = None
        self.obs_cinert_part = None
        self.obs_cvel_part = None
        self.obs_qfrc_actuator_part = None
        self.obs_cfrc_ext_part = None

    def fill_from_env(self, env, obs_np):
        u = env.unwrapped
        self.obs_vector = np.asarray(obs_np, dtype=np.float64).copy()
        self.qpos = np.asarray(u.data.qpos, dtype=np.float64).copy()
        self.qvel = np.asarray(u.data.qvel, dtype=np.float64).copy()
        self.dt = float(u.dt)
        self.torso_xy = np.asarray(self.qpos[0:2], dtype=np.float64).copy()
        self.mass_center_xy = _mass_center_xy(u.model, u.data)
        self.ctrl = np.asarray(u.data.ctrl, dtype=np.float64).copy()
        self.cfrc_ext = np.asarray(u.data.cfrc_ext, dtype=np.float64).copy()
        self.cinert_excl_world = np.asarray(u.data.cinert[1:], dtype=np.float64).copy()
        self.cvel_excl_world = np.asarray(u.data.cvel[1:], dtype=np.float64).copy()
        self.qfrc_actuator_tail = np.asarray(u.data.qfrc_actuator[6:], dtype=np.float64).copy()

        struct = getattr(u, "observation_structure", None)
        self.observation_structure = dict(struct) if struct is not None else None
        self._slice_obs_vector_from_structure()

    def _slice_obs_vector_from_structure(self):
        """Split obs_vector into named blocks using env.observation_structure sizes."""
        if not self.observation_structure:
            self.obs_qpos_part = None
            self.obs_qvel_part = None
            self.obs_cinert_part = None
            self.obs_cvel_part = None
            self.obs_qfrc_actuator_part = None
            self.obs_cfrc_ext_part = None
            return
        o = self.obs_vector
        s = self.observation_structure
        i = 0

        def take(key):
            nonlocal i
            n = int(s.get(key, 0))
            if n <= 0:
                return None
            chunk = o[i : i + n].copy()
            i += n
            return chunk

        self.obs_qpos_part = take("qpos")
        self.obs_qvel_part = take("qvel")
        self.obs_cinert_part = take("cinert")
        self.obs_cvel_part = take("cvel")
        self.obs_qfrc_actuator_part = take("qfrc_actuator")
        self.obs_cfrc_ext_part = take("cfrc_ext")
