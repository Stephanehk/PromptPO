from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoAntObservation


class AntV5Policy:
    def __init__(self):
        self.time = 0.0
        # Gait Parameters
        # A frequency of ~1.0 Hz is typical for stable trotting.
        self.freq = 1.0
        # Amplitudes for sine waves. Kept moderate to reduce control cost.
        self.amp_hip = 0.2
        self.amp_ankle = 0.25
        # Ankle bias: ~0.4 rad helps keep the legs slightly bent (standing) rather than flat or fully curled.
        # Range is roughly [-0.5, 1.2], so 0.4 is a comfortable middle-ground stance.
        self.bias_ankle = 0.4
        
        # PD Controller Gains
        # Low gains prevent action saturation (-1 to 1) to minimize ctrl_cost penalty.
        self.kp = 0.5
        self.kd = 0.05

    def act(self, obs: MujocoAntObservation):
        # Update internal time counter
        self.time += obs.dt
        
        # Extract Joint Positions and Velocities from observation vector
        # qpos structure in obs_vector[5:13]:
        # Indices relative to slice:
        # [0,1]: Leg 1 (Front Left) - Hip, Ankle
        # [2,3]: Leg 2 (Front Right)
        # [4,5]: Leg 3 (Back Left)
        # [6,7]: Leg 4 (Back Right)
        qpos_joints = obs.obs_vector[5:13]
        qvel_joints = obs.obs_vector[19:27]
        
        # CPG / Oscillator
        # Trot Gait: Diagonal legs move in sync.
        # Group 1: Leg 1 & Leg 4 (Phase 0)
        # Group 2: Leg 2 & Leg 3 (Phase PI)
        phase = 2 * np.pi * self.freq * self.time
        s1 = np.sin(phase)
        s2 = np.sin(phase + np.pi)
        
        # Calculate Targets
        # Heuristic: In-phase hip and ankle oscillation usually produces a rowing/walking motion
        # where the leg lifts (ankle flexes +) as it swings forward (hip +).
        
        # Leg 1 (FL) Targets
        t1_h = self.amp_hip * s1
        t1_a = self.bias_ankle + self.amp_ankle * s1
        
        # Leg 2 (FR) Targets
        t2_h = self.amp_hip * s2
        t2_a = self.bias_ankle + self.amp_ankle * s2
        
        # Leg 3 (BL) Targets
        t3_h = self.amp_hip * s2
        t3_a = self.bias_ankle + self.amp_ankle * s2
        
        # Leg 4 (BR) Targets
        t4_h = self.amp_hip * s1
        t4_a = self.bias_ankle + self.amp_ankle * s1
        
        # Construct Target Vector matched to Action Space Order
        # Action Space Map:
        # [0] Hip 4,   [1] Ankle 4
        # [2] Hip 1,   [3] Ankle 1
        # [4] Hip 2,   [5] Ankle 2
        # [6] Hip 3,   [7] Ankle 3
        
        target_vec = np.array([
            t4_h, t4_a,
            t1_h, t1_a,
            t2_h, t2_a,
            t3_h, t3_a
        ])
        
        # Construct Current State Vector matched to Action Space Order
        # Mapping from qpos_joints/qvel_joints (ordered 1, 2, 3, 4)
        current_pos = np.array([
            qpos_joints[6], qpos_joints[7], # Leg 4
            qpos_joints[0], qpos_joints[1], # Leg 1
            qpos_joints[2], qpos_joints[3], # Leg 2
            qpos_joints[4], qpos_joints[5]  # Leg 3
        ])
        
        current_vel = np.array([
            qvel_joints[6], qvel_joints[7],
            qvel_joints[0], qvel_joints[1],
            qvel_joints[2], qvel_joints[3],
            qvel_joints[4], qvel_joints[5]
        ])
        
        # PD Control Law
        # action = Kp * (target - current) - Kd * velocity
        action = self.kp * (target_vec - current_pos) - self.kd * current_vel
        
        # Clip to valid action range
        return np.clip(action, -1.0, 1.0)