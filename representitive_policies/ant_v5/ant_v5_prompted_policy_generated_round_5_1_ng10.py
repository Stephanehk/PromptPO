from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoAntObservation


class AntV5Policy:
    def __init__(self):
        self.time = 0.0
        
        # Gait Parameters
        # Frequency 1.5 Hz: A brisk pace that balances speed with stability.
        # dt=0.05 implies ~13 steps per cycle, which is sufficient resolution.
        self.freq = 1.5
        
        # Amplitudes (radians)
        # Hip Amplitude 0.35 (~20 deg): Provides substantial stride length.
        self.amp_hip = 0.35
        # Ankle Amplitude 0.3 (~17 deg): Provides good ground clearance during swing.
        self.amp_ankle = 0.3
        
        # Ankle Bias (radians)
        # 0.85 rad (~49 deg): Keeps legs bent "under" the body, raising the torso 
        # and providing a stable support polygon.
        # The joint range is [30, 70] deg -> [0.52, 1.22] rad.
        # Target range [0.55, 1.15] stays well within physical limits.
        self.bias_ankle = 0.85
        
        # PD Controller Gains
        # KP=1.0: Stiffer gains are required for the Ant to track the trajectory
        # against gravity and contact forces at 1.5 Hz.
        self.kp = 1.0
        self.kd = 0.05

    def act(self, obs: MujocoAntObservation):
        self.time += obs.dt
        
        # Extract Joint Data
        # qpos indices in obs [5:13] map to legs 1,2,3,4 sequentially (Hip, Ankle pairs).
        # Leg 1: FL, Leg 2: FR, Leg 3: BL, Leg 4: BR.
        qpos_joints = obs.obs_vector[5:13]
        qvel_joints = obs.obs_vector[19:27]
        
        # Construct Current State Vectors (Reordered to match Action Space)
        # Action Space Order: [Hip4, Ank4, Hip1, Ank1, Hip2, Ank2, Hip3, Ank3]
        
        # qpos indices: L1(0,1), L2(2,3), L3(4,5), L4(6,7)
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
        
        # Oscillator Phase
        p = 2 * np.pi * self.freq * self.time
        s = np.sin(p)
        c = np.cos(p)
        
        # Gait Logic: Diagonal Trot
        # Pair A: Leg 1 (FL) & Leg 4 (BR)
        # Pair B: Leg 2 (FR) & Leg 3 (BL)
        
        # Ankle Logic (Lift/Support)
        # Support phase: Low angle (Extended).
        # Swing phase: High angle (Flexed).
        # Pair A Support (0..pi, s>0): Target = Bias - Amp*s.
        # Pair B Support (pi..2pi, s<0): Target = Bias + Amp*s.
        
        t_a1 = self.bias_ankle - self.amp_ankle * s
        t_a4 = self.bias_ankle - self.amp_ankle * s
        t_a2 = self.bias_ankle + self.amp_ankle * s
        t_a3 = self.bias_ankle + self.amp_ankle * s
        
        # Hip Logic (Propulsion)
        # To propel the body +X (Forward), the foot must sweep "Backwards" relative to the body
        # while in contact with the ground (Support Phase).
        #
        # Leg 1 (FL) & Leg 3 (BL): Neutral axes angled such that CCW (+) rotation moves foot Back.
        # Support Phase Target Rate must be Positive (+).
        # Leg 2 (FR) & Leg 4 (BR): Neutral axes angled such that CW (-) rotation moves foot Back.
        # Support Phase Target Rate must be Negative (-).
        #
        # Pair A (1 & 4) Support (0..pi):
        # L1 (+) rate: -cos(p) (Slope -1->1 is positive).
        # L4 (-) rate: +cos(p) (Slope 1->-1 is negative).
        #
        # Pair B (2 & 3) Support (pi..2pi):
        # L2 (-) rate: -cos(p) (Slope 1->-1 is negative).
        # L3 (+) rate: +cos(p) (Slope -1->1 is positive).
        #
        # Result:
        # L1, L2: -Amp * cos(p)
        # L3, L4: +Amp * cos(p)
        
        t_h1 = -self.amp_hip * c
        t_h2 = -self.amp_hip * c
        t_h3 = self.amp_hip * c
        t_h4 = self.amp_hip * c
        
        # Assemble Targets in Action Space Order
        targets = np.array([
            t_h4, t_a4,
            t_h1, t_a1,
            t_h2, t_a2,
            t_h3, t_a3
        ])
        
        # PD Control
        action = self.kp * (targets - current_pos) - self.kd * current_vel
        
        return np.clip(action, -1.0, 1.0)