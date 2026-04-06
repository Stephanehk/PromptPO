from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoHalfCheetahObservation


class HalfcheetahV5Policy:
    """
    Tuned CPG policy for HalfCheetah-v5.
    
    Improvements based on reflection:
    1.  Increased frequency (omega=24.0) to drive higher velocity compared to Attempt 3.
    2.  Optimized PD gains (Kp=1.8, Kd=0.08) to improve trajectory tracking at higher speeds
        without the instability/flipping seen in Attempt 4.
    3.  Maintained the effective phase relationship from Attempt 3 (Shin leads Thigh by ~pi/2),
        which correctly aligns leg extension with the backward driving stroke (stance phase)
        and flexion with the forward swing (ground clearance).
    4.  Removed non-zero offsets which caused instability in prior attempts.
    """
    def __init__(self):
        self.t = 0.0
        # Angular frequency: 24.0 rad/s (~3.8 Hz)
        self.omega = 24.0
        
        # PD Gains
        # Thighs drive the motion (Kp=1.8).
        # Shins/Feet have slightly lower gains to allow some ground compliance.
        self.kp = np.array([1.8, 1.5, 1.0, 1.8, 1.5, 1.0], dtype=np.float64)
        self.kd = np.array([0.08, 0.08, 0.05, 0.08, 0.08, 0.05], dtype=np.float64)
        
        # Trajectory Parameters (Amplitudes in radians)
        # Thighs (1.0) and Shins (0.8) have large range of motion for stride length.
        # Feet (0.5) oscillate to assist push-off.
        self.amp = np.array([1.0, 0.8, 0.5, 1.0, 0.8, 0.5], dtype=np.float64)
        
        # Offsets (radians)
        # Kept neutral (0.0) to maintain stability and prevent the flipping
        # observed in Attempt 4.
        self.offset = np.zeros(6, dtype=np.float64)
        
        # Phase Offsets (radians)
        # Bounding gait: Front legs are pi out of phase with Back legs.
        # Intra-leg coordination: Shin/Foot lead Thigh by pi/2 (1.57).
        # This ensures the leg is extending (pushing) as the thigh swings backward,
        # and flexing (lifting) as the thigh swings forward.
        lead = 1.57
        bound_phase = np.pi
        
        self.phase_offsets = np.array([
            0.0,                # bthigh
            lead,               # bshin
            lead,               # bfoot
            bound_phase,        # fthigh
            bound_phase + lead, # fshin
            bound_phase + lead  # ffoot
        ], dtype=np.float64)

    def act(self, obs: MujocoHalfCheetahObservation):
        # Update internal timer
        self.t += obs.dt
        
        # 1. Calculate Target Positions (CPG)
        phase = self.omega * self.t
        q_target = self.offset + self.amp * np.sin(phase + self.phase_offsets)
        
        # 2. Get Current State
        # obs_vector[2:8] -> [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        q_curr = obs.obs_vector[2:8]
        # obs_vector[11:17] -> Velocities of corresponding joints
        v_curr = obs.obs_vector[11:17]
        
        # 3. PD Control
        # Action is torque. Damping term acts on velocity to stabilize.
        action = self.kp * (q_target - q_curr) - self.kd * v_curr
        
        # 4. Clip to valid range
        return np.clip(action, -1.0, 1.0)