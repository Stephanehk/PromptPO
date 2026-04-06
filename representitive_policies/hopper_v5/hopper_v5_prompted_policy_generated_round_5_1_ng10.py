from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoHopperObservation


class HopperV5Policy:
    """
    Policy for Hopper-v5 implementing a Raibert-style hopping controller.
    
    Strategy:
    1. Hopping Cycle: Uses a height threshold (1.22m) to switch between Leg Retraction (Flight)
       and Leg Extension (Stance). The threshold is tuned to be slightly below the spawn height (1.25m)
       but above the crouch height, ensuring the cycle initiates by dropping first, then kicking.
    2. Balance (Pitch Control): A strong P-gain on the thigh angle modulates the foot position
       based on torso pitch. If the torso leans forward, the leg swings forward to catch it.
       This is critical as Hopper-v5 terminates if pitch exceeds +/- 0.2 rad.
    3. Forward Velocity: Adjusts the neutral thigh angle based on velocity error.
       If moving too slow, the leg is angled backward during placement to generate forward thrust.
    """

    def __init__(self):
        # --- Target Parameters ---
        self.target_vel = 1.5         # Target forward velocity (m/s)
        self.thigh_neutral = 0.45     # Nominal thigh angle (~26 deg) for standing balance
        
        # --- State Machine Thresholds ---
        # Spawn height is ~1.25m.
        # Trigger Stance (Kick) when z < 1.22m.
        # Trigger Flight (Retract) when z >= 1.22m.
        self.z_threshold = 1.22
        
        # --- Target Angles (Radians) ---
        self.leg_push_target = 2.6    # Extension for thrust
        self.leg_pull_target = 0.9    # Retraction for clearance
        
        # --- Control Gains ---
        # Pitch Gain: High gain ensures the leg responds aggressively to small leans.
        self.k_pitch = 3.0
        
        # Velocity Gain: Modulates foot placement for speed.
        # Kept moderate to prevent destabilizing kicks (0.15 rad per m/s error).
        self.k_vel = 0.15
        
        # PD Gains for Joint Tracking
        self.kp_thigh = 5.0
        self.kd_thigh = 0.2
        
        self.kp_leg = 8.0
        self.kd_leg = 0.1
        
        self.kp_foot = 2.0
        self.kd_foot = 0.1

    def act(self, obs):
        # Extract observation
        if isinstance(obs, MujocoHopperObservation):
            o = obs.obs_vector
        else:
            o = obs
            
        # Observation Mapping (Hopper-v5):
        # [0]z, [1]pitch, [2]thigh, [3]leg, [4]foot
        # [5]vx, [8]w_thigh, [9]w_leg, [10]w_foot
        z = o[0]
        pitch = o[1]
        thigh = o[2]
        leg = o[3]
        foot = o[4]
        vx = o[5]
        
        d_thigh = o[8]
        d_leg = o[9]
        d_foot = o[10]

        # --- 1. Vertical Control (Hopping) ---
        # State machine based on height to induce limit cycle
        if z < self.z_threshold:
            # Stance: Thrust
            target_leg = self.leg_push_target
        else:
            # Flight: Retract
            target_leg = self.leg_pull_target
            
        action_leg = self.kp_leg * (target_leg - leg) - self.kd_leg * d_leg

        # --- 2. Horizontal Control (Thigh/Balance) ---
        # Calculate modifiers for foot placement
        
        # Pitch Correction: Lean forward (pitch > 0) -> Leg forward (increase thigh) -> Restore
        pitch_term = self.k_pitch * pitch
        
        # Velocity Correction: Too slow (vx < target) -> Leg backward (decrease thigh) -> Accel
        vel_term = self.k_vel * (vx - self.target_vel)
        vel_term = np.clip(vel_term, -0.4, 0.4) # Safety clamp to prevent flips
        
        target_thigh = self.thigh_neutral + pitch_term + vel_term
        
        # Mechanical limits clamp
        target_thigh = np.clip(target_thigh, -0.5, 1.5)
        
        action_thigh = self.kp_thigh * (target_thigh - thigh) - self.kd_thigh * d_thigh

        # --- 3. Foot Attitude ---
        # Maintain rigid foot
        action_foot = self.kp_foot * (0.0 - foot) - self.kd_foot * d_foot

        # Assemble Action
        action = np.array([action_thigh, action_leg, action_foot], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)