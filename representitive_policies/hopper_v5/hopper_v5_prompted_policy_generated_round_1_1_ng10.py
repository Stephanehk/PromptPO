from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoHopperObservation


class HopperV5Policy:
    """
    Policy for Hopper-v5 that implements a PD-based locomotion controller.
    Strategies:
    1. Pitch Control: Regulates torso angle by modulating thigh target (swing leg to catch fall).
    2. Hopping: Uses a state machine based on torso height (z) to switch between leg extension (thrust) and retraction (flight).
    3. Forward Bias: Maintains a base forward thigh angle to ensure forward displacement.
    """

    def __init__(self):
        # PD Gains
        self.kp_thigh = 3.0
        self.kd_thigh = 0.2
        
        self.kp_leg_ground = 5.0
        self.kp_leg_air = 1.0
        self.kd_leg = 0.1
        
        self.kp_foot = 1.5
        self.kd_foot = 0.2
        
        # Targets & Thresholds
        self.target_torso = 0.0
        # Base thigh angle: positive swings leg forward. 
        # A value of ~0.5 rad (~28 deg) encourages forward landing stability.
        self.target_thigh_base = 0.5 
        # Pitch gain: How aggressively to swing leg in response to torso lean.
        # High gain keeps the hopper within the tight healthy angle range (-0.2, 0.2).
        self.pitch_gain = 3.5 
        
        # Height threshold to trigger leg extension (hopping thrust).
        # Initial height is ~1.25. Drop below 1.10 implies compression/landing.
        self.z_ground_threshold = 1.10
        self.leg_extended_pos = 2.8  # Push ground
        self.leg_retracted_pos = 1.0 # Clear ground

    def act(self, obs):
        # Extract observation vector
        if isinstance(obs, MujocoHopperObservation):
            obs_vec = obs.obs_vector
        else:
            obs_vec = obs

        # Observation mapping for Hopper-v5 (default 11 dims):
        # [0] rootz (height)
        # [1] rooty (torso angle)
        # [2] thigh_joint angle
        # [3] leg_joint angle
        # [4] foot_joint angle
        # [5-10] velocities (rootx, rootz, rooty, thigh, leg, foot)
        
        z = obs_vec[0]
        torso_angle = obs_vec[1]
        thigh_angle = obs_vec[2]
        leg_angle = obs_vec[3]
        foot_angle = obs_vec[4]
        
        # Velocities for D-term
        w_thigh = obs_vec[8]
        w_leg = obs_vec[9]
        w_foot = obs_vec[10]

        # --- 1. Thigh / Torso Control ---
        # Regulate torso angle to 0. Correct errors by moving the leg.
        # If torso leans forward (angle > 0), swing leg forward (increase thigh angle) to catch fall.
        target_thigh = self.target_thigh_base + (self.pitch_gain * (torso_angle - self.target_torso))
        action_thigh = self.kp_thigh * (target_thigh - thigh_angle) - self.kd_thigh * w_thigh

        # --- 2. Leg / Hopping Control ---
        # Extend when low (stance), retract when high (flight).
        if z < self.z_ground_threshold:
            # Ground phase: Thrust
            target_leg = self.leg_extended_pos
            kp_leg = self.kp_leg_ground
        else:
            # Air phase: Retract
            target_leg = self.leg_retracted_pos
            kp_leg = self.kp_leg_air
            
        action_leg = kp_leg * (target_leg - leg_angle) - self.kd_leg * w_leg

        # --- 3. Foot Control ---
        # Keep foot relatively rigid/neutral
        target_foot = 0.0
        action_foot = self.kp_foot * (target_foot - foot_angle) - self.kd_foot * w_foot

        # Construct and clip action
        action = np.array([action_thigh, action_leg, action_foot], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)