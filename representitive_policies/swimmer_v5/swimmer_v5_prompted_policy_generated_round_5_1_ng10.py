from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoSwimmerObservation


class SwimmerV5Policy:
    """
    Optimized PD-controlled sinusoidal policy for Gymnasium Swimmer-v5.
    
    Evolution of parameters based on trajectory analysis:
    - Attempt 1 (A=1.2, F=3.5): Mean Return ~298. Good baseline.
    - Attempt 4 (A=1.05, F=3.8): Mean Return ~314. Best performance so far. The combination of 
      slightly higher frequency and reduced amplitude improved forward velocity to ~0.32 m/s 
      by managing torque saturation better.
      
    Strategy for this policy:
    - Extrapolate the successful trend from Attempt 1 to Attempt 4.
    - Increase Frequency to 4.0 rad/s to generate more propulsive cycles within the episode.
    - Decrease Amplitude to 1.0 rad to keep the peak angular velocity (A * F) around 4.0, 
      which appears to be the efficient limit for the Swimmer's actuators (torque [-1, 1]).
    - Increase PD gains slightly to ensure tracking fidelity at the higher frequency.
    """
    def __init__(self):
        self.current_time = 0.0
        
        # Gait parameters
        # Amplitude (radians) - Reduced slightly to avoid saturation at higher frequency
        self.amplitude = 1.0
        # Frequency (rad/s) - Increased to maximize strokes per episode
        self.frequency = 4.0
        # Phase lag (radians) - Standard traveling wave offset (pi/2)
        self.phase_lag = np.pi / 2.0
        
        # PD Controller gains
        # Increased slightly from 50/5 to 60/6 to track the faster trajectory
        self.kp = 60.0
        self.kd = 6.0

    def act(self, obs: MujocoSwimmerObservation) -> np.ndarray:
        """
        Produce an action based on the current observation.
        
        Args:
            obs: MujocoSwimmerObservation instance containing obs_vector and dt.
            
        Returns:
            np.ndarray: Action vector of shape (2,) in range [-1, 1].
        """
        # Update internal time
        # obs.dt is typically around 0.01s (frame_skip * model_timestep) for Swimmer-v5
        dt = obs.dt if obs.dt is not None else 0.01
        self.current_time += dt
        
        # Extract current joint angles (indexes 1, 2) and velocities (indexes 6, 7)
        # obs_vector structure from context: 
        # [0] free_rot, [1] motor1_rot, [2] motor2_rot
        # [3] tip_vx, [4] tip_vy, [5] free_vel, [6] motor1_vel, [7] motor2_vel
        q_current = obs.obs_vector[1:3]
        dq_current = obs.obs_vector[6:8]
        
        # Calculate target state for the traveling wave
        # Joint 0 follows sin(wt), Joint 1 follows sin(wt - phase_lag)
        phases = np.array([0.0, -self.phase_lag])
        arg = self.frequency * self.current_time + phases
        
        q_target = self.amplitude * np.sin(arg)
        dq_target = self.amplitude * self.frequency * np.cos(arg)
        
        # PD Control: u = Kp * error + Kd * error_dot
        action = self.kp * (q_target - q_current) + self.kd * (dq_target - dq_current)
        
        # Clip to action space limits
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)