from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoSwimmerObservation


class SwimmerV5Policy:
    """
    A policy for Gymnasium Swimmer-v5 that implements a sinusoidal swimming gait.
    It uses a PD controller to track a traveling wave pattern across the two joints.
    """
    def __init__(self):
        self.current_time = 0.0
        
        # Gait parameters
        # Amplitude of the joint oscillation (radians)
        self.amplitude = 1.2
        # Frequency of the oscillation (rad/s)
        self.frequency = 3.5
        # Phase lag between the first and second joint to create a traveling wave
        # A lag of pi/2 is standard for 3-link swimmers to propel forward.
        self.phase_lag = np.pi / 2.0
        
        # PD Controller gains
        # We use relatively high gains to track the trajectory closely, 
        # relying on the action limit clipping to saturate max torque when needed.
        self.kp = 20.0
        self.kd = 2.0

    def act(self, obs: MujocoSwimmerObservation) -> np.ndarray:
        """
        Produce an action based on the current observation.
        
        Args:
            obs: MujocoSwimmerObservation instance containing obs_vector and dt.
            
        Returns:
            np.ndarray: Action vector of shape (2,) in range [-1, 1].
        """
        # Update internal time
        # Use obs.dt (frame_skip * timestep) if available, else default to standard 0.01s
        dt = obs.dt if obs.dt is not None else 0.01
        self.current_time += dt
        
        # Extract current joint angles and velocities from obs_vector
        # Based on MujocoSwimmerObservation semantics:
        # Index 1: motor1_rot
        # Index 2: motor2_rot
        # Index 6: motor1_rot_vel
        # Index 7: motor2_rot_vel
        q_current = obs.obs_vector[1:3]
        dq_current = obs.obs_vector[6:8]
        
        # Calculate target joint angles and velocities
        # We define a traveling wave: Joint 1 oscillates, Joint 2 lags behind.
        phases = np.array([0.0, -self.phase_lag])
        arg = self.frequency * self.current_time + phases
        
        q_target = self.amplitude * np.sin(arg)
        dq_target = self.amplitude * self.frequency * np.cos(arg)
        
        # PD Control law: u = Kp * (q_tgt - q_cur) + Kd * (dq_tgt - dq_cur)
        action = self.kp * (q_target - q_current) + self.kd * (dq_target - dq_current)
        
        # Clip action to the environment's action space limits [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)