from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoHalfCheetahObservation


class HalfcheetahV5Policy:
    """
    A heuristic policy for HalfCheetah-v5 using a Central Pattern Generator (CPG) approach.
    It generates an open-loop sinusoidal gait that coordinates the back and front legs
    to produce forward locomotion.
    """
    def __init__(self):
        self.t = 0.0
        # Angular frequency: ~15 rad/s corresponds to approx 2.4 Hz, effective for Cheetah
        self.omega = 15.0
        # Amplitude of the torque commands (actions are in [-1, 1])
        self.amplitude = 0.8

    def act(self, obs: MujocoHalfCheetahObservation):
        """
        Produces an action based on the current observation.
        
        Args:
            obs: MujocoHalfCheetahObservation instance containing state and dt.
            
        Returns:
            action: np.array of shape (6,) with torque commands.
        """
        # Increment internal timer using the environment's timestep
        self.t += obs.dt
        
        # Calculate the base phase for the cycle
        phase = self.omega * self.t
        
        # Initialize action vector
        # Indices: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        action = np.zeros(6, dtype=np.float32)
        
        # Back Leg Control
        # We use a sine wave for the thigh and a phase-shifted sine for the shin/foot
        # to create a locomotive cycle (elliptical foot trajectory).
        action[0] = self.amplitude * np.sin(phase)           # bthigh
        action[1] = self.amplitude * np.sin(phase + 1.0)     # bshin (lagging)
        action[2] = 0.5 * np.sin(phase + 0.5)                # bfoot
        
        # Front Leg Control
        # The front leg operates in anti-phase (pi offset) to the back leg to create a
        # bounding/trotting gait, which stabilizes the torso.
        phase_front = phase + np.pi
        action[3] = self.amplitude * np.sin(phase_front)         # fthigh
        action[4] = self.amplitude * np.sin(phase_front + 1.0)   # fshin
        action[5] = 0.5 * np.sin(phase_front + 0.5)              # ffoot
        
        return action