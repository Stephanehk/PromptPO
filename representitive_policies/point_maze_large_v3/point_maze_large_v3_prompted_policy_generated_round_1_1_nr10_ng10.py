from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import PointMazeLargeV3Observation


class PointMazeLargeV3Policy:
    def act(self, obs: PointMazeLargeV3Observation):
        """
        Calculates the action based on the current observation using a PD controller.
        
        Args:
            obs (PointMazeLargeV3Observation): The current observation containing
                                               position, velocity, and goal.
                                               
        Returns:
            np.ndarray: The action vector [force_x, force_y] clipped to [-1, 1].
        """
        # Extract current position (achieved_goal) and target position (desired_goal)
        # obs.achieved_goal is shape (2,) -> [x, y]
        current_pos = np.array(obs.achieved_goal, dtype=np.float64)
        target_pos = np.array(obs.desired_goal, dtype=np.float64)
        
        # Extract current velocity from observation_vector
        # obs.observation_vector is shape (4,) -> [x, y, vx, vy]
        current_vel = np.array(obs.observation_vector[2:4], dtype=np.float64)
        
        # PD Controller Parameters
        # Kp: Proportional gain to drive the agent towards the goal.
        # Kd: Derivative gain to dampen the velocity and prevent oscillation.
        Kp = 5.0
        Kd = 1.0
        
        # Calculate position error
        error = target_pos - current_pos
        
        # Compute PD control law: u = Kp * e - Kd * v
        action = Kp * error - Kd * current_vel
        
        # Clip the action to the valid range [-1.0, 1.0] as specified in the environment details
        action = np.clip(action, -1.0, 1.0)
        
        # Return action as float32
        return action.astype(np.float32)