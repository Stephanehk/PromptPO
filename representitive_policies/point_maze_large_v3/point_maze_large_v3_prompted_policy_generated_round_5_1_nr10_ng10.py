from abc import ABCMeta
import numpy as np
from collections import deque
from direct_policy_learning.observations import PointMazeLargeV3Observation


class PointMazeLargeV3Policy:
    def __init__(self):
        # PD Controller Parameters
        # Navigation Phase: Used for general traversal.
        # Kp=12.0: High enough to saturate the action (force) when error is large.
        # Kd=3.0: Provides damping to prevent oscillations and stabilize movement.
        self.kp_nav = 12.0
        self.kd_nav = 3.0
        
        # Terminal Phase Parameters
        # When the agent is within this distance of the goal, we switch to "Terminal Mode".
        # The reward threshold is 0.45m. We switch at 0.75m to ensure we have momentum
        # and full force applied to cross the line.
        self.terminal_dist = 0.75
        
        # Stuck Recovery Parameters
        # Detection window is shortened to 20 steps (2 seconds) to react faster to hang-ups.
        self.pos_history = deque(maxlen=20)
        self.stuck_std_threshold = 0.05
        self.recovery_duration = 20
        self.recovery_steps = 0
        self.recovery_action = np.zeros(2)

    def act(self, obs: PointMazeLargeV3Observation):
        # Extract state from the customized observation class
        # Achieved Goal: Current [x, y] position
        current_pos = np.array(obs.achieved_goal, dtype=np.float64)
        # Desired Goal: Target [x, y] position
        target_pos = np.array(obs.desired_goal, dtype=np.float64)
        # Velocity: extracted from the observation vector [x, y, vx, vy]
        velocity = np.array(obs.observation_vector[2:4], dtype=np.float64)
        
        # --- 1. Teleport/Reset Detection ---
        # If the position changes abruptly (e.g., new episode), reset the history
        # to prevent false triggering of stuck detection.
        if self.pos_history:
            if np.linalg.norm(current_pos - self.pos_history[-1]) > 2.0:
                self.pos_history.clear()
                self.recovery_steps = 0
        self.pos_history.append(current_pos)
        
        # --- 2. Stuck Detection ---
        # If we are not currently recovering and the history buffer is full:
        if self.recovery_steps == 0 and len(self.pos_history) == self.pos_history.maxlen:
            # Calculate standard deviation of position to detect stagnation
            pos_arr = np.array(self.pos_history)
            std_dev = np.mean(np.std(pos_arr, axis=0))
            
            if std_dev < self.stuck_std_threshold:
                # Agent is stuck. Initiate random recovery force.
                self.recovery_steps = self.recovery_duration
                angle = np.random.uniform(0, 2 * np.pi)
                self.recovery_action = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
                self.pos_history.clear()
        
        # --- 3. Control Logic ---
        if self.recovery_steps > 0:
            # Execute open-loop recovery
            action = self.recovery_action
            self.recovery_steps -= 1
        else:
            # Calculate error to goal
            error = target_pos - current_pos
            dist = np.linalg.norm(error)
            
            if dist < self.terminal_dist:
                # Terminal Phase: Bang-Bang Control
                # We are close to the goal. Remove damping (Kd=0) and apply maximum 
                # force (normalized error vector) to ensure we penetrate the 0.45m 
                # reward radius. This overcomes steady-state errors or friction.
                if dist > 1e-5:
                    action = error / dist
                else:
                    action = np.zeros(2)
            else:
                # Navigation Phase: PD Control
                # Standard proportional-derivative control to navigate towards goal.
                action = self.kp_nav * error - self.kd_nav * velocity
        
        # Clip action to valid range [-1.0, 1.0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)