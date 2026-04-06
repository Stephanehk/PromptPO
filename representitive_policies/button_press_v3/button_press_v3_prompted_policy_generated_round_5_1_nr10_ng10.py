import numpy as np

class ButtonPressV3Policy:
    def act(self, obs):
        """
        Policy for Meta-World button-press-v3.
        
        Strategy:
        - Maintain an open gripper (action -1.0) to maximize the `tcp_closed` reward component 
          (which correlates with gripper width/openness in this task's reward function).
        - Use a P-controller with hysteresis to switch between an approach phase and a push phase.
        - Approach Phase: Align X/Z axes while targeting a point 2cm in front of the button (Y-).
          This ensures the TCP enters the 5cm 'near_button' reward zone rapidly.
        - Push Phase: Once aligned within tolerance (3.5cm), drive the TCP 10cm past the button 
          to ensure full depression and robust 'button_pressed' reward.
        
        Parameters tuned for high returns:
        - kp = 20.0: High enough for fast transit, low enough to avoid instability.
        - approach_dist = 0.02: Closer than previous attempts to trigger rewards earlier.
        - entry_tol = 0.035: Allows transition to push phase earlier (cutting the corner).
        """
        # Parse observation
        # obs vector: [hand_xyz (3), gripper (1), button_xyz (3), ...]
        o = obs.obs_vector
        tcp_pos = o[0:3]
        button_pos = o[4:7]
        
        # Control Parameters
        kp = 20.0
        approach_dist = 0.02    # Target 2cm in front of button (well within 5cm reward zone)
        push_dist = 0.10        # Target 10cm past button to saturate press
        
        # Tolerances
        # Reward zone is radius 0.05.
        # entry_tol logic: sqrt(entry_tol^2 + approach_dist^2) should be < 0.05
        # sqrt(0.035^2 + 0.02^2) = 0.0403 < 0.05. Safe.
        entry_tol = 0.035       # Allow push initiation when aligned within 3.5cm
        hold_tol = 0.05         # Hysteresis: continue pushing unless alignment degrades past 5cm
        
        # Calculate alignment error in X-Z plane (Button presses along Y)
        dx = tcp_pos[0] - button_pos[0]
        dz = tcp_pos[2] - button_pos[2]
        align_error = np.sqrt(dx**2 + dz**2)
        
        # State estimation for Hysteresis
        # Check if we are physically past the approach point (plus a small buffer)
        # approach point Y is (button_y - approach_dist)
        is_pressing = tcp_pos[1] > (button_pos[1] - approach_dist + 0.005)
        
        # Select active tolerance
        current_tol = hold_tol if is_pressing else entry_tol
        
        # Determine Target
        target = button_pos.copy()
        
        if align_error < current_tol:
            # Phase: Push
            # Drive deep into the button (Y+)
            target[1] += push_dist
        else:
            # Phase: Approach
            # Move to standoff position (Y-)
            target[1] -= approach_dist
            
        # Compute Control Action
        error = target - tcp_pos
        action_xyz = np.clip(error * kp, -1.0, 1.0)
        
        # Gripper Action
        # -1.0 forces gripper open. 
        # The reward function term `tcp_closed = max(obs[3], 0)` rewards high `obs[3]` (openness).
        action_gripper = -1.0
        
        return np.concatenate([action_xyz, [action_gripper]])