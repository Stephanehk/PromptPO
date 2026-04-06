import numpy as np

class ButtonPressV3Policy:
    def act(self, obs):
        """
        Policy for the Meta-World button-press-v3 task.
        
        Fixed to correctly access the observation vector from the MetaWorldButtonPressV3Observation wrapper.
        
        Observations (from wrapper.obs_vector):
        - obs[0:3]: End-effector (TCP) position (XYZ).
        - obs[3]: Gripper openness state (0.0 = closed, 1.0 = open).
        - obs[4:7]: Button object position (XYZ).
        
        Action Space:
        - Box(-1, 1, (4,)): [dx, dy, dz, gripper_ctrl]
        """
        # Fix: Access the numpy vector from the observation object
        o = obs.obs_vector
        
        # Extract state
        tcp_pos = o[0:3]
        button_pos = o[4:7]
        
        # Control parameters
        kp = 5.0                
        align_threshold = 0.03  # 3cm tolerance for X/Z alignment
        approach_dist = 0.12    # Distance to stay back during alignment
        push_dist = 0.05        # Distance to push past the button
        
        # Calculate alignment error (X and Z axes)
        # Button press vector is along the Y axis.
        err_x = abs(tcp_pos[0] - button_pos[0])
        err_z = abs(tcp_pos[2] - button_pos[2])
        is_aligned = (err_x < align_threshold) and (err_z < align_threshold)
        
        # Determine target position
        target = button_pos.copy()
        
        if is_aligned:
            # Phase 2: Push
            # Drive TCP 'into' the wall/button (Positive Y direction)
            target[1] += push_dist
        else:
            # Phase 1: Approach / Align
            # Target a point in front of the button (Negative Y offset)
            target[1] -= approach_dist
        
        # Compute position action (P-control)
        error_vec = target - tcp_pos
        action_xyz = np.clip(error_vec * kp, -1.0, 1.0)
        
        # Compute gripper action
        # The reward function `button_press_v3_reward_v2` multiplies by `tcp_closed = max(obs[3], 0.0)`.
        # In Meta-World, `obs[3]` is gripper openness (1.0 = open).
        # We want to maximize this term, so we keep the gripper OPEN.
        # Action -1.0 corresponds to opening the gripper.
        action_gripper = -1.0
        
        return np.concatenate([action_xyz, [action_gripper]])