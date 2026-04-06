from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldPickPlaceV3Observation


class PickPlaceV3Policy:
    def __init__(self):
        # Control parameters
        self.kp = 10.0                  # Position control gain
        
        # Thresholds for state transitions
        self.grasp_thresh_xy = 0.02     # Max XY error to start descending
        self.grasp_thresh_z = 0.02      # Max Z error to trigger grasping
        self.obj_lifted_thresh = 0.04   # Object Z height indicating successful lift
        self.gripper_closed_thresh = 0.03 # Gripper width threshold to assume grasp/closed
        
        # Target offsets
        self.lift_height = 0.15         # Lift height for carry
        self.goal_near_thresh = 0.05    # XY distance to goal to begin final descent

    def act(self, obs):
        """
        Calculates the action based on the observation for pick-place-v3.
        
        Args:
            obs (MetaWorldPickPlaceV3Observation): The observation bundle.
        
        Returns:
            np.ndarray: Action vector [dx, dy, dz, gripper].
        """
        # Unwrap observation vector from the wrapper class
        if hasattr(obs, 'obs_vector'):
            obs_vec = obs.obs_vector
        else:
            # Fallback if raw array is passed
            obs_vec = obs

        # Parse observation vector (length 39)
        # [0:3] Hand XYZ
        # [3]   Gripper open width (approx 0-1 or 0-0.04 depending on env normalization, usually ~0.04 is open)
        # [4:7] Object XYZ
        # [36:39] Goal XYZ
        hand_pos = obs_vec[0:3]
        gripper_val = obs_vec[3]
        obj_pos = obs_vec[4:7]
        goal_pos = obs_vec[36:39]

        # Calculate error metrics
        to_obj = obj_pos - hand_pos
        dist_xy = np.linalg.norm(to_obj[:2])
        dist_z = hand_pos[2] - obj_pos[2] # Positive if hand is above object

        # Initialize action
        action_xyz = np.zeros(3)
        gripper_action = -1.0 # Default Open

        # --- State Machine ---
        
        # 1. Check if we have successfully lifted the object
        if obj_pos[2] > self.obj_lifted_thresh:
            # STATE: CARRY
            # Object is in air. Move to goal.
            gripper_action = 1.0 # Keep closed
            
            # Determine target position
            target = goal_pos.copy()
            
            # If far from goal XY, stay elevated to avoid collisions
            dist_goal_xy = np.linalg.norm(goal_pos[:2] - obj_pos[:2])
            if dist_goal_xy > self.goal_near_thresh:
                # Ensure we stay at lift height until above goal
                target[2] = max(target[2], self.lift_height)
            
            action_xyz = (target - hand_pos) * self.kp
            
        else:
            # STATE: PICK
            # Object is on the table/shelf.
            
            if dist_xy > self.grasp_thresh_xy:
                # Sub-state: APPROACH (Align XY)
                # Hover above the object
                target = obj_pos.copy()
                target[2] += self.lift_height
                
                action_xyz = (target - hand_pos) * self.kp
                gripper_action = -1.0 # Open
                
            elif dist_z > self.grasp_thresh_z:
                # Sub-state: DESCEND (Align Z)
                # XY is aligned, move down to object
                target = obj_pos.copy()
                
                action_xyz = (target - hand_pos) * self.kp
                gripper_action = -1.0 # Open
                
            else:
                # Sub-state: GRASP
                # Hand is at the object.
                
                # Logic: If gripper is still wide open, close it while staying put.
                # If gripper is closed (grasping), start lifting.
                if gripper_val > self.gripper_closed_thresh:
                    # Closing phase
                    target = obj_pos.copy() # Stay at object
                    action_xyz = (target - hand_pos) * self.kp
                    gripper_action = 1.0 # Close
                else:
                    # Lifting phase
                    target = obj_pos.copy()
                    target[2] += self.lift_height
                    action_xyz = (target - hand_pos) * self.kp
                    gripper_action = 1.0 # Keep closed

        # Construct and clip final action
        action = np.concatenate([action_xyz, [gripper_action]])
        action = np.clip(action, -1.0, 1.0)
        
        return action