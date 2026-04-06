from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldPickPlaceV3Observation


class PickPlaceV3Policy:
    """
    Refined Policy for Meta-World pick-place-v3.
    
    Optimizations to maximize reward:
    - Tight XY alignment tolerance (0.01) to ensure gripper fingers clear the object edges.
    - Robust Grasp logic: Relaxed Z-trigger (0.03) combined with active downward bias 
      ensures a firm grasp even if collision geometry prevents perfect centering.
    - Retry mechanism: Detects failed lifts (hand high, object low) and restarts approach.
    - Object-centric Transport: Drives the object directly to the goal with collision avoidance.
    """
    def __init__(self):
        # FSM States: 0=Approach, 1=Descend, 2=Grasp, 3=Lift, 4=Transport
        self.state = 0
        self.grasp_timer = 0
        
        # Control parameters
        self.kp = 25.0
        
        # Thresholds
        self.xy_align_tol = 0.01   # Tight tolerance prevents finger collision
        self.z_grasp_tol = 0.03    # Trigger grasp phase when close in Z
        self.lift_success_z = 0.04 # Z height to confirm object is lifted
        self.goal_xy_tol = 0.02    # Precision for placement
        
        # Timers
        self.grasp_dwell = 10      # Steps to wait for gripper to stabilize
        
        # Geometry / Target Offsets
        self.approach_height = 0.12 # Hover height
        self.grasp_depth = 0.02     # Push TCP below object center to grasp
        self.transport_z = 0.15     # Safe travel height
        
    def act(self, obs):
        # Unwrap observation
        if hasattr(obs, 'obs_vector'):
            obs_vec = obs.obs_vector
        else:
            obs_vec = obs

        # Parse observation
        hand_pos = obs_vec[0:3]
        obj_pos = obs_vec[4:7]
        goal_pos = obs_vec[36:39]
        
        # Action defaults
        action_xyz = np.zeros(3)
        gripper = -1.0 # Open
        
        # --- Error Recovery ---
        # 1. Dropped object during transport
        if self.state == 4 and obj_pos[2] < 0.025:
            self.state = 0
            self.grasp_timer = 0
            
        # --- Finite State Machine ---
        
        if self.state == 0: # APPROACH
            # Move to hover position above object
            target = obj_pos.copy()
            target[2] += self.approach_height
            
            action_xyz = (target - hand_pos) * self.kp
            gripper = -1.0
            
            # Check XY alignment
            xy_dist = np.linalg.norm(target[:2] - hand_pos[:2])
            if xy_dist < self.xy_align_tol:
                self.state = 1
                
        elif self.state == 1: # DESCEND
            # Move straight down
            target = obj_pos.copy()
            target[2] -= 0.01 # Aim slightly below center to ensure we reach threshold
            
            action_xyz = (target - hand_pos) * self.kp
            gripper = -1.0
            
            # Check Z distance (hand relative to object)
            z_dist = hand_pos[2] - obj_pos[2]
            if z_dist < self.z_grasp_tol:
                self.state = 2
                
        elif self.state == 2: # GRASP
            # Apply downward pressure and close gripper
            target = obj_pos.copy()
            target[2] -= self.grasp_depth 
            
            action_xyz = (target - hand_pos) * self.kp
            gripper = 1.0 # Close
            
            self.grasp_timer += 1
            if self.grasp_timer >= self.grasp_dwell:
                self.state = 3
                
        elif self.state == 3: # LIFT
            # Raise object
            target = obj_pos.copy()
            target[2] = self.transport_z
            
            action_xyz = (target - hand_pos) * self.kp
            gripper = 1.0
            
            # Success check
            if obj_pos[2] > self.lift_success_z:
                self.state = 4
            
            # Failed grasp check: Hand is high, Object is low
            # Retry if hand is 5cm above object and object is still on table
            if hand_pos[2] > (obj_pos[2] + 0.05) and obj_pos[2] < 0.03:
                self.state = 0
                self.grasp_timer = 0
                
        elif self.state == 4: # TRANSPORT
            # Object-centric control: Drive object to goal
            target_obj = goal_pos.copy()
            
            # Path planning: Stay high until close to goal XY
            obj_goal_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
            
            if obj_goal_dist > self.goal_xy_tol:
                # Maintain safe height (clear goal bin or table obstacles)
                safe_z = max(self.transport_z, goal_pos[2] + 0.1)
                target_obj[2] = safe_z
            else:
                # Descend to goal
                target_obj[2] = goal_pos[2]
            
            # Command hand based on object error
            action_xyz = (target_obj - obj_pos) * self.kp
            gripper = 1.0
            
        action = np.concatenate([action_xyz, [gripper]])
        return np.clip(action, -1.0, 1.0)