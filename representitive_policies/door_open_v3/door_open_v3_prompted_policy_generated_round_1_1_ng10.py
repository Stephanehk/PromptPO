from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldDoorOpenV3Observation


class DoorOpenV3Policy:
    def __init__(self):
        # State machine phases: 0 = Approach, 1 = Grasp, 2 = Pull
        self.phase = 0
        self.grasp_timer = 0
        
        # Hyperparameters
        self.APPROACH_THRESH = 0.02  # Distance threshold to switch to grasp (2cm)
        self.GRASP_STEPS = 15        # Steps to wait for gripper to close fully
        self.KP = 10.0               # Proportional gain for position control

    def act(self, obs: MetaWorldDoorOpenV3Observation):
        # Extract the raw observation vector
        # Layout based on SawyerXYZEnv (MT1):
        # [0:3]   : Hand XYZ
        # [3]     : Gripper (scalar distance/state)
        # [4:7]   : Handle XYZ (first 3 dims of object features)
        # ...
        # [36:39] : Goal XYZ (at the end of the 39-dim vector)
        
        o = obs.obs_vector
        
        hand_pos = o[0:3]
        handle_pos = o[4:7]
        goal_pos = o[36:39]
        
        # Calculate distance to target handle
        dist_to_handle = np.linalg.norm(hand_pos - handle_pos)
        
        # Initialize action: [dx, dy, dz, gripper_torque]
        action = np.zeros(4, dtype=np.float32)
        
        # State Machine
        if self.phase == 0:
            # --- Phase 0: Approach ---
            # Move towards the handle position
            err = handle_pos - hand_pos
            action[:3] = err * self.KP
            
            # Keep gripper open
            action[3] = 1.0
            
            # Transition to grasp if close enough
            if dist_to_handle < self.APPROACH_THRESH:
                self.phase = 1
                
        elif self.phase == 1:
            # --- Phase 1: Grasp ---
            # Maintain position at handle to prevent drift while closing
            err = handle_pos - hand_pos
            action[:3] = err * self.KP
            
            # Close gripper
            action[3] = -1.0
            
            # Wait for gripper to fully close
            self.grasp_timer += 1
            if self.grasp_timer >= self.GRASP_STEPS:
                self.phase = 2
                
        elif self.phase == 2:
            # --- Phase 2: Pull/Open ---
            # Move towards the goal position (defined by the task as the open state)
            err = goal_pos - hand_pos
            action[:3] = err * self.KP
            
            # Keep gripper closed
            action[3] = -1.0
            
        # Clip actions to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return action