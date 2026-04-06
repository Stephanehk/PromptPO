from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldDoorOpenV3Observation


class DoorOpenV3Policy:
    def __init__(self):
        # Phases:
        # 0: Hover (Approach from top-back to clear door face)
        # 1: Align (Move directly above the handle)
        # 2: Descend (Lower gripper onto the handle)
        # 3: Grasp (Close gripper)
        # 4: Pull (Move handle to goal)
        self.phase = 0
        self.timer = 0
        
        # Hyperparameters
        # Approach from -Y (robot side) and +Z (above).
        # We start by hovering back and up to avoid hitting the door panel from the front.
        # This assumes a Top-Down grasp strategy which is more robust for horizontal levers/knobs.
        self.HOVER_OFFSET = np.array([0.0, -0.20, 0.20]) 
        self.ALIGN_OFFSET = np.array([0.0, 0.0, 0.20])
        
        # Thresholds
        self.HOVER_THRESH = 0.08
        self.ALIGN_XY_THRESH = 0.05
        self.DESCEND_THRESH = 0.04  # 4cm threshold to trigger grasp
        self.GRASP_WAIT_STEPS = 25
        
        # Gains
        self.KP = 20.0
        self.KP_PULL = 40.0 # High gain to pull open the door
        
    def act(self, obs: MetaWorldDoorOpenV3Observation):
        o = obs.obs_vector
        hand_pos = o[0:3]
        handle_pos = o[4:7]
        goal_pos = o[36:39]
        
        action = np.zeros(4, dtype=np.float32)
        gripper = -1.0 # Default Open
        
        dist_to_handle = np.linalg.norm(hand_pos - handle_pos)
        
        if self.phase == 0:
            # Phase 0: Move to high hover point (Back & Up)
            target = handle_pos + self.HOVER_OFFSET
            err = target - hand_pos
            action[:3] = err * self.KP
            
            if np.linalg.norm(err) < self.HOVER_THRESH:
                self.phase = 1
                
        elif self.phase == 1:
            # Phase 1: Move directly above handle (Align XY)
            target = handle_pos + self.ALIGN_OFFSET
            err = target - hand_pos
            action[:3] = err * self.KP
            
            # Check XY alignment
            # We want to be directly above before descending to avoid hitting the door face
            xy_dist = np.linalg.norm(target[:2] - hand_pos[:2])
            if xy_dist < self.ALIGN_XY_THRESH:
                self.phase = 2
                
        elif self.phase == 2:
            # Phase 2: Descend onto handle
            target = handle_pos
            err = target - hand_pos
            action[:3] = err * self.KP
            
            # Transition to grasp if close enough
            if dist_to_handle < self.DESCEND_THRESH:
                self.phase = 3
                self.timer = 0
                
        elif self.phase == 3:
            # Phase 3: Grasp
            # Hold position at handle and close gripper
            target = handle_pos
            err = target - hand_pos
            action[:3] = err * self.KP
            gripper = 1.0 # Close
            
            self.timer += 1
            if self.timer >= self.GRASP_WAIT_STEPS:
                self.phase = 4
                
        elif self.phase == 4:
            # Phase 4: Pull
            # Move towards the goal state (open door)
            target = goal_pos
            err = target - hand_pos
            action[:3] = err * self.KP_PULL
            gripper = 1.0 # Keep Closed
            
        action[3] = gripper
        action = np.clip(action, -1.0, 1.0)
        
        return action