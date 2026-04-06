from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldDrawerOpenV3Observation


class DrawerOpenV3Policy:
    def __init__(self):
        # State machine to handle multi-stage task:
        # 0: Approach handle
        # 1: Grasp (close gripper and wait for stability)
        # 2: Pull drawer open
        self.stage = 0
        self.timer = 0

    def act(self, obs: MetaWorldDrawerOpenV3Observation):
        """
        Policy to open the drawer.
        
        Uses a state machine to ensure the grasp is established before pulling.
        Maintains a stiff position control on the handle during the pull phase to prevent slipping.
        """
        # Extract observation vector
        # [0:3]   : Hand XYZ
        # [4:7]   : Handle XYZ (updates as drawer moves)
        # [36:39] : Goal XYZ
        o = obs.obs_vector
        hand_pos = o[0:3]
        handle_pos = o[4:7]
        goal_pos = o[36:39]

        # Vector from hand to handle (error for position control)
        to_handle = handle_pos - hand_pos
        dist_handle = np.linalg.norm(to_handle)

        # Reset logic:
        # If the hand ends up far from the handle (e.g. new episode start, or slip),
        # reset the state machine to Approach.
        if dist_handle > 0.15:
            self.stage = 0
            self.timer = 0

        action = np.zeros(4)

        # Hyperparameters
        APPROACH_THRESH = 0.025  # Distance to trigger grasp phase
        GRASP_WAIT = 15          # Steps to wait for gripper to fully close
        
        KP_APPROACH = 10.0       # Approach gain
        KP_HOLD = 40.0           # High gain to maintain grasp relative to moving handle
        PULL_SPEED = 0.5         # Velocity bias to pull the drawer

        if self.stage == 0:
            # Phase 1: Approach
            # Servo towards handle, gripper open
            action[:3] = to_handle * KP_APPROACH
            action[3] = -1.0
            
            if dist_handle < APPROACH_THRESH:
                self.stage = 1
                self.timer = 0

        elif self.stage == 1:
            # Phase 2: Grasp
            # Stay at handle position, close gripper
            action[:3] = to_handle * KP_HOLD
            action[3] = 1.0
            
            # Wait for gripper to close physically
            self.timer += 1
            if self.timer >= GRASP_WAIT:
                self.stage = 2

        elif self.stage == 2:
            # Phase 3: Pull
            # Determine pull direction (Handle -> Goal)
            to_goal = goal_pos - handle_pos
            dist_goal = np.linalg.norm(to_goal)
            
            if dist_goal > 1e-4:
                pull_dir = to_goal / dist_goal
            else:
                pull_dir = np.zeros(3)
            
            # Action combines:
            # 1. Stiff position control to stay attached to handle (to_handle * KP_HOLD)
            # 2. Velocity bias to move the drawer towards goal (pull_dir * PULL_SPEED)
            action[:3] = to_handle * KP_HOLD + pull_dir * PULL_SPEED
            action[3] = 1.0 # Maintain closed gripper

        # Clip action to valid range [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        return action