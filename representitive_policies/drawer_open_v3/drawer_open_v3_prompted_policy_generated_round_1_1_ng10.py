from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MetaWorldDrawerOpenV3Observation


class DrawerOpenV3Policy:
    def act(self, obs: MetaWorldDrawerOpenV3Observation):
        """
        Executes a policy to open the drawer.
        
        Strategy:
        1. Identify hand, handle, and goal positions from the observation vector.
        2. Approach the handle with the gripper open.
        3. Once close to the handle, close the gripper and move towards the goal position.
        4. If the gripper slips and distance to handle increases, automatically revert to approach phase.
        """
        # Extract the numpy observation vector from the wrapper class
        o = obs.obs_vector

        # Observation indices based on SawyerXYZEnv (MT1) layout:
        # [0:3]   : Hand (End-Effector) XYZ
        # [4:7]   : Object (Handle) XYZ - typically the center of the handle
        # [36:39] : Goal XYZ - the target position for the handle/drawer
        
        hand_pos = o[0:3]
        handle_pos = o[4:7]
        goal_pos = o[36:39]

        # Calculate vector and distance from hand to handle
        to_handle = handle_pos - hand_pos
        dist_to_handle = np.linalg.norm(to_handle)
        
        # Calculate vector from hand to goal (this defines the pull direction)
        to_goal = goal_pos - hand_pos

        # Initialize action: [dx, dy, dz, gripper_act]
        action = np.zeros(4)
        
        # Hyperparameters
        # Distance (meters) to switch from reaching to grasping/pulling
        GRASP_THRESHOLD = 0.03  
        # Proportional gain for velocity control
        KP = 10.0

        if dist_to_handle > GRASP_THRESHOLD:
            # Phase 1: Reach for the handle
            # Move towards handle, keep gripper open (-1.0)
            action[:3] = to_handle * KP
            action[3] = -1.0 
        else:
            # Phase 2: Grasp and Pull
            # We are close enough to the handle.
            # Move towards the goal (pulling the drawer), close gripper (1.0)
            # Since 'handle_pos' in the observation updates as the drawer moves,
            # 'dist_to_handle' remains small as long as we maintain the grasp,
            # keeping us in this phase.
            action[:3] = to_goal * KP
            action[3] = 1.0

        # Clip actions to the environment's expected range [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        return action