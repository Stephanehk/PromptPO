from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoReacherObservation


class ReacherV5Policy:
    def act(self, obs: MujocoReacherObservation) -> np.ndarray:
        # Reacher-v5 physical parameters
        l1 = 0.1
        l2 = 0.11
        
        # Extract joint state
        # qpos[0]: joint0 angle, qpos[1]: joint1 angle
        theta = obs.qpos[0:2]
        # qvel[0]: joint0 velocity, qvel[1]: joint1 velocity
        dtheta = obs.qvel[0:2]
        
        # Extract target position
        # qpos[2]: target_x, qpos[3]: target_y
        target = obs.qpos[2:4]
        tx, ty = target[0], target[1]
        
        # --- Inverse Kinematics (IK) ---
        # 1. Clamp target to workspace radius
        max_reach = l1 + l2
        dist_sq = tx**2 + ty**2
        if dist_sq > max_reach**2:
            scale = max_reach / np.sqrt(dist_sq)
            tx *= scale
            ty *= scale
            dist_sq = tx**2 + ty**2
            
        # 2. Compute Elbow Angle (q2)
        # Law of Cosines: r^2 = l1^2 + l2^2 + 2*l1*l2*cos(q2)
        cos_q2 = (dist_sq - l1**2 - l2**2) / (2 * l1 * l2)
        # Clip for numerical stability
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        q2_mag = np.arccos(cos_q2)
        
        # Two solutions: Elbow Down and Elbow Up
        candidates_q2 = [q2_mag, -q2_mag]
        
        best_diff = np.zeros(2)
        min_dist_sq = float('inf')
        
        # 3. Choose solution minimizing angular travel (closest to current state)
        for q2_val in candidates_q2:
            # Solve for Shoulder Angle (q1)
            # x = l1*c1 + l2*c(1+2)
            # y = l1*s1 + l2*s(1+2)
            k1 = l1 + l2 * np.cos(q2_val)
            k2 = l2 * np.sin(q2_val)
            q1_val = np.arctan2(ty, tx) - np.arctan2(k2, k1)
            
            cand_config = np.array([q1_val, q2_val])
            
            # Compute wrapped angular difference to find shortest path
            diff = (cand_config - theta + np.pi) % (2 * np.pi) - np.pi
            
            # Metric: Euclidean norm of joint difference
            d_sq = np.sum(diff**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_diff = diff
                
        # --- PD Control ---
        # Gains selected to balance rapid convergence with low control cost.
        # Kp: Sufficient stiffness to reach target quickly.
        # Kd: Low damping to prevent excessive resistance (negative work) while stabilizing.
        Kp = np.array([0.5, 0.4])
        Kd = np.array([0.1, 0.1])
        
        action = Kp * best_diff - Kd * dtheta
        
        # --- Action Clamping ---
        # Limit torque magnitude to 0.3. The quadratic control penalty makes high torques
        # very expensive. 0.3 is sufficient for the lightweight arm to reach the target 
        # within 50 steps while saving significant cost compared to 0.4 or 1.0.
        action = np.clip(action, -0.3, 0.3)
        
        return action