from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoInvertedDoublePendulumObservation


class InvertedDoublePendulumV5Policy:
    def act(self, obs: MujocoInvertedDoublePendulumObservation):
        # Physical constants for Gymnasium InvertedDoublePendulum-v5 (standard MuJoCo model)
        # Pole lengths are approx 0.6m each.
        L1 = 0.6
        L2 = 0.6
        
        # 1. Extract State
        # qpos[0]: cart position (x)
        cart_x = obs.qpos[0]
        # qvel[0]: cart velocity (v_x)
        cart_v = obs.qvel[0]
        
        # Tip horizontal position (world frame)
        tip_x = obs.tip_site_pos[0]
        
        # 2. Compute "Lean" terms
        # The horizontal distance between the tip and the cart.
        # If tip_x > cart_x (lean > 0), pendulum leans right.
        lean_pos = tip_x - cart_x
        
        # Estimate the horizontal velocity of the tip relative to the cart.
        # Using linearized kinematics for upright position:
        # v_tip_rel = L1 * theta1_dot + L2 * (theta1_dot + theta2_dot)
        theta1_dot = obs.qvel[1]
        theta2_dot = obs.qvel[2]
        lean_vel = (L1 + L2) * theta1_dot + L2 * theta2_dot
        
        # 3. PD Control
        # Balancing Gains (Dominant):
        # High gains to keep the pendulum upright. 
        # Logic: If leaning right (pos > 0), push cart right (force > 0) to get under it.
        kp_lean = 60.0  
        kd_lean = 12.0
        
        # Centering/Damping Gains (Weak):
        # Pull cart back to x=0 and damp cart velocity.
        # Logic: If cart is right (x > 0), push left (force < 0).
        kp_cart = -0.05
        kd_cart = -0.5
        
        # Calculate total force
        force = (kp_lean * lean_pos + 
                 kd_lean * lean_vel + 
                 kp_cart * cart_x + 
                 kd_cart * cart_v)
        
        # 4. Clip to Action Space
        # The environment expects an action in [-1, 1]
        action = np.clip(np.array([force]), -1.0, 1.0)
        
        return action