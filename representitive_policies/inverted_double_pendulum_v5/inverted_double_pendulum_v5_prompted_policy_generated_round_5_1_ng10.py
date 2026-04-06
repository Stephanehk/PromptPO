from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoInvertedDoublePendulumObservation


class InvertedDoublePendulumV5Policy:
    def act(self, obs: MujocoInvertedDoublePendulumObservation):
        """
        Policy for Gymnasium InvertedDoublePendulum-v5.
        
        Uses a PD controller tuned to stabilize the double inverted pendulum.
        Based on previous attempts, low-to-moderate gains with high damping 
        provide the best stability within the clipped action space [-1, 1].
        
        The controller applies a horizontal force to the cart.
        Sign Convention:
        - q (angle) > 0 corresponds to a Left lean (in standard MuJoCo view).
        - To correct a Left lean, the Cart must accelerate Left.
        - Force < 0 is Left.
        - Therefore, Force = - (K * q).
        """
        
        # 1. Extract State
        # qpos: [slider_pos, hinge1_angle, hinge2_angle]
        x = obs.qpos[0]
        q1 = obs.qpos[1]
        q2 = obs.qpos[2]
        
        # qvel: [slider_vel, hinge1_vel, hinge2_vel]
        dx = obs.qvel[0]
        dq1 = obs.qvel[1]
        dq2 = obs.qvel[2]
        
        # 2. Control Gains
        # Tuned to balance the poles while keeping the cart roughly centered.
        # High derivative (D) gains are used to dampen the chaotic energy of the double pendulum.
        
        # Cart Regulation
        # Minimal stiffness to center, moderate damping to prevent drift.
        k_x = 0.1
        k_dx = 0.5
        
        # Angle Regulation
        # k_q1: Stiffness for the bottom pole (carries the whole system).
        # k_q2: Stiffness for the top pole (relative joint angle).
        # Values around 4.0-5.0 utilize the action range [-1, 1] effectively 
        # for small deviations (< 15 degrees) without immediate saturation.
        k_q1 = 5.0
        k_q2 = 4.0
        
        # Angular Velocity Damping
        # Critical for survival. dampens oscillations.
        k_dq1 = 4.0
        k_dq2 = 4.0
        
        # 3. Compute Control Action
        # u = -K * x
        force_unclipped = - (k_x * x + k_dx * dx + 
                             k_q1 * q1 + k_dq1 * dq1 + 
                             k_q2 * q2 + k_dq2 * dq2)
        
        # 4. Clip to Action Space
        action = np.clip(np.array([force_unclipped]), -1.0, 1.0)
        
        return action