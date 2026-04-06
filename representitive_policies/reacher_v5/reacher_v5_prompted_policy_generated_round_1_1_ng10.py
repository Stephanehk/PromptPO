from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import MujocoReacherObservation


class ReacherV5Policy:
    def act(self, obs: MujocoReacherObservation):
        """
        Computes the action (torques) for the Reacher-v5 environment using a 
        Jacobian Transpose controller (Cartesian PD control).
        
        The goal is to drive the fingertip to the target position.
        """
        # Standard Reacher link lengths
        l1 = 0.1
        l2 = 0.11

        # Extract current joint state
        # qpos[0]: joint0 angle, qpos[1]: joint1 angle
        theta1 = obs.qpos[0]
        theta2 = obs.qpos[1]
        
        # qvel[0]: joint0 velocity, qvel[1]: joint1 velocity
        dtheta1 = obs.qvel[0]
        dtheta2 = obs.qvel[1]

        # Extract error vector in Cartesian space
        # obs.obs_vector[8:10] contains (fingertip - target) xy coordinates.
        # We want to move fingertip to target, so desired direction is -(fingertip - target).
        vec_tip_minus_target = obs.obs_vector[8:10]
        error_pos = -vec_tip_minus_target  # Vector from tip to target

        # Compute Jacobian Matrix J(q)
        # Kinematics:
        # x = l1*c1 + l2*c12
        # y = l1*s1 + l2*s12
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)

        # J = [ dx/dth1  dx/dth2 ]
        #     [ dy/dth1  dy/dth2 ]
        j11 = -l1 * s1 - l2 * s12
        j12 = -l2 * s12
        j21 = l1 * c1 + l2 * c12
        j22 = l2 * c12

        J = np.array([
            [j11, j12],
            [j21, j22]
        ])

        # Compute fingertip Cartesian velocity: v = J * dq
        dq = np.array([dtheta1, dtheta2])
        v_tip = J @ dq

        # PD Control Law in Cartesian Space
        # F = Kp * (pos_error) - Kd * (vel_error)
        # We want v_tip to be 0 at target, so vel_error = 0 - v_tip = -v_tip
        Kp = 15.0  # Proportional gain (stiffness)
        Kd = 1.0   # Derivative gain (damping)

        F_virtual = Kp * error_pos - Kd * v_tip

        # Map Cartesian force to Joint torques using Jacobian Transpose
        # tau = J^T * F
        tau = J.T @ F_virtual

        # Clip torques to valid action space [-1, 1]
        action = np.clip(tau, -1.0, 1.0)

        return action