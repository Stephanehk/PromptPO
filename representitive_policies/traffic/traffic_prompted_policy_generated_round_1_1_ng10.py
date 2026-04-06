import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation

class TrafficPolicy:
    """
    A policy that controls RL vehicles using the Intelligent Driver Model (IDM) logic.
    Ensures the output action always matches the environment's action space shape (10,),
    padding with zeros if fewer vehicles are present.
    """
    def act(self, obs: TrafficObservation) -> np.ndarray:
        # The environment wrapper expects a fixed action shape of (10,)
        # regardless of the number of currently active RL vehicles.
        action_space_size = 10
        full_action = np.zeros(action_space_size, dtype=np.float32)

        # 1. Identify number of active RL vehicles
        num_rl = len(obs.rl_ids)
        if num_rl == 0:
            return full_action

        # 2. Extract state for active vehicles
        v_ego = obs.ego_speeds
        h_leader = obs.leader_headways
        
        # IDM uses delta_v = v_ego - v_leader
        # Observation provides leader_speed_diff = v_leader - v_ego
        dv = -obs.leader_speed_diffs

        # 3. IDM Parameters
        # Use observed target velocity or default to 30.0 if 0/unset
        target_v = obs.target_velocity if obs.target_velocity > 0.1 else 30.0
        
        a_max = 1.0     # Maximum acceleration parameter
        b_des = 2.0     # Comfortable deceleration parameter
        T_gap = 1.5     # Safe time headway (seconds)
        s_0 = 4.0       # Minimum stopping distance (meters)
        delta = 4.0     # Acceleration exponent

        # 4. Compute IDM Action (Vectorized)
        
        # Term 1: Free road acceleration -> a_max * [1 - (v/v0)^delta]
        # Prevents exceeding target velocity
        free_road_term = 1.0 - np.power(v_ego / target_v, delta)
        
        # Term 2: Interaction/Braking -> - a_max * (s*/h)^2
        # Desired gap s* = s0 + vT + (v * dv) / (2 * sqrt(ab))
        ab_term = 2.0 * np.sqrt(a_max * b_des)
        dynamic_gap = (v_ego * dv) / ab_term
        s_star = s_0 + (v_ego * T_gap) + dynamic_gap
        
        # Important: If leader is much faster (dv << 0), s_star can become negative.
        # We clamp s_star to 0 to prevent squaring a negative value, which would 
        # incorrectly cause braking (phantom obstacle).
        s_star = np.maximum(s_star, 0.0)
        
        # Interaction ratio (s* / h)^2
        # Clip headway to avoid division by zero
        h_safe = np.maximum(h_leader, 0.1)
        interaction_term = np.square(s_star / h_safe)
        
        # Total IDM acceleration
        raw_accel = a_max * (free_road_term - interaction_term)

        # 5. Clip to environment action bounds [-1.5, 1.5]
        clipped_accel = np.clip(raw_accel, -1.5, 1.5)

        # 6. Map to fixed-size action vector
        # Fill the slots corresponding to the active vehicles. 
        # The remaining slots stay 0.0 (ignored by env for non-existent IDs).
        count = min(num_rl, action_space_size)
        full_action[:count] = clipped_accel[:count]

        return full_action