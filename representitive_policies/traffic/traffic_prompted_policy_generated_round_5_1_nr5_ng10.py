from abc import ABCMeta
import numpy as np
from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation


class TrafficPolicy:
    def __init__(self):
        """
        Initializes the Intelligent Driver Model (IDM) policy.
        
        Optimized for MergePOEnv and TrafficRewardFunction:
        - T=1.05s: Just above the penalty threshold (1.0s) to maximize density/throughput 
          without incurring Cost 2.
        - s0=0.5m: Reduced jam distance to allow tighter packing in congestion, improving 
          bottleneck discharge rate.
        - Noise Suppression: Aggressively filters accelerations < 0.35 m/s^2 when safe 
          to minimize Cost 3 (mean accel penalty).
        """
        self.v0 = 30.0      # Default, updated in act()
        self.T = 1.05       # Headway buffer > 1.0s
        self.s0 = 0.5       # Tight jam distance
        self.a_max = 1.5    # Max accel
        self.b = 2.0        # Comfortable braking
        self.delta = 4.0    # Accel exponent

    def act(self, obs: TrafficObservation) -> np.ndarray:
        """
        Calculates actions for RL vehicles.
        """
        # Update target velocity from observation
        if obs.target_velocity > 1e-3:
            self.v0 = obs.target_velocity

        # Prepare arrays
        ego_speeds = obs.ego_speeds
        leader_headways = obs.leader_headways
        # delta_v = v_ego - v_leader
        delta_vs = -obs.leader_speed_diffs
        
        num_rl = len(ego_speeds)
        actions = np.zeros(num_rl, dtype=np.float32)

        for i in range(num_rl):
            v = ego_speeds[i]
            h = leader_headways[i]
            dv = delta_vs[i]

            # Filter ghost/inactive vehicles
            if h < 1e-2 and v < 1e-2:
                actions[i] = 0.0
                continue

            # --- IDM Logic ---
            
            # 1. Desired gap s*
            dynamic_term = (v * dv) / (2.0 * np.sqrt(self.a_max * self.b))
            s_star = self.s0 + v * self.T + dynamic_term
            s_star = max(0.0, s_star)

            # 2. Acceleration
            ratio = v / self.v0 if self.v0 > 1e-5 else 1.0
            free_road_term = 1.0 - ratio ** self.delta
            
            safe_h = max(h, 0.01)
            interaction_term = (s_star / safe_h) ** 2
            
            accel = self.a_max * (free_road_term - interaction_term)

            # --- Filtering / Optimization ---
            
            # Calculate safety metrics
            # Time Headway: h / v
            time_headway = h / (v + 1e-6)
            
            # Safety Condition:
            # We are safe if time_headway is comfortably above the 1.0s penalty zone.
            # Using 1.1s as a buffer.
            # Also consider low-speed "jam" safety where h is small but v is small.
            is_safe = (time_headway > 1.1) or (v < 2.0 and h > 2.0)
            
            # Stability Condition:
            # We are stable if we are not closing in rapidly on the leader.
            is_stable = dv < 0.5 

            # Coasting Logic (Minimize Cost 3):
            # If the requested acceleration is "noise" (small adjustments)
            # and the situation is safe and stable, force acceleration to 0.
            # This reduces the mean absolute acceleration significantly.
            
            noise_threshold = 0.35  # Aggressive filtering threshold
            
            if abs(accel) < noise_threshold:
                if is_safe and is_stable:
                    accel = 0.0
                # If we are just maintaining speed near target, also coast
                elif abs(v - self.v0) < 2.0 and accel > 0:
                    accel = 0.0
            
            # Always zero out negligible values
            if abs(accel) < 0.05:
                accel = 0.0

            actions[i] = accel

        # Clip to environment bounds
        actions = np.clip(actions, -1.5, 1.5)
        
        return actions