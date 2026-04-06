from abc import ABCMeta
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation


class GlucosePolicy:
    def act(self, obs: GlucoseObservation):
        """
        Calculates the insulin action based on the current observation.
        Uses a Proportional-Derivative (PD) controller strategy to maintain Blood Glucose (BG)
        around a safe target, minimizing risk of hypoglycemia (death) and insulin penalties.
        """
        # If the patient is deceased, return 0 to satisfy interface but simulation is over.
        if obs.is_terminated:
            return np.array([0.0])

        # Extract Blood Glucose history
        bg_hist = obs.bg
        
        # Handle cases where history might be empty or unavailable
        if bg_hist is None or len(bg_hist) == 0:
            return np.array([0.0])
            
        # Get the most recent glucose reading
        # The environment pads history with -1 if it's not full yet.
        bg_current = bg_hist[-1]
        
        # If the last reading is a padding value (-1), search backwards for the last valid one
        if bg_current == -1:
            valid_indices = np.where(bg_hist > -1)[0]
            if len(valid_indices) > 0:
                bg_current = bg_hist[valid_indices[-1]]
            else:
                # Fallback if no valid data is found (e.g., start of sim)
                return np.array([0.0])
        
        # Calculate Trend (rate of change in mg/dL per step)
        trend = 0.0
        if len(bg_hist) > 1:
            bg_prev = bg_hist[-2]
            # Ensure previous value is valid before calculating trend
            if bg_prev != -1:
                trend = bg_current - bg_prev

        # --- Controller Logic ---
        
        # Target BG: 140 mg/dL
        # Rationale:
        # 1. The Magni Risk function used in the reward minimizes around 135 mg/dL.
        # 2. Safety buffer: A target of 140 keeps the patient away from the fatal hypoglycemia zone (<50-70).
        # 3. Cost efficiency: Insulin is expensive in the reward function (-10 * dose), so we avoid over-treating.
        target_bg = 140.0
        
        # Proportional Gain (Kp)
        # Determines reaction to the magnitude of hyperglycemia.
        # Example: BG 200 (Error 60) * 0.0015 = 0.09 (Near max action 0.1)
        kp = 0.0015
        
        # Derivative Gain (Kd)
        # Determines reaction to the speed of BG change.
        # Helps curb rising BG early and prevent overshoot on falling BG.
        kd = 0.005

        action = 0.0
        
        # Only apply control if BG is above target to act as a basal/correction bolus
        if bg_current > target_bg:
            # PD Control Law
            p_term = kp * (bg_current - target_bg)
            d_term = kd * trend
            action = p_term + d_term
        
        # --- Safety Overrides ---
        
        # Hypoglycemia Guard:
        # If BG is close to the danger zone, cut insulin completely to let BG drift up.
        if bg_current < 120:
            action = 0.0
            
        # rapid Drop Guard:
        # If BG is falling fast, cut insulin to prevent undershoot.
        elif trend < -2.0:
            action = 0.0

        # --- Action Space Constraints ---
        # Action space is Box(0.0, 0.1)
        action = np.clip(action, 0.0, 0.1)
        
        return np.array([action])