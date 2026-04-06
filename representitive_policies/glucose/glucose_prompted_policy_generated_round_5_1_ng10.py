from abc import ABCMeta
import numpy as np
from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation


class GlucosePolicy:
    def act(self, obs: GlucoseObservation):
        """
        Attempt 5 Policy:
        Refined aggressive controller with strict safety guards.
        
        Key Improvements over previous attempts:
        1. **Stratified Correction**: Distinguishes between "Very High" (>200) and "High" (>160) BG.
           Allows continued aggressive insulin application at very high levels even if dropping slowly,
           fixing the issue where correction stopped too early in previous attempts.
        2. **Optimized Braking**: Relaxed the "Approach Brake" logic to allow small corrections when BG 
           is drifting down slowly near target, reducing the average BG compared to Attempt 3.
        3. **Safety**: Retained strict low-BG cutoffs (<110) and panic brakes for rapid drops (<-3.0)
           to ensure survival and minimize exponential risk penalties.
        """
        
        # 0. Terminal check
        if obs.is_terminated:
            return np.array([0.0])
            
        # 1. Parse Data
        bg_hist = obs.bg
        # Filter valid values (padding is -1)
        valid_bg = bg_hist[bg_hist > 10.0]
        
        # Fallback if no valid data
        if len(valid_bg) == 0:
            return np.array([0.0])
            
        bg_curr = valid_bg[-1]
        
        # Calculate Trend (smooth over last 10 mins/3 steps if possible)
        # Unit: mg/dL per step (approx 5 mins)
        # Trend > 0 means rising, Trend < 0 means falling
        trend = 0.0
        if len(valid_bg) >= 3:
            trend = (valid_bg[-1] - valid_bg[-3]) / 2.0
        elif len(valid_bg) == 2:
            trend = valid_bg[-1] - valid_bg[-2]
            
        # Parse CHO (Meal)
        cho_hist = obs.cho
        valid_cho = cho_hist[cho_hist > -0.1]
        recent_cho_sum = 0.0
        # Look back 30 mins (6 steps) to detect active meal absorption
        if len(valid_cho) > 0:
            window = min(len(valid_cho), 6)
            recent_cho_sum = np.sum(valid_cho[-window:])
            
        # Constants
        TARGET_BG = 120.0
        MAX_ACTION = 0.1
        MIN_ACTION = 0.0
        
        # --- Policy Logic ---
        
        # 1. Critical Low Safety (Hypo Protection)
        # If BG is below 110, cut insulin immediately.
        # This provides a buffer before the danger zone (<70), minimizing Magni risk penalties.
        if bg_curr < 110.0:
            return np.array([MIN_ACTION])
            
        # 2. Crash Safety (Panic Brake)
        # If BG is falling rapidly (<-3.0 mg/dL per 5 min = -36 mg/dL/hr), stop insulin.
        # This overrides high BG logic to prevent undershoot/hypo.
        if trend < -3.0:
            return np.array([MIN_ACTION])
            
        # 3. Approach Safety (Soft Landing)
        # If BG is approaching target (<140) and falling consistently (<-1.0),
        # stop insulin to let it glide in.
        # Note: Relaxed from -0.5 in Attempt 3 to -1.0 to allow slight correction of drifts.
        if bg_curr < 140.0 and trend < -1.0:
            return np.array([MIN_ACTION])
            
        # 4. Meal Trigger
        # If carbs detected recently, saturation bolus is required to handle absorption.
        # Priority over PD, but respects Safety Brakes above.
        if recent_cho_sum > 0.5:
            return np.array([MAX_ACTION])
            
        # 5. Hyperglycemia Correction (Stratified)
        
        # Tier A: Very High (> 200). 
        # Blast with max insulin. Only the "Panic Brake" (Trend < -3) stops this.
        # We accept -1 or -2 trend here because BG is high enough that we need to accelerate the drop.
        if bg_curr > 200.0:
            return np.array([MAX_ACTION])
            
        # Tier B: High (> 160).
        # Blast with max insulin, but stop if dropping moderately (<-2.0).
        # Prevents stacking insulin if trajectory is already good.
        if bg_curr > 160.0:
            if trend < -2.0:
                return np.array([MIN_ACTION])
            return np.array([MAX_ACTION])
            
        # 6. Maintenance Control (PD)
        # Active in range [110, 160] (if not handled by brakes/meal).
        # Target 120.
        kp = 0.005
        kd = 0.02
        
        error = bg_curr - TARGET_BG
        p_term = kp * error
        d_term = kd * trend
        
        action = p_term + d_term
        
        # Clip to valid range
        action = np.clip(action, MIN_ACTION, MAX_ACTION)
        
        return np.array([action])