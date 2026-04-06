from abc import ABCMeta
import numpy as np
from pandemic.python.pandemic_simulator.environment.interfaces import (
    PandemicObservation,
    InfectionSummary,
    sorted_infection_summary
)


class PandemicPolicy:
    def act(self, obs: PandemicObservation) -> int:
        """
        Policy maximizing PandemicRewardFunction.
        
        Key objectives:
        1. Avoid PoliticalStageReward penalty (Raise only if Rate > 0.005 or emergency).
        2. Minimize CriticalCasesAbsoluteReward (Suppress waves).
        3. Minimize LowerStagePreferenceReward (Use lowest effective stage).
        4. Minimize SmoothStageChangesPreferenceReward (Hysteresis).
        """
        
        # Action Constants (Discrete(3))
        ACTION_LOWER = 0
        ACTION_STAY = 1
        ACTION_RAISE = 2
        
        # --- Extract State ---
        current_stage = int(obs.stage[-1, 0, 0])
        
        idx_inf = sorted_infection_summary.index(InfectionSummary.INFECTED)
        idx_crit = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        idx_dead = sorted_infection_summary.index(InfectionSummary.DEAD)
        
        curr_vec = obs.global_infection_summary[-1, 0]
        curr_inf = curr_vec[idx_inf]
        curr_crit = curr_vec[idx_crit]
        curr_dead = curr_vec[idx_dead]
        
        # Risk Metric for Political Constraint (Infected + Critical + Dead)
        risk_metric = curr_inf + curr_crit + curr_dead
        
        # Growth Trend Analysis
        is_growing = False
        if obs.global_infection_summary.shape[0] > 1:
            prev_inf = obs.global_infection_summary[-2, 0, idx_inf]
            prev_crit = obs.global_infection_summary[-2, 0, idx_crit]
            # Check if infection is expanding (using a small margin for noise)
            if (curr_inf > prev_inf * 1.001) or (curr_crit > prev_crit):
                is_growing = True
        
        # --- Determine Target Stage ---
        # Default to current stage
        target_stage = current_stage
        
        if is_growing:
            # Escalation Logic: Suppress rising waves
            
            if current_stage == 0:
                # Early detection: If infection is widespread or criticals starting
                if curr_crit > 0.0005:
                    target_stage = 2 # Criticals rising: Needs strong suppression (Masks/Distancing)
                elif curr_inf > 0.003:
                    target_stage = 1 # High prevalence: Pre-emptive mild restrictions
            
            elif current_stage == 1:
                # If mild restrictions fail
                if curr_crit > 0.0008:
                    target_stage = 2
            
            elif current_stage == 2:
                # If strong restrictions fail (Emergency)
                if curr_crit > 0.0025:
                    target_stage = 3
            
            elif current_stage == 3:
                # Last resort
                if curr_crit > 0.0050:
                    target_stage = 4

        else:
            # De-escalation Logic: Hysteresis to prevent rapid switching
            # We require critical cases to be very low to relax, ensuring the wave is crushed.
            
            if current_stage == 4:
                if curr_crit < 0.0030:
                    target_stage = 3
            
            elif current_stage == 3:
                if curr_crit < 0.0015:
                    target_stage = 2
            
            elif current_stage == 2:
                if curr_crit < 0.0006:
                    target_stage = 1
            
            elif current_stage == 1:
                if curr_crit < 0.0002: # Wait until almost clear
                    target_stage = 0

        # --- Resolve Action with Political Constraints ---
        
        if target_stage == current_stage:
            return ACTION_STAY
        
        elif target_stage < current_stage:
            # Lowering is always allowed and encouraged by rewards
            return ACTION_LOWER
            
        else: # target > current (Raise)
            # We want to raise. Check if we can afford the Political Penalty.
            # Penalty = -10 * ((risk - 0.005)/0.005)^2 if risk < 0.005
            
            if risk_metric >= 0.005:
                # Safe zone: No penalty
                return ACTION_RAISE
            else:
                # Danger zone: Raising incurs penalty.
                # Calculate potential penalty
                diff = risk_metric - 0.005
                penalty = -10.0 * (diff / 0.005) ** 2
                
                # Decision: Is the emergency severe enough to eat the penalty?
                # We define a 'Soft Threshold' around 0.0044 where penalty is small (< -0.15).
                # We only override if Critical cases are actively threatening (> 0.0005).
                
                is_emergency = (curr_crit > 0.0005)
                acceptable_penalty = (penalty > -0.15)
                
                if is_emergency and acceptable_penalty:
                    return ACTION_RAISE
                else:
                    # Forced to wait by politics
                    return ACTION_STAY