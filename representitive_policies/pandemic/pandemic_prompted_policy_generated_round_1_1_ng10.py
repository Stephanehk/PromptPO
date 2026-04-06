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
        Policy to maximize reward in the PandemicSimulator.
        
        Strategies:
        1. Critical Cases (Weight -10.0): Primary objective is to keep critical cases low.
           We set a low threshold (0.1% population) to trigger lockdowns to prevent exponential growth.
        2. Political Reward: We must strictly avoid raising the stage if the total infection rate
           (Infected + Critical + Dead) is less than 0.005, otherwise we incur a large squared penalty.
        3. Economy (Weight -0.1 max): We prefer lower stages. We lower the stage when infection
           metrics drop below safe lower bounds.
        4. Smoothness: We use hysteresis (gap between upper and lower thresholds) to avoid 
           oscillating between stages, which incurs a small penalty.
        
        Action Space (Discrete(3)):
        0: Decrease Stage (-1)
        1: Stay (0)
        2: Increase Stage (+1)
        """
        
        # --- 1. State Parsing ---
        
        # Identify indices for the infection summary vector
        # sorted_infection_summary contains [NONE, INFECTED, CRITICAL, RECOVERED, DEAD]
        idx_inf = sorted_infection_summary.index(InfectionSummary.INFECTED)
        idx_crit = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        idx_dead = sorted_infection_summary.index(InfectionSummary.DEAD)
        
        # Retrieve the most recent observation data
        # shape is (history_size, 1, num_summaries)
        latest_stats = obs.global_infection_summary[-1, 0, :]
        
        prop_infected = latest_stats[idx_inf]
        prop_critical = latest_stats[idx_crit]
        prop_dead = latest_stats[idx_dead]
        
        # Retrieve current regulation stage (0 to 4)
        current_stage = int(obs.stage[-1, 0, 0])
        
        # --- 2. Threshold Definitions ---
        
        # Political Threshold: from PoliticalStageReward
        # Raising stage when (inf + crit + dead) < 0.005 incurs heavy penalty
        POLITICAL_MIN_PREVALENCE = 0.005
        current_prevalence = prop_infected + prop_critical + prop_dead
        
        # Lockdown Triggers (Upper Bounds)
        # If critical cases exceed 0.1%, we are nearing the point where health penalty > max economic penalty.
        # We must act proactively.
        CRITICAL_UPPER_BOUND = 0.001 
        # High infection rate (0.5%) is a leading indicator for critical cases.
        INFECTED_UPPER_BOUND = 0.005 
        
        # Reopening Triggers (Lower Bounds)
        # Must be significantly lower than upper bounds to prevent oscillation (SmoothStageChanges)
        # and ensure the wave is suppressed.
        CRITICAL_LOWER_BOUND = 0.0005
        INFECTED_LOWER_BOUND = 0.001
        
        # --- 3. Decision Logic ---
        
        action = 1 # Default: Stay
        
        should_raise = False
        should_lower = False
        
        # Check condition to raise restrictions
        if current_stage < 4:
            if prop_critical > CRITICAL_UPPER_BOUND:
                should_raise = True
            elif prop_infected > INFECTED_UPPER_BOUND and prop_critical > 0.0001:
                # Infection is high and critical cases are non-zero; growing wave.
                should_raise = True
                
        # Check condition to lower restrictions
        # Only lower if we aren't planning to raise and stats are very safe
        if current_stage > 0 and not should_raise:
            if prop_critical < CRITICAL_LOWER_BOUND and prop_infected < INFECTED_LOWER_BOUND:
                should_lower = True
        
        # --- 4. Apply Constraints and Select Action ---
        
        if should_raise:
            # STRICT CONSTRAINT: Check Political Reward condition
            # If we raise when prevalence is low, the penalty dominates the reward function.
            # We add a tiny epsilon to 0.005 to ensure we are safely above the penalty threshold.
            if current_prevalence >= (POLITICAL_MIN_PREVALENCE + 1e-6):
                action = 2 # Raise
            else:
                # We need to raise for safety, but politics prevents it.
                # We must wait for the infection to cross the 0.005 threshold.
                action = 1 # Stay
        elif should_lower:
            action = 0 # Lower
        else:
            action = 1 # Stay
            
        return action