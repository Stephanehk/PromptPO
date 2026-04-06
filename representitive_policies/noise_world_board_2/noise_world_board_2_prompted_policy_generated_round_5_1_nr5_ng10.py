from abc import ABCMeta
import numpy as np
import heapq
from direct_policy_learning.observations import NoiseWorldObservation

class NoiseWorldBoard2Policy:
    def act(self, obs: NoiseWorldObservation):
        n = 10
        # Convert normalized coordinates back to absolute grid indices
        agent_r = int(round(obs.agent_row * (n - 1)))
        agent_c = int(round(obs.agent_col * (n - 1)))
        goal_r = int(round(obs.goal_row * (n - 1)))
        goal_c = int(round(obs.goal_col * (n - 1)))
        
        # Terminate early if already at the goal (fallback safety)
        if agent_r == goal_r and agent_c == goal_c:
            return 0
            
        # Reshape the flat board into a 2D (n x n) matrix
        board = np.array(obs.board).reshape((n, n))
        
        # The true optimal policy assigns edge costs based on the expected number of 
        # attempts to LEAVE the current cell, representing the inverse of success probability.
        # This properly evaluates the penalty of stochastic failures without falsely applying 
        # transition penalties to reaching the ultimate goal cell.
        def expected_steps_to_leave(r, c):
            v = int(board[r, c])
            if v == 0: return 1.0
            if v == 1: return 5.0
            if v == 2: return 20.0
            if v == 3: return 100.0
            if v == 4: return 500.0
            return 2000.0
            
        # Dijkstra's algorithm to find the mathematically optimal expected-cost path
        pq = [(0.0, agent_r, agent_c)]
        dist = {(agent_r, agent_c): 0.0}
        parent = {}
        
        while pq:
            d, r, c = heapq.heappop(pq)
            
            # Skip if we previously found a shorter path to this cell
            if d > dist.get((r, c), float('inf')):
                continue
                
            # Stop searching once the goal is definitively expanded
            if r == goal_r and c == goal_c:
                break
                
            # Base cost: expected steps to depart the CURRENT cell
            step_cost = expected_steps_to_leave(r, c)
            
            # Semantic mapping for available actions:
            # 0: UP (row-1), 1: DOWN (row+1), 2: LEFT (col-1), 3: RIGHT (col+1)
            for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    # Infinitesimal sub-integer tie-breaker: slightly repels paths from hazards.
                    # This ensures we strictly pick the safest path among those with the exact same expected length.
                    risk = 0.0
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nnr, nnc = nr + ddr, nc + ddc
                        if 0 <= nnr < n and 0 <= nnc < n:
                            val = int(board[nnr, nnc])
                            if val > 0:
                                risk += 1e-4 * val
                                
                    new_d = d + step_cost + risk
                    
                    if new_d < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_d
                        parent[(nr, nc)] = (r, c, action)
                        heapq.heappush(pq, (new_d, nr, nc))
                        
        # Fallback heuristic if the goal is completely walled off
        if (goal_r, goal_c) not in parent:
            if goal_r < agent_r: return 0
            if goal_r > agent_r: return 1
            if goal_c < agent_c: return 2
            if goal_c > agent_c: return 3
            return 0
            
        # Backtrack from the optimal goal path to find the immediate next action to take
        curr = (goal_r, goal_c)
        path = []
        while curr != (agent_r, agent_c):
            r, c, action = parent[curr]
            path.append(action)
            curr = (r, c)
            
        return path[-1]