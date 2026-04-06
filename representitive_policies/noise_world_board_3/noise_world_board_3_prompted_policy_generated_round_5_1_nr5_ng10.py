from abc import ABCMeta
import numpy as np
import heapq
from direct_policy_learning.observations import NoiseWorldObservation

class NoiseWorldBoard3Policy:
    def __init__(self):
        self.n = 10
        # Action semantic mappings
        # 0: Up (decrease r), 1: Down (increase r), 2: Left (decrease c), 3: Right (increase c)
        self.dirs = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
        
    def act(self, obs: NoiseWorldObservation) -> int:
        n = self.n
        # Extract and unnormalize grid positions (0 to 9)
        agent_r = int(round(obs.agent_row * (n - 1)))
        agent_c = int(round(obs.agent_col * (n - 1)))
        goal_r = int(round(obs.goal_row * (n - 1)))
        goal_c = int(round(obs.goal_col * (n - 1)))
        
        # If the agent is already somehow precisely on the goal cell, output a safe fallback action
        if agent_r == goal_r and agent_c == goal_c:
            return 0
            
        # Reshape flat board to 2D
        board = np.array(obs.board, dtype=int).reshape((n, n))
        
        # Setup Dijkstra's algorithm to plan the expected shortest path
        pq = [(0.0, agent_r, agent_c)]
        dists = np.full((n, n), np.inf)
        dists[agent_r, agent_c] = 0.0
        
        parents = {}
        
        # Determine movement cost through a particular cell.
        # Instead of treating all non-zero cells as infinite-cost traps, we assign expected step
        # delays (assuming higher numbers correlate with higher slip probabilities/noise).
        # We cap 4 and 5 at high costs to avoid catastrophic failure traps, but allow shortcuts 
        # through 1, 2, and 3 if it saves substantial baseline steps.
        def get_cost(r, c):
            if r == goal_r and c == goal_c:
                return 1.0
            val = board[r, c]
            if val == 0: return 1.0
            if val == 1: return 1.2
            if val == 2: return 1.6
            if val == 3: return 2.5
            if val == 4: return 1000.0
            if val == 5: return 1000.0
            return 1000.0

        while pq:
            d, r, c = heapq.heappop(pq)
            
            if d > dists[r, c]: 
                continue
                
            if r == goal_r and c == goal_c: 
                break
                
            for dr, dc, act in self.dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    cost = get_cost(nr, nc)
                    new_dist = d + cost
                    
                    if new_dist < dists[nr, nc]:
                        dists[nr, nc] = new_dist
                        parents[(nr, nc)] = (r, c, act)
                        heapq.heappush(pq, (new_dist, nr, nc))
                        
        # Backtrack from the goal to find the optimal immediate next action for closed-loop control
        if (goal_r, goal_c) in parents:
            curr = (goal_r, goal_c)
            path = []
            while curr != (agent_r, agent_c):
                pr, pc, act = parents[curr]
                path.append(act)
                curr = (pr, pc)
            return path[-1]
            
        # Robust fallback mechanism in the event of an unreachable goal/disconnected graph
        if agent_r > goal_r: return 0
        if agent_r < goal_r: return 1
        if agent_c > goal_c: return 2
        if agent_c < goal_c: return 3
        return 0