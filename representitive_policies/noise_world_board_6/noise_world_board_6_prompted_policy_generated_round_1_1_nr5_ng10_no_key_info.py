from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation
import heapq

class NoiseWorldBoard6Policy:
    def act(self, obs):
        # Access the observation object attributes correctly
        agent_row = int(round(obs.agent_row * 9))
        agent_col = int(round(obs.agent_col * 9))
        goal_row = int(round(obs.goal_row * 9))
        goal_col = int(round(obs.goal_col * 9))
        
        # Ensure coordinates are within grid bounds
        agent_row = max(0, min(9, agent_row))
        agent_col = max(0, min(9, agent_col))
        goal_row = max(0, min(9, goal_row))
        goal_col = max(0, min(9, goal_col))
        
        # Extract the board layout (10x10 flattened)
        board = obs.board
        
        # Count frequencies of each tile value to identify anomalies (like walls or lava traps)
        counts = {}
        for i in range(100):
            v = int(board[i])
            counts[v] = counts.get(v, 0) + 1
            
        # Prioritize cell types based on how common they are in the environment
        board_vals = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        most_common = board_vals[0] if board_vals else 0
        agent_val = int(board[agent_row * 10 + agent_col])
        goal_val = int(board[goal_row * 10 + goal_col])
        
        # Safe tiles we know we can traverse or that represent standard empty space
        safe_v = {agent_val, goal_val, most_common}
        
        # Generate traversal costs. Pathing minimizes the use of rare, potentially dangerous tiles.
        cost_map = {}
        for v in counts:
            if v in safe_v:
                cost_map[v] = 1
            else:
                idx = board_vals.index(v)
                # Steep exponential cost for rarer tile types to strictly minimize max danger
                cost_map[v] = 1000 ** idx
                
        # Dijkstra's Algorithm to find the safest shortest path
        pq = [(0, 0, agent_row, agent_col)]
        best_cost = {(agent_row, agent_col): 0}
        parent = {}
        
        while pq:
            cost, steps, r, c = heapq.heappop(pq)
            
            if (r, c) == (goal_row, goal_col):
                break
                
            # Skip if we already found a strictly better path to this node
            if cost > best_cost.get((r, c), float('inf')):
                continue
                
            # Iterate through available actions mapped by semantics
            for dr, dc, action in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    v = int(board[nr * 10 + nc])
                    step_cost = cost_map.get(v, 1000000)
                    ncost = cost + step_cost
                    
                    if ncost < best_cost.get((nr, nc), float('inf')):
                        best_cost[(nr, nc)] = ncost
                        heapq.heappush(pq, (ncost, steps + 1, nr, nc))
                        parent[(nr, nc)] = (r, c, action)
                        
        # Backtrack to reconstruct the first action on the optimal path
        if (goal_row, goal_col) in parent:
            curr = (goal_row, goal_col)
            path = []
            while curr != (agent_row, agent_col):
                pr, pc, action = parent[curr]
                path.append(action)
                curr = (pr, pc)
            if path:
                return path[-1]
                
        # Fallback heuristic
        if agent_row > goal_row: return 0
        if agent_row < goal_row: return 1
        if agent_col > goal_col: return 2
        if agent_col < goal_col: return 3
        
        return 0