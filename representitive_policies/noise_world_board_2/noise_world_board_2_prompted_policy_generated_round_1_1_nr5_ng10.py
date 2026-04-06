from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation
import heapq

class NoiseWorldBoard2Policy:
    def act(self, obs: NoiseWorldObservation):
        n = 10
        # Convert normalized coordinates back to absolute grid indices
        agent_r = int(round(obs.agent_row * (n - 1)))
        agent_c = int(round(obs.agent_col * (n - 1)))
        goal_r = int(round(obs.goal_row * (n - 1)))
        goal_c = int(round(obs.goal_col * (n - 1)))
        
        # Reshape the flat board into a 2D (n x n) matrix
        board = np.array(obs.board).reshape((n, n))
        
        # We assume 0 is free space, and >0 are obstacles, hazards or noise.
        # We assign a very high cost to >0 so the agent avoids them if possible,
        # but allows traversing them if there is no other way.
        def get_cost(v):
            if v == 0:
                return 1
            return 10000 + int(v)
            
        # Dijkstra's algorithm to find the optimal path to the goal
        pq = [(0, agent_r, agent_c)]
        dist = {(agent_r, agent_c): 0}
        parent = {}
        
        while pq:
            d, r, c = heapq.heappop(pq)
            
            if d > dist.get((r, c), float('inf')):
                continue
                
            if r == goal_r and c == goal_c:
                break
                
            # Available moves matching the specific semantic mapping 
            # 0: UP (row-1), 1: DOWN (row+1), 2: LEFT (col-1), 3: RIGHT (col+1)
            for nr, nc, action in [(r-1, c, 0), (r+1, c, 1), (r, c-1, 2), (r, c+1, 3)]:
                if 0 <= nr < n and 0 <= nc < n:
                    cost = get_cost(board[nr, nc])
                    new_d = d + cost
                    
                    if new_d < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_d
                        parent[(nr, nc)] = (r, c, action)
                        heapq.heappush(pq, (new_d, nr, nc))
                        
        # If the goal is unreachable (or we are already on the goal), fallback to simple greedy heuristic
        if (goal_r, goal_c) not in parent:
            if goal_r < agent_r: return 0
            if goal_r > agent_r: return 1
            if goal_c < agent_c: return 2
            if goal_c > agent_c: return 3
            return 0
            
        # Backtrack from goal to find the first action to take from the current cell
        curr = (goal_r, goal_c)
        path = []
        while curr != (agent_r, agent_c):
            r, c, action = parent[curr]
            path.append(action)
            curr = (r, c)
            
        return path[-1]