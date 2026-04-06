from abc import ABCMeta
import numpy as np
import heapq
from direct_policy_learning.observations import NoiseWorldObservation

class NoiseWorldBoard5Policy:
    def __init__(self):
        self.last_pos = None
        self.last_action = None
        self.cell_penalties = {}

    def act(self, obs):
        # Unwrap gym tuple if needed (e.g., if observation is wrapped as (obs, info))
        if isinstance(obs, tuple) and len(obs) == 2:
            if hasattr(obs[0], 'agent_row') or isinstance(obs[1], dict):
                obs = obs[0]

        # Extract observation fields based on the layout description
        # NoiseWorldObservation exposes positions and the board sequence directly
        if hasattr(obs, 'agent_row') and hasattr(obs, 'board'):
            agent_row = int(round(obs.agent_row * 9))
            agent_col = int(round(obs.agent_col * 9))
            goal_row = int(round(obs.goal_row * 9))
            goal_col = int(round(obs.goal_col * 9))
            board = np.array(obs.board).reshape((10, 10))
        elif isinstance(obs, dict) and 'agent_row' in obs:
            agent_row = int(round(obs['agent_row'] * 9))
            agent_col = int(round(obs['agent_col'] * 9))
            goal_row = int(round(obs['goal_row'] * 9))
            goal_col = int(round(obs['goal_col'] * 9))
            board = np.array(obs['board']).reshape((10, 10))
        else:
            # Fallback to pure array parsing if it's strictly a raw vector
            try:
                obs_arr = np.array(obs, dtype=float).flatten()
            except Exception:
                obs_arr = None
                # Last ditch effort to find any internally stored raw array
                for attr in dir(obs):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(obs, attr)
                            arr = np.array(val, dtype=float)
                            if arr.ndim > 0 and arr.size >= 106:
                                obs_arr = arr.flatten()
                                break
                        except Exception:
                            pass
                if obs_arr is None:
                    return 0

            agent_row = int(round(obs_arr[0] * 9))
            agent_col = int(round(obs_arr[1] * 9))
            goal_row = int(round(obs_arr[2] * 9))
            goal_col = int(round(obs_arr[3] * 9))
            board = obs_arr[4:104].reshape((10, 10))

        current_pos = (agent_row, agent_col)

        # If the agent tried to move but failed (e.g., stochastic slip or impassable unseen wall),
        # incrementally penalize that target cell so the path planner avoids recurring traps
        if self.last_pos == current_pos and self.last_action is not None:
            r, c = current_pos
            target = None
            if self.last_action == 0: target = (r - 1, c)
            elif self.last_action == 1: target = (r + 1, c)
            elif self.last_action == 2: target = (r, c - 1)
            elif self.last_action == 3: target = (r, c + 1)
            
            if target is not None:
                # Modest penalty: allows some retries for stochastic noise, but forces rerouting if it consistently fails
                self.cell_penalties[target] = self.cell_penalties.get(target, 0) + 50
                
        self.last_pos = current_pos
        
        start_val = board[agent_row, agent_col]
        goal_val = board[goal_row, goal_col]
        
        start = current_pos
        goal = (goal_row, goal_col)
        
        if start == goal:
            self.last_action = 0
            return 0
            
        # Use Dijkstra's algorithm to compute the shortest safe path 
        pq = [(0, start)]
        visited = set()
        came_from = {}
        cost_so_far = {start: 0}
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            if current == goal:
                break
                
            if current in visited:
                continue
            visited.add(current)
            
            r, c = current
            # Action space mappings
            neighbors = [
                (0, r - 1, c),
                (1, r + 1, c),
                (2, r, c - 1),
                (3, r, c + 1)
            ]
            
            for action, nr, nc in neighbors:
                if 0 <= nr < 10 and 0 <= nc < 10:
                    v = board[nr, nc]
                    
                    # Compute dynamic costs based on cell terrain values
                    if v == start_val or v == goal_val or v == 0:
                        base_cost = 1
                    else:
                        base_cost = 1 + v * 100
                        
                    step_cost = base_cost + self.cell_penalties.get((nr, nc), 0)
                    new_cost = current_cost + step_cost
                    next_node = (nr, nc)
                    
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        came_from[next_node] = (current, action)
                        heapq.heappush(pq, (new_cost, next_node))
                        
        # Backtrack to find the first action in the shortest path
        if goal in came_from:
            curr = goal
            path_actions = []
            while curr != start and curr in came_from:
                prev, action = came_from[curr]
                path_actions.append(action)
                curr = prev
            chosen_action = path_actions[-1]
        else:
            # Fallback direct heuristic if mathematically disconnected
            dr = goal_row - agent_row
            dc = goal_col - agent_col
            if abs(dr) > abs(dc):
                chosen_action = 1 if dr > 0 else 0
            else:
                chosen_action = 3 if dc > 0 else 2
                
        self.last_action = chosen_action
        return chosen_action