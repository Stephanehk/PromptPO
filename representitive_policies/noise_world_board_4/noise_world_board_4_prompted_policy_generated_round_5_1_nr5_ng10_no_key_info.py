from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation
import heapq
import itertools


class NoiseWorldBoard4Policy:
    def __init__(self):
        self.counter = itertools.count()
        self.blocked_counts = {}
        self.last_pos = None
        self.last_action = None
        self.last_board = None

    def act(self, obs):
        # Safely parse the observation
        try:
            agent_r = int(round(obs.agent_row * 9))
            agent_c = int(round(obs.agent_col * 9))
            goal_r = int(round(obs.goal_row * 9))
            goal_c = int(round(obs.goal_col * 9))
            board = np.round(np.array(obs.board)).astype(int).reshape((10, 10))
        except AttributeError:
            arr = np.asarray(obs).flatten()
            agent_r = int(round(arr[0] * 9))
            agent_c = int(round(arr[1] * 9))
            goal_r = int(round(arr[2] * 9))
            goal_c = int(round(arr[3] * 9))
            board = np.round(arr[4:104]).astype(int).reshape((10, 10))
            
        current_pos = (agent_r, agent_c)
        target_r, target_c = goal_r, goal_c

        # Detect new episodes or board resets
        if self.last_board is not None and not np.array_equal(board, self.last_board):
            self.blocked_counts.clear()
            self.last_action = None
            self.last_pos = None

        # Check for discontinuities (teleportation) or blocked movements
        if self.last_pos is not None:
            dist_r = abs(current_pos[0] - self.last_pos[0])
            dist_c = abs(current_pos[1] - self.last_pos[1])
            if dist_r > 1 or dist_c > 1:
                # Teleported! Must be a new episode
                self.blocked_counts.clear()
                self.last_action = None
            elif self.last_pos == current_pos and self.last_action is not None:
                # Agent did not move; bumped into an impassable wall or stochastic noise
                dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[self.last_action]
                blocked_r, blocked_c = current_pos[0] + dr, current_pos[1] + dc
                self.blocked_counts[(blocked_r, blocked_c)] = self.blocked_counts.get((blocked_r, blocked_c), 0) + 1

        self.last_board = board.copy()
        self.last_pos = current_pos
        
        # Determine the frequencies of cell types
        values, counts = np.unique(board, return_counts=True)
        counts_dict = dict(zip(values, counts))
        floor_val = values[np.argmax(counts)]  # The most common cell is the safe floor
        
        # Determine pathfinding costs based on rarity
        def get_cell_cost(r, c):
            # Target cell and agent's safe spot
            if r == target_r and c == target_c:
                return 1
            if r == agent_r and c == agent_c:
                return 1
            
            val = board[r, c]
            if val == floor_val:
                return 1
                
            # If the cell is blocked repeatedly, heavily penalize it
            b_count = self.blocked_counts.get((r, c), 0)
            blocked_penalty = b_count * 1000
            
            # Key insight: the goal is surrounded by walls (high frequency) but likely accessible 
            # via a door/passage (rare frequency). 
            # Crucially, we do NOT set all cells matching the goal's value to 1. Only the specific 
            # (target_r, target_c) coordinate gets a cost of 1, preserving the high penalty for walls 
            # even if the goal itself shares the wall's underlying grid value.
            return 10 + counts_dict.get(val, 0) * 10 + blocked_penalty
                
        # Dijkstra's Algorithm implementation
        pq = []
        # Tuple format: (cumulative_cost, order_counter, r, c, first_action_in_path)
        heapq.heappush(pq, (0, next(self.counter), agent_r, agent_c, None))
        
        dist_to = {(agent_r, agent_c): 0}
        best_action = None
        
        while pq:
            cost, _, r, c, first_action = heapq.heappop(pq)
            
            if (r, c) == (target_r, target_c):
                best_action = first_action
                break
                
            if cost > dist_to.get((r, c), float('inf')):
                continue
            
            # Semantic mapping: 0: Up, 1: Down, 2: Left, 3: Right
            for dr, dc, a in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    step_cost = get_cell_cost(nr, nc)
                    new_cost = cost + step_cost
                    
                    if new_cost < dist_to.get((nr, nc), float('inf')):
                        dist_to[(nr, nc)] = new_cost
                        fa = a if first_action is None else first_action
                        heapq.heappush(pq, (new_cost, next(self.counter), nr, nc, fa))
                        
        # Fallback heuristic if somehow completely boxed in or already exactly at the target
        if best_action is None:
            dr_diff = target_r - agent_r
            dc_diff = target_c - agent_c
            if abs(dr_diff) > abs(dc_diff):
                best_action = 1 if dr_diff > 0 else 0
            elif abs(dc_diff) > 0:
                best_action = 3 if dc_diff > 0 else 2
            else:
                best_action = 0  # Default valid move
                
        self.last_action = best_action
        return best_action