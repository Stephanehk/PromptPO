from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation
import heapq
import itertools


class NoiseWorldBoard4Policy:
    def __init__(self):
        self.safe_vals = set()
        self.counter = itertools.count()
        self.blocked_counts = {}
        self.last_pos = None
        self.last_action = None
        self.last_board = None

    def act(self, obs):
        # Safely parse the observation into a numpy array
        try:
            arr = np.asarray(obs).flatten()
            if len(arr) == 106:
                agent_r = int(round(arr[0] * 9))
                agent_c = int(round(arr[1] * 9))
                goal_r = int(round(arr[2] * 9))
                goal_c = int(round(arr[3] * 9))
                board = np.round(arr[4:104]).astype(int).reshape((10, 10))
            else:
                agent_r = int(round(obs.agent_row * 9))
                agent_c = int(round(obs.agent_col * 9))
                goal_r = int(round(obs.goal_row * 9))
                goal_c = int(round(obs.goal_col * 9))
                board = np.round(np.array(obs.board)).astype(int).reshape((10, 10))
        except Exception:
            agent_r = int(round(obs.agent_row * 9))
            agent_c = int(round(obs.agent_col * 9))
            goal_r = int(round(obs.goal_row * 9))
            goal_c = int(round(obs.goal_col * 9))
            board = np.round(np.array(obs.board)).astype(int).reshape((10, 10))
            
        current_pos = (agent_r, agent_c)

        # Detect new episodes or board resets
        if self.last_board is not None and not np.array_equal(board, self.last_board):
            self.safe_vals.clear()
            self.blocked_counts.clear()
            self.last_action = None
            self.last_pos = None

        # Check for discontinuities (teleportation) or blocked movements
        if self.last_pos is not None:
            dist_r = abs(current_pos[0] - self.last_pos[0])
            dist_c = abs(current_pos[1] - self.last_pos[1])
            if dist_r > 1 or dist_c > 1:
                # Teleported! Must be a new episode
                self.safe_vals.clear()
                self.blocked_counts.clear()
                self.last_action = None
            elif self.last_pos == current_pos and self.last_action is not None:
                # Agent did not move; potentially blocked by a wall or stochastic noise
                dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[self.last_action]
                blocked_r, blocked_c = current_pos[0] + dr, current_pos[1] + dc
                self.blocked_counts[(blocked_r, blocked_c)] = self.blocked_counts.get((blocked_r, blocked_c), 0) + 1

        self.last_board = board.copy()
        self.last_pos = current_pos
        
        # Add the current and goal values to the safe list
        self.safe_vals.add(board[agent_r, agent_c])
        self.safe_vals.add(board[goal_r, goal_c])
        
        # Determine frequency rank for all values to distinguish likely floors from hazards
        values, counts = np.unique(board, return_counts=True)
        sorted_vals = values[np.argsort(-counts)]
        
        val_costs = {}
        for v in range(8):
            if v in self.safe_vals:
                val_costs[v] = 1
            else:
                if v in sorted_vals:
                    rank = np.where(sorted_vals == v)[0][0]
                else:
                    rank = 8
                # Penalize unknown cells, prioritizing more frequent values (like open floors) over rare hazards
                val_costs[v] = 100 + rank * 100
                
        # Dijkstra's Algorithm implementation
        pq = []
        # Tuple format: (cumulative_cost, order_counter, r, c, first_action_in_path)
        heapq.heappush(pq, (0, next(self.counter), agent_r, agent_c, None))
        
        visited_costs = {}
        best_action = None
        
        while pq:
            cost, _, r, c, first_action = heapq.heappop(pq)
            
            if (r, c) == (goal_r, goal_c):
                best_action = first_action
                break
                
            if (r, c) in visited_costs and visited_costs[(r, c)] <= cost:
                continue
            visited_costs[(r, c)] = cost
            
            # 0: Up, 1: Down, 2: Left, 3: Right
            for dr, dc, a in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    val = board[nr, nc]
                    step_cost = val_costs.get(val, 1000)
                    
                    # Apply an increasing penalty if the agent has repeatedly failed to enter this cell
                    b_count = self.blocked_counts.get((nr, nc), 0)
                    step_cost += 500 * b_count
                    
                    new_cost = cost + step_cost
                    
                    if (nr, nc) not in visited_costs or new_cost < visited_costs[(nr, nc)]:
                        fa = a if first_action is None else first_action
                        heapq.heappush(pq, (new_cost, next(self.counter), nr, nc, fa))
                        
        # Fallback if entirely boxed in or at the goal
        if best_action is None:
            dr = goal_r - agent_r
            dc = goal_c - agent_c
            if abs(dr) > abs(dc):
                best_action = 1 if dr > 0 else 0
            else:
                best_action = 3 if dc > 0 else 2
                
        self.last_action = best_action
        return best_action