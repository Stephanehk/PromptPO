from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation


class NoiseWorldBoard1Policy:
    def __init__(self):
        self.V = None
        self.goal = None
        self.board = None

    def compute_V(self, board, goal_r, goal_c, n):
        """
        Computes the shortest path cost to the goal using Bellman-Ford / Value Iteration.
        Costs scale exponentially with board value to strongly prefer traversing lower-valued 
        (safer/less noisy) cells whenever possible.
        """
        V = np.full((n, n), float('inf'))
        V[goal_r, goal_c] = 0.0
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for _ in range(n * n):
            new_V = np.copy(V)
            for r in range(n):
                for c in range(n):
                    if r == goal_r and c == goal_c:
                        continue
                    
                    min_cost = float('inf')
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            actual_r, actual_c = nr, nc
                        else:
                            # If hitting the boundary, the agent stays in the same cell
                            actual_r, actual_c = r, c
                            
                        # Cost of entering the state
                        if actual_r == goal_r and actual_c == goal_c:
                            step_cost = 1.0
                        else:
                            # 10^v strongly penalizes higher values (traps, high noise, obstacles)
                            step_cost = float(10 ** board[actual_r, actual_c])
                            
                        cost = step_cost + V[actual_r, actual_c]
                        if cost < min_cost:
                            min_cost = cost
                    new_V[r, c] = min_cost
            
            # Early stopping if values have converged
            if np.array_equal(V, new_V):
                break
            V = new_V
            
        return V

    def act(self, obs: NoiseWorldObservation) -> int:
        n = int(np.round(np.sqrt(len(obs.board))))
        scale = max(n - 1, 1)
        
        agent_r = int(round(obs.agent_row * scale))
        agent_c = int(round(obs.agent_col * scale))
        goal_r = int(round(obs.goal_row * scale))
        goal_c = int(round(obs.goal_col * scale))
        
        if agent_r == goal_r and agent_c == goal_c:
            return 0  # Already at goal
            
        board = np.array(obs.board).reshape((n, n))
        
        # Recompute V if the environment configuration or goal has changed
        if self.V is None or self.goal != (goal_r, goal_c) or not np.array_equal(self.board, board):
            self.goal = (goal_r, goal_c)
            self.board = board
            self.V = self.compute_V(board, goal_r, goal_c, n)
            
        best_action = 0
        min_cost = float('inf')
        
        # 0: decreasing vertical (Up), 1: increasing vertical (Down)
        # 2: decreasing horizontal (Left), 3: increasing horizontal (Right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for a, (dr, dc) in enumerate(moves):
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < n and 0 <= nc < n:
                actual_r, actual_c = nr, nc
            else:
                actual_r, actual_c = agent_r, agent_c
                
            if actual_r == goal_r and actual_c == goal_c:
                step_cost = 1.0
            else:
                step_cost = float(10 ** board[actual_r, actual_c])
                
            cost = step_cost + self.V[actual_r, actual_c]
            
            # Tie-breaking defaults to earlier actions in the enumeration
            if cost < min_cost:
                min_cost = cost
                best_action = a
                
        return best_action