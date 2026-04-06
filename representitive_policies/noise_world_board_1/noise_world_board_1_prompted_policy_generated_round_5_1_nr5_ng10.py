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
        Computes the expected minimum cost to the goal using Value Iteration.
        Costs are assigned based on the cell being exited, reflecting the 
        expected steps to traverse a cell plus a variance penalty for higher noise.
        """
        V = np.full((n, n), float('inf'))
        V[goal_r, goal_c] = 0.0
        
        # Exponentially increasing cost map strongly penalizes high-noise cells
        # to ensure the agent prefers safer, lower-variance paths even if slightly longer.
        cost_map = {
            0: 1.0,
            1: 2.0,
            2: 5.0,
            3: 15.0,
            4: 50.0,
            5: 1000.0
        }
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for _ in range(n * n):
            new_V = np.copy(V)
            for r in range(n):
                for c in range(n):
                    if r == goal_r and c == goal_c:
                        continue
                    
                    val = int(board[r, c])
                    step_cost = cost_map.get(val, 1000.0)
                    
                    min_next_v = float('inf')
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            next_v = V[nr, nc]
                        else:
                            # Hitting the boundary keeps the agent in the same cell
                            next_v = V[r, c]
                            
                        if next_v < min_next_v:
                            min_next_v = next_v
                            
                    new_V[r, c] = step_cost + min_next_v
            
            # Early stopping if values have converged across the board
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
        
        # Cache and recompute V if the environment configuration or goal has changed
        if self.V is None or self.goal != (goal_r, goal_c) or not np.array_equal(self.board, board):
            self.goal = (goal_r, goal_c)
            self.board = board
            self.V = self.compute_V(board, goal_r, goal_c, n)
            
        best_action = 0
        min_next_v = float('inf')
        
        # Action Semantic mapping:
        # 0: decreasing vertical (Up), 1: increasing vertical (Down)
        # 2: decreasing horizontal (Left), 3: increasing horizontal (Right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for a, (dr, dc) in enumerate(moves):
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < n and 0 <= nc < n:
                actual_r, actual_c = nr, nc
            else:
                actual_r, actual_c = agent_r, agent_c
                
            next_v = self.V[actual_r, actual_c]
            
            # Choose the action that leads to the state with the lowest expected cost-to-go.
            if next_v < min_next_v:
                min_next_v = next_v
                best_action = a
                
        return best_action