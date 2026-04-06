from abc import ABCMeta
import numpy as np
import heapq
from direct_policy_learning.observations import NoiseWorldObservation


class NoiseWorldBoard3Policy:
    def __init__(self):
        self.n = 10

    def act(self, obs: NoiseWorldObservation) -> int:
        n = self.n
        # Extract and unnormalize positions
        agent_r = int(round(obs.agent_row * (n - 1)))
        agent_c = int(round(obs.agent_col * (n - 1)))
        goal_r = int(round(obs.goal_row * (n - 1)))
        goal_c = int(round(obs.goal_col * (n - 1)))

        # Reshape the flat board array to 2D
        board = np.array(obs.board).reshape((n, n))

        # Setup Dijkstra's algorithm to find the shortest path to the goal
        # Priority queue stores tuples of (cumulative_cost, r, c)
        pq = [(0.0, agent_r, agent_c)]
        
        # Track minimum distances to each cell
        dists = np.full((n, n), np.inf)
        dists[agent_r, agent_c] = 0.0
        
        # Store parents for path reconstruction: (r, c) -> (parent_r, parent_c, action_taken)
        parents = {}

        # Directions corresponding to actions 0, 1, 2, 3
        # 0: Up (decrease r), 1: Down (increase r), 2: Left (decrease c), 3: Right (increase c)
        directions = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

        while pq:
            d, r, c = heapq.heappop(pq)

            if d > dists[r, c]:
                continue

            if r == goal_r and c == goal_c:
                break

            for dr, dc, action in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    step_cost = 1.0
                    
                    # Apply a huge penalty if the cell is an obstacle/trap (represented by non-zero values).
                    # We don't penalize the goal cell even if it happens to be non-zero.
                    if board[nr, nc] != 0 and (nr, nc) != (goal_r, goal_c):
                        step_cost += 1000000.0
                    
                    # Tie-breaker logic: slightly penalize moving adjacent to hazards 
                    # to keep the agent centralized and robust to stochastic transitions
                    adj_walls = 0
                    for ar, ac in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        adj_r, adj_c = nr + ar, nc + ac
                        if 0 <= adj_r < n and 0 <= adj_c < n:
                            if board[adj_r, adj_c] != 0 and (adj_r, adj_c) != (goal_r, goal_c):
                                adj_walls += 1
                    
                    step_cost += 0.01 * adj_walls

                    new_dist = d + step_cost
                    if new_dist < dists[nr, nc]:
                        dists[nr, nc] = new_dist
                        parents[(nr, nc)] = (r, c, action)
                        heapq.heappush(pq, (new_dist, nr, nc))

        # Reconstruct path from goal to agent to find the first action
        if (goal_r, goal_c) in parents:
            curr = (goal_r, goal_c)
            path = []
            while curr != (agent_r, agent_c):
                p_r, p_c, act = parents[curr]
                path.append(act)
                curr = (p_r, p_c)
            
            if path:
                # Return the first action on the path (since we traced backward)
                return path[-1]
            else:
                return 0
        else:
            # Fallback heuristic if no path whatsoever exists
            if agent_r > goal_r: return 0
            if agent_r < goal_r: return 1
            if agent_c > goal_c: return 2
            if agent_c < goal_c: return 3
            return 0