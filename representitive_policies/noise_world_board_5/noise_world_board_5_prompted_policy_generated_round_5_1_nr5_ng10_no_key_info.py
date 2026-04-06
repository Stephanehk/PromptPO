from abc import ABCMeta
import numpy as np
from direct_policy_learning.observations import NoiseWorldObservation

class NoiseWorldBoard5Policy:
    def __init__(self):
        # Empirical transition model: T[state][action][next_state] = count
        self.T = {}
        for r in range(10):
            for c in range(10):
                self.T[(r, c)] = {0: {}, 1: {}, 2: {}, 3: {}}
                
        # Memory of actions that directly resulted in early episode termination
        self.fatal_actions = set()
        # Memory of terrain types that appear to be traps
        self.fatal_types = set()
        
        self.start_pos = None
        self.goal_pos = None
        
        self.episode_steps = 0
        self.last_pos = None
        self.last_action = None
        self.last_trailing_1 = None
        
    def get_target(self, r, c, a):
        if a == 0: return (r - 1, c)
        if a == 1: return (r + 1, c)
        if a == 2: return (r, c - 1)
        if a == 3: return (r, c + 1)
        return (r, c)

    def act(self, obs):
        # --- Observation Parsing ---
        if isinstance(obs, tuple) and len(obs) == 2:
            if hasattr(obs[0], 'agent_row') or isinstance(obs[1], dict):
                obs = obs[0]

        trailing_1 = None
        if hasattr(obs, 'agent_row') and hasattr(obs, 'board'):
            agent_row = int(round(obs.agent_row * 9))
            agent_col = int(round(obs.agent_col * 9))
            goal_row = int(round(obs.goal_row * 9))
            goal_col = int(round(obs.goal_col * 9))
            board = np.array(obs.board).reshape((10, 10))
            # Try parsing trailing scalars securely 
            try:
                obs_arr = np.array(obs, dtype=float).flatten()
                if obs_arr.size >= 106:
                    trailing_1 = obs_arr[104]
            except Exception:
                pass
        else:
            try:
                obs_arr = np.array(obs, dtype=float).flatten()
                agent_row = int(round(obs_arr[0] * 9))
                agent_col = int(round(obs_arr[1] * 9))
                goal_row = int(round(obs_arr[2] * 9))
                goal_col = int(round(obs_arr[3] * 9))
                board = obs_arr[4:104].reshape((10, 10))
                if obs_arr.size >= 106:
                    trailing_1 = obs_arr[104]
            except Exception:
                agent_row, agent_col, goal_row, goal_col = 0, 0, 9, 9
                board = np.zeros((10, 10))

        current_pos = (agent_row, agent_col)
        self.goal_pos = (goal_row, goal_col)

        if self.start_pos is None:
            self.start_pos = current_pos

        # --- Episode Boundary Detection ---
        is_reset = False
        # 1. Use trailing scalar if it's a temporal value that resets
        if self.last_trailing_1 is not None and trailing_1 is not None:
            if trailing_1 < self.last_trailing_1 - 0.001:
                is_reset = True
                
        # 2. Time horizon maximum
        if self.episode_steps >= 100:
            is_reset = True
            
        # 3. Spatial jumps violating grid adjacency physics
        if not is_reset and self.last_pos is not None:
            dr = abs(current_pos[0] - self.last_pos[0])
            dc = abs(current_pos[1] - self.last_pos[1])
            if dr > 1 or dc > 1:
                is_reset = True
            elif current_pos == self.start_pos and self.last_pos != self.start_pos and self.episode_steps > 0:
                # If we suddenly respawned at start (and didn't step there manually)
                target = self.get_target(self.last_pos[0], self.last_pos[1], self.last_action)
                if target != self.start_pos:
                    is_reset = True

        self.last_trailing_1 = trailing_1

        # --- Transition & Trap Memory Update ---
        if is_reset:
            if 0 < self.episode_steps < 100:
                # Episode ended prematurely, it was either a goal success or a trap fatality.
                if self.last_pos is not None and self.last_action is not None:
                    last_target = self.get_target(self.last_pos[0], self.last_pos[1], self.last_action)
                    if last_target != self.goal_pos:
                        # Agent died before horizon without targeting the goal: mark action as fatal
                        self.fatal_actions.add((self.last_pos, self.last_action))
                        if 0 <= last_target[0] < 10 and 0 <= last_target[1] < 10:
                            t_type = int(board[last_target[0], last_target[1]])
                            s_type = int(board[self.start_pos[0], self.start_pos[1]])
                            g_type = int(board[self.goal_pos[0], self.goal_pos[1]])
                            # Log the global terrain label of the trap to avoid paths taking identical tile types
                            if t_type != s_type and t_type != g_type:
                                self.fatal_types.add(t_type)
                    else:
                        # Targeting the goal resulted in early termination -> Reached Goal!
                        self.T[self.last_pos][self.last_action][self.goal_pos] = self.T[self.last_pos][self.last_action].get(self.goal_pos, 0) + 1
            
            self.episode_steps = 0
            self.last_pos = None
            self.last_action = None

        self.episode_steps += 1
        
        # Log successful organic stochastic transition step
        if self.last_pos is not None and self.last_action is not None and not is_reset:
            self.T[self.last_pos][self.last_action][current_pos] = self.T[self.last_pos][self.last_action].get(current_pos, 0) + 1

        if current_pos == self.goal_pos:
            self.last_pos = current_pos
            self.last_action = 0
            return 0

        # --- Value Iteration MDP Solver ---
        # Seamlessly incorporates map blockages and stochastic slip distributions
        V = {}
        for r in range(10):
            for c in range(10):
                V[(r, c)] = 0.0
                if int(board[r, c]) in self.fatal_types:
                    V[(r, c)] = -1000.0
                    
        V[self.goal_pos] = 1000.0
        
        # Propagate values for the known local graph
        for _ in range(100):
            new_V = {}
            for r in range(10):
                for c in range(10):
                    s = (r, c)
                    if s == self.goal_pos:
                        new_V[s] = 1000.0
                        continue
                    if int(board[r, c]) in self.fatal_types:
                        new_V[s] = -1000.0
                        continue
                        
                    max_v = -float('inf')
                    for a in range(4):
                        if (s, a) in self.fatal_actions:
                            val = -1000.0
                            if val > max_v: max_v = val
                            continue
                            
                        transitions = self.T[s][a]
                        total_count = sum(transitions.values())
                        
                        if total_count == 0:
                            # Optimistic initialization evaluates unseen edges ideally
                            nxt = self.get_target(r, c, a)
                            if 0 <= nxt[0] < 10 and 0 <= nxt[1] < 10:
                                val = -1.0 + V[nxt]
                            else:
                                val = -1.0 + V[s]
                        else:
                            expected_v = 0.0
                            for nxt, count in transitions.items():
                                expected_v += (count / total_count) * V[nxt]
                            val = -1.0 + expected_v
                            
                        if val > max_v:
                            max_v = val
                            
                    if max_v < -1000.0:
                        max_v = -1000.0
                    new_V[s] = max_v
            V = new_V

        # Extract optimal step from the empirical Value Iteration mapping
        best_a = 0
        best_val = -float('inf')
        for a in range(4):
            s = current_pos
            if (s, a) in self.fatal_actions:
                val = -1000.0
            else:
                transitions = self.T[s][a]
                total_count = sum(transitions.values())
                if total_count == 0:
                    nxt = self.get_target(s[0], s[1], a)
                    if 0 <= nxt[0] < 10 and 0 <= nxt[1] < 10:
                        val = -1.0 + V[nxt]
                    else:
                        val = -1.0 + V[s]
                else:
                    expected_v = 0.0
                    for nxt, count in transitions.items():
                        expected_v += (count / total_count) * V[nxt]
                    val = -1.0 + expected_v
            
            if val > best_val:
                best_val = val
                best_a = a
            elif val == best_val:
                # Direct geographic tie-breaker if mapping hits an identical plateau
                nxt = self.get_target(s[0], s[1], a)
                curr_dist = abs(nxt[0] - self.goal_pos[0]) + abs(nxt[1] - self.goal_pos[1])
                best_dist = abs(self.get_target(s[0], s[1], best_a)[0] - self.goal_pos[0]) + abs(self.get_target(s[0], s[1], best_a)[1] - self.goal_pos[1])
                if curr_dist < best_dist:
                    best_a = a

        self.last_pos = current_pos
        self.last_action = best_a
        return int(best_a)