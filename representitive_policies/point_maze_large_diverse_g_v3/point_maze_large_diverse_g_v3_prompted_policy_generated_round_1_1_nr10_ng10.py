from abc import ABCMeta
import math
import numpy as np
from direct_policy_learning.observations.point_maze_large_diverse_g_v3_observation import PointMazeLargeDiverseGV3Observation

class PointMazeLargeDiverseGV3Policy:
    def __init__(self):
        # We explicitly represent the LARGE_MAZE_DIVERSE_G layout, with 'r' and 'g' designated as free space (0).
        self.grid = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

    def xy_to_cell(self, x, y):
        # Maps Mujoco continuous world coordinates to discrete maze array indices.
        # Gym robotics point maze is scaled to 1.0 per cell, centered precisely at (0, 0).
        # Bounds are roughly x in [-6.0, 6.0], y in [-4.5, 4.5] based on the (12, 9) size.
        col = int(math.floor(x + 6.0))
        row = int(math.floor(4.5 - y))
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return (row, col)

    def cell_to_xy(self, row, col):
        # Inverse mapping: calculates the exact center coordinates of a grid cell.
        x = col - 5.5
        y = 4.0 - row
        return x, y

    def a_star(self, start, goal):
        # Standard A* to find the shortest path in our fully connected orthogonal maze.
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            open_set.remove(current)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    # Treat the dynamically found start and goal as temporarily unblocked 
                    # for safety in case uniform spawn noise pushes a point slightly into walls.
                    if self.grid[nr][nc] == 0 or (nr, nc) == goal or (nr, nc) == start:
                        tentative_g = g_score[current] + 1
                        if tentative_g < g_score.get((nr, nc), float('inf')):
                            came_from[(nr, nc)] = current
                            g_score[(nr, nc)] = tentative_g
                            f_score[(nr, nc)] = tentative_g + abs(nr - goal[0]) + abs(nc - goal[1])
                            open_set.add((nr, nc))
        return None

    def act(self, obs):
        x = float(obs.observation_vector[0])
        y = float(obs.observation_vector[1])
        vx = float(obs.observation_vector[2])
        vy = float(obs.observation_vector[3])

        goal_x = float(obs.desired_goal[0])
        goal_y = float(obs.desired_goal[1])

        current_cell = self.xy_to_cell(x, y)
        goal_cell = self.xy_to_cell(goal_x, goal_y)

        # Re-compute route step-by-step
        path = self.a_star(current_cell, goal_cell)

        # Because of convexity, moving to the direct center of the strictly NEXT immediate mapped 
        # grid cell in our path assures zero obstacle collisions until reaching the goal's cell.
        if path and len(path) > 1:
            target_x, target_y = self.cell_to_xy(path[1][0], path[1][1])
        else:
            # Reached goal cell (or in a highly anomalous disconnected spot) -> converge to exact target
            target_x, target_y = goal_x, goal_y

        # Basic PD position-controller.
        dir_x = target_x - x
        dir_y = target_y - y

        KP = 3.0
        KD = 1.0

        action_x = KP * dir_x - KD * vx
        action_y = KP * dir_y - KD * vy

        # Actuator action space limits
        return np.clip(np.array([action_x, action_y], dtype=np.float32), -1.0, 1.0)