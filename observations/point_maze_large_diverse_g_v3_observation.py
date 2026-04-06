"""
Structured observation for Gymnasium-Robotics PointMaze_Large_Diverse_G-v3.

The environment returns a goal-conditioned dict observation (see Farama docs):
https://robotics.farama.org/envs/maze/point_maze/

Keys:
- observation: (4,) ball x, y, vx, vy
- desired_goal: (2,) target position (one of the maze cells marked as goal-capable)
- achieved_goal: (2,) current ball position (same as observation[:2])

Assumptions:
- obs_dict is the raw observation from env.reset() / env.step() for PointMaze_*-v3
  (after parsing reset/step tuples if needed).
- Each value is array-like convertible to float64 ndarray.

Lives under direct_policy_learning/observations/.
"""

import numpy as np


class PointMazeLargeDiverseGV3Observation:
    """
    PointMaze large diverse-G v3 observation bundle for policies and reward code.

    Action space: Box(-1, 1, (2,), float32) — force on the point mass in x and y (N).

    observation_vector — length 4: [ball_x, ball_y, ball_vx, ball_vy] in meters / (m/s).

    desired_goal — length 2: [goal_x, goal_y] world coordinates of the red target.

    achieved_goal — length 2: current ball (x, y); equals observation_vector[0:2].
    """

    def __init__(self):
        self.observation_vector = None
        self.desired_goal = None
        self.achieved_goal = None

    def fill_from_env(self, env, obs):
        """
        Populate fields from the environment's observation dict.

        Inputs:
        - env: unused; kept for the same call pattern as MuJoCo observation classes.
        - obs: dict with keys "observation", "desired_goal", "achieved_goal".

        Assumptions:
        - obs contains the three keys with shapes (4,), (2,), (2,).
        """
        assert isinstance(obs, dict), "Point Maze observation must be a dict."
        assert "observation" in obs, "obs dict must contain observation."
        assert "desired_goal" in obs, "obs dict must contain desired_goal."
        assert "achieved_goal" in obs, "obs dict must contain achieved_goal."
        self.observation_vector = np.asarray(obs["observation"], dtype=np.float64).copy()
        self.desired_goal = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
        self.achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
