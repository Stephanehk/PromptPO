"""
Custom Point Maze layouts for direct_policy_learning.

The training script prompts the LLM with text from env_contexts/point_maze_large_v3_obs_context.txt.
That file must describe the same grid as POINT_MAZE_LARGE_V3_CUSTOM_MAZE_MAP below; if you edit
one, update the other.

This map is intentionally not the published Farama PointMaze_Large LARGE_MAZE so that policies
cannot rely on memorized online diagrams.

For point_maze_large_diverse_g_v3, rollouts use the published LARGE_MAZE_DIVERSE_G layout
(gymnasium_robotics.envs.maze.maps.LARGE_MAZE_DIVERSE_G); env_contexts must stay consistent.

Encoding (same as gymnasium_robotics):
- 1: wall
- 0: free cell (goal and reset positions are sampled from free cells when no g/r/c markers)
- "r": valid agent reset cell
- "g": goal-capable cell (one goal sampled among g cells per reset)
"""

# 9 rows × 10 columns. Row index i is top-to-bottom, column j is left-to-right.
POINT_MAZE_LARGE_V3_CUSTOM_MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# 9 rows × 12 columns — matches gymnasium_robotics LARGE_MAZE_DIVERSE_G for PointMaze_Large_Diverse_G-v3.
POINT_MAZE_LARGE_DIVERSE_G_V3_CUSTOM_MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 0, 0, 1, "g", 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, "g", 0, 1, 0, 0, "g", 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, "g", 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, "g", 0, "g", 1, 0, "g", 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def point_maze_custom_maze_map(env_name):
    """
    Return the maze_map list-of-lists passed to gym.make(..., maze_map=...) for this env.

    Assumptions:
    - env_name is a supported Point Maze id in point_maze_env_names.POINT_MAZE_ENV_NAMES.
    """
    if env_name == "point_maze_large_v3":
        return POINT_MAZE_LARGE_V3_CUSTOM_MAZE_MAP
    if env_name == "point_maze_large_diverse_g_v3":
        return POINT_MAZE_LARGE_DIVERSE_G_V3_CUSTOM_MAZE_MAP
    assert False, "no custom map for env_name: %r" % (env_name,)
