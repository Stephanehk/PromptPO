# PromptPO experiment examples

Each script runs **`PromptPO`** with defaults matching the CLI (`num_episodes=5`, `model_name="gemini-3.1-pro-preview"`, `run_n=""`, `num_rounds=10`, `n_gens_per_round=1`, `manual_reasoning=False`, `train(resume=False)` unless you edit the file).

| Script | Family | Set `env_name` to any of |
|--------|--------|---------------------------|
| `run_mujoco.py` | MuJoCo Gymnasium `*_v5` | `ant_v5`, `halfcheetah_v5`, `hopper_v5`, `humanoid_v5`, `swimmer_v5`, `reacher_v5`, `inverted_double_pendulum_v5` |
| `run_metaworld.py` | Meta-World MT1 | `button_press_v3`, `pick_place_v3`, `door_open_v3`, `drawer_open_v3` |
| `run_point_maze.py` | Point Maze v3 | `point_maze_large_v3`, `point_maze_large_diverse_g_v3` |
| `run_noise_world.py` | NoiseWorld | `noise_world_board_0` … `noise_world_board_6` |
| `run_synthetic.py` | Flow / sim benchmarks | `traffic`, `pandemic`, `glucose` |

From the repo root (with `direct_policy_learning_for_release` on `PYTHONPATH` as `direct_policy_learning`):

```bash
python -m experiments.run_mujoco
```

(Adjust the module path if your layout differs.)

Requires Vertex/Gemini credentials and dependencies for the env you select (MuJoCo, Meta-World, etc.).
