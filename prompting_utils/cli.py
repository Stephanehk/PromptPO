"""Argument parser for ``train_policy_via_prompting``."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a policy with Gemini and roll it out."
    )
    parser.add_argument(
        "--env_name",
        required=True,
        choices=(
            "traffic",
            "pandemic",
            "glucose",
            "ant_v5",
            "halfcheetah_v5",
            "hopper_v5",
            "humanoid_v5",
            "swimmer_v5",
            "reacher_v5",
            "inverted_double_pendulum_v5",
            "button_press_v3",
            "pick_place_v3",
            "door_open_v3",
            "drawer_open_v3",
            "point_maze_large_v3",
            "point_maze_large_diverse_g_v3",
            "noise_world_board_0",
            "noise_world_board_1",
            "noise_world_board_2",
            "noise_world_board_3",
            "noise_world_board_4",
            "noise_world_board_5",
            "noise_world_board_6",
        ),
        help=(
            "Environment name (MuJoCo: <task>_v5 -> <Task>-v5; "
            "Meta-World: MT1 tasks e.g. button_press_v3 -> button-press-v3; "
            "Point Maze: point_maze_large_v3 -> PointMaze_Large-v3; "
            "point_maze_large_diverse_g_v3 -> PointMaze_Large_Diverse_G-v3; "
            "NoiseWorld: noise_world_board_* -> fixed board config)."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of rollout episodes (default 5).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model name (3.1 Pro preview default).",
    )
    parser.add_argument("--exp_tag", type=str, default="singleagent_merge_bus")
    parser.add_argument("--glucose_universal", action="store_true", default=False)
    parser.add_argument("--pandemic_town_size", type=str, default="tiny")
    parser.add_argument("--pandemic_obs_history_size", type=int, default=3)
    parser.add_argument("--pandemic_num_days_in_obs", type=int, default=8)
    parser.add_argument(
        "--gt_rf",
        type=str,
        default="0",
        help=(
            "GT reward id used in rollout score computation (env-specific). "
            "point_maze_*_v3 point mazes use sparse reward only; this flag is ignored. "
            "noise_world_board_* uses environment step reward; this flag is ignored."
        ),
    )
    parser.add_argument(
        "--run_n",
        type=str,
        default="",
        help=(
            "Suffix appended to generated policy filenames (e.g. run_0). "
            "If empty, use default names without this suffix."
        ),
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="Number of rounds of refinement (default 10).",
    )
    parser.add_argument(
        "--n_gens_per_round",
        type=int,
        default=1,
        help=(
            "Policies to sample per round with the same prompt; each is rolled out "
            "and the one with highest mean episode return is kept (default 1)."
        ),
    )
    parser.add_argument(
        "--manual_reasoning",
        action="store_true",
        default=False,
        help=(
            "If set, before each refinement round ask the LLM to briefly explain "
            "why the last policy did poorly, and include that text in the next "
            "policy-generation prompt."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "If generated_policies/{env_name}_prompt_training_state{suffix}.json "
            "exists next to this script (suffix matches "
            "artifact naming: run_n, num_rounds, n_gens_per_round, and for NoiseWorld "
            "boards 4–6 also _no_key_info unless --use_key_info), load it and "
            "continue from the next policy round (skips Feedback LLM call). If the "
            "file is missing, run a full fresh start. Stored config must match except "
            "--num_rounds may be increased to add more rounds."
        ),
    )
    parser.add_argument(
        "--use_key_info",
        action="store_true",
        default=False,
        help=(
            "NoiseWorld boards 4-6 only: use {env_name}_obs_context.txt "
            "instead of {env_name}_no_key_info_obs_context.txt."
        ),
    )
    return parser.parse_args()
