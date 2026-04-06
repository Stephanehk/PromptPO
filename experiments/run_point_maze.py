"""
Example: Farama Point Maze v3.

Set ``env_name`` to ``point_maze_large_v3`` or ``point_maze_large_diverse_g_v3``.
"""

from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    PromptPO(
        env_name="point_maze_large_v3",
        num_episodes=5,
        model_name="gemini-3.1-pro-preview",
        run_n="",
        num_rounds=10,
        n_gens_per_round=10,
        manual_reasoning=True,
    ).train(resume=False)


if __name__ == "__main__":
    main()
