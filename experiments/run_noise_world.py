"""
Example: NoiseWorld grid tasks (fixed board per id).

Set ``env_name`` to ``noise_world_board_1`` … ``noise_world_board_5``.
"""

from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    PromptPO(
        env_name="noise_world_board_q",
        num_episodes=5,
        model_name="gemini-3.1-pro-preview",
        run_n="",
        num_rounds=10,
        n_gens_per_round=10,
        manual_reasoning=True,
    ).train(resume=False)


if __name__ == "__main__":
    main()
