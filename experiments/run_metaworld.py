"""
Example: Meta-World MT1 (Sawyer v3) tasks.

Set ``env_name`` to any of:
``button_press_v3``, ``pick_place_v3``, ``door_open_v3``, ``drawer_open_v3``.
"""

from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    PromptPO(
        env_name="button_press_v3",
        num_episodes=5,
        model_name="gemini-3.1-pro-preview",
        run_n="",
        num_rounds=10,
        n_gens_per_round=10,
        manual_reasoning=True,
    ).train(resume=False)


if __name__ == "__main__":
    main()
