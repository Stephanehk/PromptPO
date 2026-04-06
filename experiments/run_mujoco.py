"""
Example: MuJoCo Gymnasium v5 tasks.

Set ``env_name`` to any of:
``ant_v5``, ``halfcheetah_v5``, ``hopper_v5``, ``humanoid_v5``, ``swimmer_v5``,
``reacher_v5``, ``inverted_double_pendulum_v5``.
"""

from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    PromptPO(
        env_name="ant_v5",
        num_episodes=5,
        model_name="gemini-3.1-pro-preview",
        run_n="",
        num_rounds=10,
        n_gens_per_round=10,
        manual_reasoning=True,
    ).train(resume=False)


if __name__ == "__main__":
    main()
