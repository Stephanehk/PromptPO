"""
Example: Synthetic / external sim benchmarks (Flow traffic, pandemic, glucose).

Set ``env_name`` to ``traffic``, ``pandemic``, or ``glucose`` (requires those
projects installed and on the Python path as in the full codebase).
"""

from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    PromptPO(
        env_name="traffic",
        num_episodes=5,
        model_name="gemini-3.1-pro-preview",
        run_n="",
        num_rounds=10,
        n_gens_per_round=10,
        manual_reasoning=True,
    ).train(resume=False)


if __name__ == "__main__":
    main()
