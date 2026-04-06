"""
Generate a Python policy with Gemini (Vertex AI) and score it with environment rollouts.

**Flow**
1. Load reward source and observation / policy context text for ``--env_name``.
2. Ask the LLM for a ``Feedback`` class (trajectory summaries), then for a policy class
   per refinement round. Parse each reply as a single `` ```python`` … ``` `` block.
3. Instantiate the policy, roll out ``--num_episodes`` episodes (default 5), optionally
   repair crashy policies or Feedback via follow-up prompts (bounded retries).
4. Refinement rounds append prior attempts (returns, optional trajectory feedback, optional
   manual reflection) to the prompt. Meta-World ranks by mean episode *success*; other envs
   by mean episode return.
5. Save artifacts under ``generated_policies/`` next to this package; ``--resume`` continues
   from JSON state if config matches (``--num_rounds`` may increase).

Details: prompts, resume format, and artifact suffixes live under ``prompting_utils/``.
Rollout mechanics and env construction: ``env_utils/rollout_python_policy.py``.
"""

from direct_policy_learning.prompting_utils.cli import parse_args
from direct_policy_learning.prompting_utils.prompt_po import PromptPO


def main():
    args = parse_args()
    PromptPO(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        model_name=args.model_name,
        run_n=args.run_n,
        num_rounds=args.num_rounds,
        n_gens_per_round=args.n_gens_per_round,
        manual_reasoning=args.manual_reasoning,
    ).train(args.resume)


if __name__ == "__main__":
    main()
