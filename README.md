# Direct policy learning (PromptPO)

Train a Python policy with a Gemini model and evaluate it with rollouts. The programmatic entry point is **`PromptPO`**; the CLI wraps the same logic.

## Running PromptPO

Train via **`train_policy_via_prompting.py`** (add this package to `PYTHONPATH` as `direct_policy_learning`, then run from a working directory where imports resolve):

```python
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
```

**Arguments (≤1 sentence each, with default):**

- `env_name` — benchmark id (`train_policy_via_prompting.py` `--help` lists choices); no default in the snippet (CLI requires it).
- `num_episodes` — number of rollout episodes per evaluation; default **5**.
- `model_name` — Gemini model name; default **`gemini-3.1-pro-preview`**.
- `run_n` — optional string appended to saved artifact names; default **""**.
- `num_rounds` — number of PromptPO update rounds; default **10**.
- `n_gens_per_round` — independent policy samples per round (best is kept); default **10**.
- `manual_reasoning` — if true, ask the LLM to reflect on the prior policy before the next round; default **True**.
- `train(resume=…)` — if true, load compatible state from `generated_policies/`; default **`False`**.

Other flags (traffic/pandemic/glucose options, `gt_rf`, `use_key_info`, …) are not passed through `PromptPO`; they use the same fixed defaults as the CLI (see `prompting_utils/prompt_po.py`).

## Prompts

Prompt strings (policy, Feedback, repair, refinement, manual reflection) live in **`prompting_utils/prompt_builders.py`**.

## Environment contexts

Per-environment text for the LLM is under **`env_contexts/`**. For each environment you will typically find:

- **`*_obs_context.txt`** — observation description.
- **`*_reward_context.txt`** — reward semantics where we expose a separate readable spec (e.g. Meta-World); otherwise the reward source may be the Python file under `reward_functions/`.
- **`*_policy_context.txt`** — extra implementation notes (e.g. action space, interfaces).

NoiseWorld boards with prerequisite milestones may use **`*_no_key_info_obs_context.txt`** unless you opt into full semantics via the CLI (`PromptPO` fixes the non-exposed flags to CLI defaults).

## Representative policies

**`representitive_policies/`** contains one subdirectory per environment with **example generated policy sources** from runs using **`n_gens_per_round == 10`** (filenames include `_ng10`): typically **refinement round 1** and **round 5** checkpoints.

## Example runs (all supported environments)

See **`experiments/`** for small scripts that call `PromptPO` with one representative `env_name` per family; copy and change `env_name` to match any benchmark listed in each file’s docstring.
