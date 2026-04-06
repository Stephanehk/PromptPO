"""
Programmatic API for policy prompting: ``PromptPO`` mirrors CLI training with a small
constructor surface; fixed rollout / env flags use the same defaults as ``cli.parse_args``.
"""

import traceback
from types import SimpleNamespace

from direct_policy_learning.prompting_utils.constants import MAX_SUMMARIZE_RESTART_ATTEMPTS
from direct_policy_learning.prompting_utils.training import run_training_once
from direct_policy_learning.prompting_utils.trajectory_feedback import (
    is_summarize_trajectory_failure,
)


def _fixed_fields_from_cli_defaults():
    """
    Values for argparse flags that PromptPO does not expose; must match defaults in
    ``prompting_utils/cli.py``.
    """
    return {
        "exp_tag": "singleagent_merge_bus",
        "glucose_universal": False,
        "pandemic_town_size": "tiny",
        "pandemic_obs_history_size": 3,
        "pandemic_num_days_in_obs": 8,
        "gt_rf": "0",
        "use_key_info": False,
    }


class PromptPO:
    """
    Policy optimization via prompting: holds the user-facing hyperparameters, then
    ``train(resume=...)`` runs ``run_training_once`` with full argparse-equivalent state.
    """

    def __init__(
        self,
        env_name,
        num_episodes,
        model_name,
        run_n,
        num_rounds,
        n_gens_per_round,
        manual_reasoning,
    ):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.model_name = model_name
        self.run_n = run_n
        self.num_rounds = num_rounds
        self.n_gens_per_round = n_gens_per_round
        self.manual_reasoning = manual_reasoning

    def _training_args(self, resume):
        fixed = _fixed_fields_from_cli_defaults()
        return SimpleNamespace(
            env_name=self.env_name,
            num_episodes=self.num_episodes,
            model_name=self.model_name,
            run_n=self.run_n,
            num_rounds=self.num_rounds,
            n_gens_per_round=self.n_gens_per_round,
            manual_reasoning=self.manual_reasoning,
            resume=resume,
            exp_tag=fixed["exp_tag"],
            glucose_universal=fixed["glucose_universal"],
            pandemic_town_size=fixed["pandemic_town_size"],
            pandemic_obs_history_size=fixed["pandemic_obs_history_size"],
            pandemic_num_days_in_obs=fixed["pandemic_num_days_in_obs"],
            gt_rf=fixed["gt_rf"],
            use_key_info=fixed["use_key_info"],
        )

    def train(self, resume):
        """
        Run one logical training job. On Feedback trajectory-summary failures, may
        restart from scratch up to ``MAX_SUMMARIZE_RESTART_ATTEMPTS`` (same as the CLI).

        Assumptions:
        - ``resume`` is bool; other fields come from ``__init__`` and fixed CLI defaults.
        """
        args = self._training_args(resume)
        for attempt_idx in range(MAX_SUMMARIZE_RESTART_ATTEMPTS):
            try:
                run_training_once(args, force_fresh_start=(attempt_idx > 0))
                return
            except Exception:
                err_text = traceback.format_exc()
                is_summarize_error = is_summarize_trajectory_failure(err_text)
                if (not is_summarize_error) or (
                    attempt_idx + 1 >= MAX_SUMMARIZE_RESTART_ATTEMPTS
                ):
                    raise
                print(
                    "summarize_trajectory_failed attempt=%d/%d restarting_from_scratch=1\n%s"
                    % (
                        attempt_idx + 1,
                        MAX_SUMMARIZE_RESTART_ATTEMPTS,
                        err_text,
                    )
                )
