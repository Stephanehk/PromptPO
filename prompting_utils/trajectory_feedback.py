"""
Run Feedback.summarize_trajectory on rollouts; repair Feedback via LLM on failure.
"""

import traceback

from direct_policy_learning.prompting_utils.code_loading import materialize_feedback_class
from direct_policy_learning.prompting_utils.constants import (
    MAX_FEEDBACK_TRAJECTORY_REPAIR_ATTEMPTS,
)
from direct_policy_learning.prompting_utils.gemini_client import call_gemini
from direct_policy_learning.prompting_utils.prompt_builders import build_feedback_fix_prompt
from direct_policy_learning.prompting_utils.serialization import extract_python_code_block


def compute_round_trajectory_feedback(feedback_obj, trajectories):
    lines = []
    for ep_idx, episode_steps in enumerate(trajectories):
        traj = [step["obs"] for step in episode_steps]
        summary = feedback_obj.summarize_trajectory(traj)
        lines.append(f"Episode {ep_idx + 1}: {summary}")
    return "\n".join(lines)


def compute_round_trajectory_feedback_with_repairs(
    feedback_obj,
    feedback_code,
    trajectories,
    env_name,
    n_timesteps,
    reward_src,
    obs_context,
    policy_context,
    model_name,
    save_feedback_code_path,
):
    attempt = 0
    while True:
        try:
            text = compute_round_trajectory_feedback(feedback_obj, trajectories)
            return text, feedback_obj, feedback_code
        except Exception:
            err_text = traceback.format_exc()
            attempt += 1
            if attempt > MAX_FEEDBACK_TRAJECTORY_REPAIR_ATTEMPTS:
                raise
            fix_prompt = build_feedback_fix_prompt(
                env_name=env_name,
                n_timesteps=n_timesteps,
                reward_src=reward_src,
                obs_context=obs_context,
                policy_context=policy_context,
                broken_feedback_code=feedback_code,
                error_traceback=err_text,
            )
            print(
                "\n===== Feedback summarize_trajectory failed; repair prompt "
                "(attempt %d/%d) =====\n%s\n"
                % (attempt, MAX_FEEDBACK_TRAJECTORY_REPAIR_ATTEMPTS, fix_prompt)
            )
            repair_raw = call_gemini(
                prompt=fix_prompt,
                model_name=model_name,
            )
            feedback_code = extract_python_code_block(repair_raw)
            feedback_obj = materialize_feedback_class(feedback_code)
            with open(save_feedback_code_path, "w", encoding="utf-8") as f:
                f.write(feedback_code)
            print(
                "feedback_repair_attempt=%d saved_feedback_code=%s"
                % (attempt, save_feedback_code_path)
            )


def is_summarize_trajectory_failure(traceback_text):
    assert traceback_text, "Expected non-empty traceback text."
    return (
        "summarize_trajectory" in traceback_text
        or "_compute_round_trajectory_feedback" in traceback_text
        or "compute_round_trajectory_feedback" in traceback_text
    )
