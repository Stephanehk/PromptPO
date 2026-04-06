"""
UTF-8 file reads, fenced-code extraction, JSON-safe snapshots, and --resume state.

Resume payloads pair with ``train_policy_via_prompting`` CLI via
``config_dict_for_resume`` / ``assert_resume_matches_args``.
"""

import json

from direct_policy_learning.env_utils.noise_world_env_names import (
    is_noise_world_env_name,
    noise_world_prerequisite_pair,
)
from direct_policy_learning.prompting_utils.constants import RESUME_STATE_VERSION


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_python_code_block(text):
    start_tag = "```python"
    end_tag = "```"
    start = text.find(start_tag)
    assert start != -1, "Model output must contain an opening ```python fence."
    start = start + len(start_tag)
    end = text.find(end_tag, start)
    assert end != -1, "Model output must contain a closing ``` fence."
    code = text[start:end].strip()
    assert code, "Extracted Python code block is empty."
    return code


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj
    return float(obj)


def noise_world_uses_no_key_info_obs(args):
    return (
        is_noise_world_env_name(args.env_name)
        and noise_world_prerequisite_pair(args.env_name)
        and (not args.use_key_info)
    )


def artifact_suffix(args):
    parts = []
    if args.run_n.strip():
        parts.append("_%s" % args.run_n.strip())
    if args.num_rounds != 20:
        parts.append("_nr%d" % args.num_rounds)
    if args.n_gens_per_round != 20:
        parts.append("_ng%d" % args.n_gens_per_round)
    if noise_world_uses_no_key_info_obs(args):
        parts.append("_no_key_info")
    return "".join(parts)


def config_dict_for_resume(args):
    return {
        "env_name": args.env_name,
        "num_rounds": args.num_rounds,
        "n_gens_per_round": args.n_gens_per_round,
        "run_n": args.run_n,
        "gt_rf": args.gt_rf,
        "num_episodes": args.num_episodes,
        "manual_reasoning": args.manual_reasoning,
        "model_name": args.model_name,
        "exp_tag": args.exp_tag,
        "glucose_universal": args.glucose_universal,
        "pandemic_town_size": args.pandemic_town_size,
        "pandemic_obs_history_size": args.pandemic_obs_history_size,
        "pandemic_num_days_in_obs": args.pandemic_num_days_in_obs,
        "use_key_info": args.use_key_info,
        "noise_world_no_key_info_obs": noise_world_uses_no_key_info_obs(args),
    }


def save_training_state(
    path,
    env_name,
    horizon,
    args,
    feedback_code,
    policy_history,
    per_round_returns,
    latest_class_name,
):
    payload = {
        "format_version": RESUME_STATE_VERSION,
        "env_name": env_name,
        "horizon": horizon,
        "config": config_dict_for_resume(args),
        "feedback_code": feedback_code,
        "policy_history": json_safe(policy_history),
        "per_round_returns": json_safe(per_round_returns),
        "latest_class_name": latest_class_name,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def assert_resume_matches_args(state, args, horizon):
    assert state.get("format_version") == RESUME_STATE_VERSION, (
        "Unsupported or missing format_version in resume state."
    )
    assert state["env_name"] == args.env_name, (
        "Resume state env_name %r does not match --env_name %r."
        % (state["env_name"], args.env_name)
    )
    assert state["horizon"] == horizon, (
        "Resume state horizon %s does not match current horizon %s."
        % (state["horizon"], horizon)
    )
    n_done = len(state["policy_history"])
    assert args.num_rounds >= n_done, (
        "--num_rounds (%d) must be >= completed rounds in state (%d)."
        % (args.num_rounds, n_done)
    )
    saved_cfg = dict(state["config"])
    if "use_key_info" not in saved_cfg:
        saved_cfg["use_key_info"] = False
    if "noise_world_no_key_info_obs" not in saved_cfg:
        saved_cfg["noise_world_no_key_info_obs"] = False
    current_cfg = config_dict_for_resume(args)
    assert set(saved_cfg.keys()) == set(current_cfg.keys()), (
        "Resume state config keys differ from current CLI."
    )
    for key in saved_cfg:
        if key == "num_rounds":
            continue
        assert saved_cfg[key] == current_cfg[key], (
            "Resume config mismatch on %r: saved %r vs current %r"
            % (key, saved_cfg[key], current_cfg[key])
        )
