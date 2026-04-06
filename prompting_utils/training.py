"""
One full run: load contexts, optionally resume, generate Feedback then iterative policies.

This module holds the procedural details; see ``train_policy_via_prompting`` for the
high-level method description.
"""

import json
import os
import traceback

from direct_policy_learning.env_utils.metaworld_rollout_metrics import (
    meta_world_episode_successes_from_trajectories,
)
from direct_policy_learning.env_utils.meta_world_env_names import (
    is_meta_world_env_name,
    meta_world_reward_fn_basename,
)
from direct_policy_learning.env_utils.mujoco_env_names import (
    is_mujoco_env_name,
    mujoco_reward_fn_basename,
)
from direct_policy_learning.env_utils.noise_world_env_names import (
    is_noise_world_env_name,
    noise_world_reward_fn_basename,
)
from direct_policy_learning.env_utils.point_maze_env_names import (
    is_point_maze_env_name,
    point_maze_reward_fn_basename,
)
from direct_policy_learning.env_utils.rollout_python_policy import (
    _make_env,
    _make_gt_reward_set,
    rollout_python_policy,
)
from direct_policy_learning.prompting_utils.code_loading import (
    materialize_feedback_class,
    materialize_policy_class,
)
from direct_policy_learning.prompting_utils.constants import MAX_ROLLOUT_FIX_ATTEMPTS
from direct_policy_learning.prompting_utils.gemini_client import call_gemini
from direct_policy_learning.prompting_utils.paths import GENERATED_POLICIES_DIR, PACKAGE_ROOT
from direct_policy_learning.prompting_utils.prompt_builders import (
    build_feedback_prompt,
    build_manual_reflection_prompt,
    build_policy_fix_prompt,
    build_prompt,
    build_refinement_context,
)
from direct_policy_learning.prompting_utils.serialization import (
    artifact_suffix,
    assert_resume_matches_args,
    extract_python_code_block,
    noise_world_uses_no_key_info_obs,
    read_text,
    save_training_state,
)
from direct_policy_learning.prompting_utils.env_kinds import (
    get_horizon_for_env,
)
from direct_policy_learning.prompting_utils.trajectory_feedback import (
    compute_round_trajectory_feedback_with_repairs,
)


def run_training_once(args, force_fresh_start=False):
    use_resume = args.resume and (not force_fresh_start)
    assert args.num_episodes == 5, "This script is intended to roll out exactly 5 episodes."
    assert args.n_gens_per_round >= 1, "n_gens_per_round must be at least 1."
    horizon = get_horizon_for_env(args.env_name)

    base_dir = PACKAGE_ROOT
    if is_mujoco_env_name(args.env_name):
        reward_fname = mujoco_reward_fn_basename(args.env_name)
        reward_path = os.path.join(base_dir, "reward_functions", reward_fname)
    elif is_meta_world_env_name(args.env_name):
        reward_fname = meta_world_reward_fn_basename(args.env_name)
        reward_path = os.path.join(base_dir, "reward_functions", reward_fname)
    elif is_point_maze_env_name(args.env_name):
        reward_fname = point_maze_reward_fn_basename(args.env_name)
        reward_path = os.path.join(base_dir, "reward_functions", reward_fname)
    elif is_noise_world_env_name(args.env_name):
        reward_fname = noise_world_reward_fn_basename(args.env_name)
        reward_path = os.path.join(base_dir, "reward_functions", reward_fname)
    else:
        reward_path = os.path.join(
            base_dir, "reward_functions", "%s_gt_rew_fns.py" % args.env_name
        )
    if noise_world_uses_no_key_info_obs(args):
        obs_ctx_basename = "%s_no_key_info_obs_context.txt" % args.env_name
    else:
        obs_ctx_basename = "%s_obs_context.txt" % args.env_name
    obs_ctx_path = os.path.join(base_dir, "env_contexts", obs_ctx_basename)
    policy_ctx_path = os.path.join(
        base_dir, "env_contexts", "%s_policy_context.txt" % args.env_name
    )
    reward_ctx_path = None
    if is_meta_world_env_name(args.env_name):
        reward_ctx_path = os.path.join(
            base_dir, "env_contexts", "%s_reward_context.txt" % args.env_name
        )
    elif is_noise_world_env_name(args.env_name):
        reward_ctx_path = os.path.join(
            base_dir, "env_contexts", "noise_world_reward_context.txt"
        )

    assert os.path.exists(reward_path), f"Missing reward file: {reward_path}"
    assert os.path.exists(obs_ctx_path), f"Missing obs context file: {obs_ctx_path}"
    assert os.path.exists(policy_ctx_path), f"Missing policy context file: {policy_ctx_path}"
    if reward_ctx_path is not None:
        assert os.path.isfile(reward_ctx_path), (
            "Missing reward context for prompts: %s" % reward_ctx_path
        )
        reward_src = read_text(reward_ctx_path)
    else:
        reward_src = read_text(reward_path)
    obs_context = read_text(obs_ctx_path)
    policy_context = read_text(policy_ctx_path)
    base_prompt = build_prompt(
        env_name=args.env_name,
        n_timesteps=horizon,
        reward_src=reward_src,
        obs_context=obs_context,
        policy_context=policy_context,
    )
    feedback_prompt = build_feedback_prompt(
        env_name=args.env_name,
        n_timesteps=horizon,
        reward_src=reward_src,
        obs_context=obs_context,
        policy_context=policy_context,
    )
    suffix = artifact_suffix(args)
    save_generated_code_path = os.path.join(
        GENERATED_POLICIES_DIR,
        f"{args.env_name}_prompted_policy_generated{suffix}.py",
    )
    save_feedback_code_path = os.path.join(
        GENERATED_POLICIES_DIR,
        f"{args.env_name}_feedback_generated{suffix}.py",
    )
    resume_state_path = os.path.join(
        GENERATED_POLICIES_DIR,
        f"{args.env_name}_prompt_training_state{suffix}.json",
    )
    save_dir = os.path.dirname(save_generated_code_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    state = None
    if use_resume and not os.path.isfile(resume_state_path):
        print(
            "resume requested but no state file yet; starting fresh. path=%s"
            % resume_state_path
        )
        use_resume = False
    if use_resume:
        with open(resume_state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        assert_resume_matches_args(state, args, horizon)
        feedback_code = state["feedback_code"]
        feedback_obj = materialize_feedback_class(feedback_code)
        policy_history = state["policy_history"]
        per_round_returns = state["per_round_returns"]
        with open(save_feedback_code_path, "w", encoding="utf-8") as f:
            f.write(feedback_code)
        print(
            "resume_loaded=%s rounds_completed=%d feedback_restored=%s"
            % (resume_state_path, len(policy_history), save_feedback_code_path)
        )
    else:
        print(
            f"\n===== Prompt to LLM (Feedback class, before policy rounds) =====\n"
            f"{feedback_prompt}\n"
        )
        feedback_llm_output = call_gemini(
            prompt=feedback_prompt,
            model_name=args.model_name,
        )
        feedback_code = extract_python_code_block(feedback_llm_output)
        feedback_obj = materialize_feedback_class(feedback_code)
        with open(save_feedback_code_path, "w", encoding="utf-8") as f:
            f.write(feedback_code)
        print(
            f"saved_feedback_code={save_feedback_code_path} "
            f"feedback_class_instantiated=Feedback"
        )
        policy_history = []
        per_round_returns = []
        save_training_state(
            resume_state_path,
            args.env_name,
            horizon,
            args,
            feedback_code,
            policy_history,
            per_round_returns,
            "",
        )

    latest_policy_code = ""
    latest_returns = None
    latest_successes = None
    latest_class_name = ""
    if use_resume and policy_history:
        last_item = policy_history[-1]
        latest_policy_code = last_item["generated_policy"]
        latest_returns = last_item["episode_returns"]
        latest_class_name = state.get("latest_class_name", "")
        if is_meta_world_env_name(args.env_name):
            latest_successes = last_item["episode_successes"]

    start_round = len(policy_history)
    if start_round >= args.num_rounds:
        print(
            "No remaining policy rounds (completed=%d, num_rounds=%d). "
            "Writing final artifacts only."
            % (start_round, args.num_rounds)
        )
        if not policy_history:
            assert False, (
                "Nothing to finalize: policy_history is empty and "
                "num_rounds does not require further rounds."
            )
        with open(save_generated_code_path, "w", encoding="utf-8") as f:
            f.write(policy_history[-1]["generated_policy"])
        print(f"generated_class={latest_class_name}")
        print(f"horizon={horizon}")
        if is_meta_world_env_name(args.env_name):
            ls = policy_history[-1]["episode_successes"]
            print(f"episode_successes={ls}")
            mean_ls = sum(ls) / len(ls)
            print(f"mean_episode_success={mean_ls:.6f}")
        else:
            print(f"episode_returns={policy_history[-1]['episode_returns']}")
        print(f"saved_generated_code={save_generated_code_path}")
        print("per_round_summary:")
        for row in per_round_returns:
            ridx = row["round_index"]
            if is_meta_world_env_name(args.env_name):
                sucs = row["episode_successes"]
                ms = row["mean_success"]
                print(
                    f"  round_idx={ridx} episode_successes={sucs} "
                    f"mean_episode_success={ms:.6f}"
                )
            else:
                rets = row["episode_returns"]
                mret = row["mean_return"]
                print(
                    f"  round_idx={ridx} episode_returns={rets} mean_return={mret:.6f}"
                )
        return

    for round_idx in range(start_round, args.num_rounds):
        manual_reflection_text = ""
        if args.manual_reasoning and round_idx > 0:
            last_item = policy_history[-1]
            reflection_prompt = build_manual_reflection_prompt(
                args.env_name,
                horizon,
                reward_src,
                last_item["generated_policy"],
                last_item["episode_returns"],
                last_item.get("trajectory_feedback", ""),
                policy_history[:-1],
                last_item["round_index"],
                last_item.get("episode_successes")
                if is_meta_world_env_name(args.env_name)
                else None,
            )
            print(
                f"\n===== Manual reflection prompt (round {round_idx + 1}) =====\n"
                f"{reflection_prompt}\n"
            )
            manual_reflection_raw = call_gemini(
                prompt=reflection_prompt,
                model_name=args.model_name,
            )
            manual_reflection_text = manual_reflection_raw.strip()
            print(
                f"\n===== Manual reflection output (round {round_idx + 1}) =====\n"
                f"{manual_reflection_text}\n"
            )
            if manual_reflection_text:
                policy_history[-1]["manual_reflection_followup"] = (
                    manual_reflection_text
                )

        if round_idx == 0:
            prompt = base_prompt
        else:
            refinement_history, refinement_tail = build_refinement_context(
                policy_history=policy_history,
                env_name=args.env_name,
                n_timesteps=horizon,
                omit_manual_from_last=bool(manual_reflection_text),
            )
            reflection_block = ""
            if manual_reflection_text:
                reflection_block = (
                    "Reflection on the previous policy (use this to improve):\n"
                    f"{manual_reflection_text}\n\n"
                )
            prompt = (
                f"{base_prompt}\n\n"
                f"{reflection_block}"
                f"{refinement_history}\n\n"
                f"{refinement_tail}"
            )

        round_save_path = os.path.join(
            GENERATED_POLICIES_DIR,
            f"{args.env_name}_prompted_policy_generated_round_{round_idx + 1}"
            f"{suffix}.py",
        )

        best_mean_score = None
        best_policy_code = None
        best_class_name = None
        best_rollout = None
        best_gen_index = None
        total_rollout_repairs_round = 0

        for gen_idx in range(args.n_gens_per_round):
            gen_label = f"round {round_idx + 1}, gen {gen_idx + 1}/{args.n_gens_per_round}"
            print(
                f"\n===== Prompt to LLM ({gen_label}) =====\n{prompt}\n"
            )
            llm_output = call_gemini(
                prompt=prompt,
                model_name=args.model_name,
            )
            policy_code = extract_python_code_block(llm_output)
            policy, class_name = materialize_policy_class(policy_code, args.env_name)

            env = _make_env(
                env_name=args.env_name,
                exp_tag=args.exp_tag,
                glucose_universal=args.glucose_universal,
                pandemic_town_size=args.pandemic_town_size,
                pandemic_obs_history_size=args.pandemic_obs_history_size,
                pandemic_num_days_in_obs=args.pandemic_num_days_in_obs,
            )
            gt_reward_set = _make_gt_reward_set(args.env_name, args.gt_rf)
            rollout_fix_count = 0
            rollout = None
            while True:
                try:
                    rollout = rollout_python_policy(
                        env_name=args.env_name,
                        env=env,
                        policy=policy,
                        gt_reward_set=gt_reward_set,
                        num_episodes=args.num_episodes,
                        max_steps=horizon,
                    )
                    break
                except Exception:
                    err_text = traceback.format_exc()
                    print(
                        f"\n===== Rollout failed ({gen_label}, "
                        f"repair_attempt={rollout_fix_count + 1}) =====\n{err_text}"
                    )
                    rollout_fix_count += 1
                    assert rollout_fix_count <= MAX_ROLLOUT_FIX_ATTEMPTS, (
                        f"Rollout failed after {MAX_ROLLOUT_FIX_ATTEMPTS} repair attempts "
                        f"({gen_label})."
                    )
                    if hasattr(env, "close"):
                        env.close()
                    fix_prompt = build_policy_fix_prompt(
                        env_name=args.env_name,
                        horizon=horizon,
                        reward_src=reward_src,
                        obs_context=obs_context,
                        policy_context=policy_context,
                        crashed_policy_code=policy_code,
                        error_traceback=err_text,
                    )
                    print(
                        f"\n===== Prompt to LLM (policy repair only, {gen_label}) "
                        f"=====\n{fix_prompt}\n"
                    )
                    fix_output = call_gemini(
                        prompt=fix_prompt,
                        model_name=args.model_name,
                    )
                    policy_code = extract_python_code_block(fix_output)
                    policy, class_name = materialize_policy_class(policy_code, args.env_name)
                    env = _make_env(
                        env_name=args.env_name,
                        exp_tag=args.exp_tag,
                        glucose_universal=args.glucose_universal,
                        pandemic_town_size=args.pandemic_town_size,
                        pandemic_obs_history_size=args.pandemic_obs_history_size,
                        pandemic_num_days_in_obs=args.pandemic_num_days_in_obs,
                    )
                    gt_reward_set = _make_gt_reward_set(args.env_name, args.gt_rf)
            if hasattr(env, "close"):
                env.close()
            total_rollout_repairs_round += rollout_fix_count
            if rollout_fix_count > 0:
                print(
                    f"{gen_label} rollout_repairs={rollout_fix_count} "
                    "(repaired policy used for this candidate's score)"
                )

            if is_meta_world_env_name(args.env_name):
                scores_candidate = meta_world_episode_successes_from_trajectories(
                    rollout["trajectories"]
                )
                mean_score_candidate = sum(scores_candidate) / len(scores_candidate)
                print(
                    f"{gen_label} generated_class={class_name} "
                    f"mean_episode_success={mean_score_candidate:.6f} "
                    f"episode_successes={scores_candidate}"
                )
            else:
                scores_candidate = rollout["episode_returns"]
                mean_score_candidate = sum(scores_candidate) / len(scores_candidate)
                print(
                    f"{gen_label} generated_class={class_name} "
                    f"mean_return={mean_score_candidate:.6f} "
                    f"episode_returns={scores_candidate}"
                )
            if best_mean_score is None or mean_score_candidate > best_mean_score:
                best_mean_score = mean_score_candidate
                best_policy_code = policy_code
                best_class_name = class_name
                best_rollout = rollout
                best_gen_index = gen_idx + 1

        assert best_rollout is not None, "Internal error: no successful rollout candidate."
        policy_code = best_policy_code
        class_name = best_class_name
        rollout = best_rollout
        if args.n_gens_per_round > 1:
            if is_meta_world_env_name(args.env_name):
                sel_msg = (
                    f"mean_episode_success={best_mean_score:.6f}"
                )
            else:
                sel_msg = f"mean_return={best_mean_score:.6f}"
            print(
                f"\n===== Round {round_idx + 1} selected gen {best_gen_index}/"
                f"{args.n_gens_per_round} with {sel_msg} =====\n"
            )
        with open(round_save_path, "w", encoding="utf-8") as f:
            f.write(policy_code)
        if total_rollout_repairs_round > 0:
            print(
                f"round={round_idx + 1} total_rollout_repairs_across_gens="
                f"{total_rollout_repairs_round} "
                "(fixed winning policy saved and used for history / later rounds)"
            )

        returns = rollout["episode_returns"]
        (
            trajectory_feedback_text,
            feedback_obj,
            feedback_code,
        ) = compute_round_trajectory_feedback_with_repairs(
            feedback_obj,
            feedback_code,
            rollout["trajectories"],
            args.env_name,
            horizon,
            reward_src,
            obs_context,
            policy_context,
            args.model_name,
            save_feedback_code_path,
        )
        if is_meta_world_env_name(args.env_name):
            episode_successes = meta_world_episode_successes_from_trajectories(
                rollout["trajectories"]
            )
            mean_success = sum(episode_successes) / len(episode_successes)
            per_round_returns.append(
                {
                    "round_index": round_idx + 1,
                    "episode_successes": list(episode_successes),
                    "mean_success": mean_success,
                }
            )
            print(
                f"round={round_idx + 1} generated_class={class_name} "
                f"episode_successes={episode_successes} "
                f"mean_episode_success={mean_success:.6f} "
                f"saved_generated_code={round_save_path}"
            )
        else:
            mean_return = sum(returns) / len(returns)
            per_round_returns.append(
                {
                    "round_index": round_idx + 1,
                    "episode_returns": list(returns),
                    "mean_return": mean_return,
                }
            )
            print(
                f"round={round_idx + 1} generated_class={class_name} "
                f"episode_returns={returns} mean_return={mean_return:.6f} "
                f"saved_generated_code={round_save_path}"
            )
        print(
            f"\n===== Trajectory feedback (Feedback.summarize_trajectory) "
            f"round={round_idx + 1} =====\n"
            f"{trajectory_feedback_text}\n"
            f"===== End trajectory feedback round={round_idx + 1} =====\n"
        )

        latest_policy_code = policy_code
        latest_returns = returns
        latest_class_name = class_name
        history_item = {
            "round_index": round_idx + 1,
            "generated_policy": policy_code,
            "episode_returns": returns,
            "trajectory_feedback": trajectory_feedback_text,
        }
        if is_meta_world_env_name(args.env_name):
            history_item["episode_successes"] = episode_successes
            latest_successes = episode_successes
        policy_history.append(history_item)
        save_training_state(
            resume_state_path,
            args.env_name,
            horizon,
            args,
            feedback_code,
            policy_history,
            per_round_returns,
            latest_class_name,
        )

    with open(save_generated_code_path, "w", encoding="utf-8") as f:
        f.write(latest_policy_code)
    print(f"generated_class={latest_class_name}")
    print(f"horizon={horizon}")
    if is_meta_world_env_name(args.env_name):
        if latest_successes is not None:
            print(f"episode_successes={latest_successes}")
            mean_ls = sum(latest_successes) / len(latest_successes)
            print(f"mean_episode_success={mean_ls:.6f}")
    else:
        print(f"episode_returns={latest_returns}")
    print(f"saved_generated_code={save_generated_code_path}")
    print("per_round_summary:")
    for row in per_round_returns:
        ridx = row["round_index"]
        if is_meta_world_env_name(args.env_name):
            sucs = row["episode_successes"]
            ms = row["mean_success"]
            print(
                f"  round_idx={ridx} episode_successes={sucs} "
                f"mean_episode_success={ms:.6f}"
            )
        else:
            rets = row["episode_returns"]
            mret = row["mean_return"]
            print(
                f"  round_idx={ridx} episode_returns={rets} mean_return={mret:.6f}"
            )
