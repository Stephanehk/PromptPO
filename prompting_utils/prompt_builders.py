"""
String templates for policy generation, repair, Feedback, refinement, and manual reflection.
"""

from direct_policy_learning.env_utils.meta_world_env_names import (
    is_meta_world_env_name,
    meta_world_feedback_obs_class_name,
)
from direct_policy_learning.env_utils.mujoco_env_names import (
    is_mujoco_env_name,
    mujoco_feedback_obs_class_name,
)
from direct_policy_learning.env_utils.point_maze_env_names import (
    is_point_maze_env_name,
    point_maze_feedback_obs_class_name,
)
from direct_policy_learning.env_utils.noise_world_env_names import (
    is_noise_world_env_name,
    noise_world_feedback_obs_class_name,
)
from direct_policy_learning.prompting_utils.env_kinds import (
    policy_class_base_for_env,
    reward_function_title_for_prompt,
    starter_code_for_env,
)


def build_prompt(env_name, n_timesteps, reward_src, obs_context, policy_context):
    starter_code = starter_code_for_env(env_name)
    pcb = policy_class_base_for_env(env_name)
    policy_class_name = "%sPolicy" % pcb
    rf = reward_function_title_for_prompt(env_name)

    prompt = f"""Observation Context: {obs_context}

Implementation Details: {policy_context}

{rf}: {reward_src}

Starter code (begin your policy implementation with these imports):
```python
{starter_code}

class {policy_class_name}:
    def act(self, obs):
        pass

```

Given the {rf}, implement a policy in python that inputs an observation and outputs an action. The policy should maximize the expected sum of rewards with respect to the {rf} over {n_timesteps} timesteps.
Think step-by-step internally before producing final code.
Implement the policy in a class called {policy_class_name} with a function act(obs) that takes in the observation and returns a valid action. Only return the generated class in python code surrounded in tags:
```python

```
Do not return any other code
"""
    return prompt


def build_policy_fix_prompt(
    env_name,
    horizon,
    reward_src,
    obs_context,
    policy_context,
    crashed_policy_code,
    error_traceback,
):
    starter_code = starter_code_for_env(env_name)
    pcb = policy_class_base_for_env(env_name)
    policy_class_name = "%sPolicy" % pcb
    rf = reward_function_title_for_prompt(env_name)

    prompt = f"""Observation Context: {obs_context}

Implementation Details: {policy_context}

{rf}: {reward_src}

Starter code (your fixed policy should remain compatible with these imports):
```python
{starter_code}

class {policy_class_name}:
    def act(self, obs):
        pass
```

The following policy code raised an error during a {horizon}-timestep rollout (5 episodes):

```python
{crashed_policy_code}
```

Traceback:
```
{error_traceback}
```

Fix the policy so it runs without error. The class must be named {policy_class_name} with act(obs) returning a valid action for this environment. Rely only on the observation context and implementation details above. Do not assume access to any other generated policies.

Only return the full fixed class in python code surrounded in tags:
```python

```
Do not return any other code
"""
    return prompt


def _feedback_obs_class_name(env_name):
    if env_name == "traffic":
        return "TrafficObservation"
    if env_name == "pandemic":
        return "PandemicObservation"
    if env_name == "glucose":
        return "GlucoseObservation"
    if is_mujoco_env_name(env_name):
        return mujoco_feedback_obs_class_name(env_name)
    if is_meta_world_env_name(env_name):
        return meta_world_feedback_obs_class_name(env_name)
    if is_point_maze_env_name(env_name):
        return point_maze_feedback_obs_class_name(env_name)
    if is_noise_world_env_name(env_name):
        return noise_world_feedback_obs_class_name(env_name)
    assert False, "unsupported env_name for Feedback: %r" % (env_name,)


def build_feedback_prompt(env_name, n_timesteps, reward_src, obs_context, policy_context):
    starter_code = starter_code_for_env(env_name)
    rf = reward_function_title_for_prompt(env_name)
    obs_cls = _feedback_obs_class_name(env_name)

    mw_success_note = ""
    if is_meta_world_env_name(env_name):
        mw_success_note = (
            "\nFor Meta-World environments: after each env.step, the returned info "
            'dictionary includes success; info["success"] denotes whether the robot '
            "succeeded at the task on that timestep.\n"
        )

    prompt = f"""Observation Context: {obs_context}

Implementation Details: {policy_context}

{rf}: {reward_src}

Starter code (begin your implementation with these imports):
```python
{starter_code}

class Feedback:
    def summarize_trajectory(self, traj):
        pass

```

Implement a class called Feedback with a method summarize_trajectory(self, traj).
Here traj is a list of observation objects (each is a {obs_cls} instance) for one
episode, in time order—the observations the policy saw before each step.
{mw_success_note}
The method should compute brief, useful statistics about the trajectory that will
help improve a policy to maximize the expected sum of rewards under the
{rf} over {n_timesteps} timesteps. Examples: aggregates
such as sum, min, max, or mean of reward-related quantities you can derive from
observations (or other brief summaries you find useful). Keep the returned string
as short as possible, but don't use abbreviations that the policy designer would not understand.

Return only the generated class in python code surrounded in tags:
```python

```
Do not return any other code
"""
    return prompt


def build_feedback_fix_prompt(
    env_name,
    n_timesteps,
    reward_src,
    obs_context,
    policy_context,
    broken_feedback_code,
    error_traceback,
):
    starter_code = starter_code_for_env(env_name)
    rf = reward_function_title_for_prompt(env_name)
    obs_cls = _feedback_obs_class_name(env_name)

    mw_success_note = ""
    if is_meta_world_env_name(env_name):
        mw_success_note = (
            "\nFor Meta-World: info from env.step is not passed into traj; only "
            "observation objects appear in traj, in time order.\n"
        )

    prompt = f"""Observation Context: {obs_context}

Implementation Details: {policy_context}

{rf}: {reward_src}

Starter code (your fixed Feedback must remain compatible with these imports):
```python
{starter_code}

class Feedback:
    def summarize_trajectory(self, traj):
        pass

```

The following Feedback class raised an error when summarize_trajectory(self, traj) was
called after rollouts. Here traj is a Python list of observation objects (each is a
{obs_cls} instance) for one episode, in time order—the observations the policy saw
before each step. Do not assume traj elements are numpy arrays unless the observation
context says so; use attributes and methods described for {obs_cls}.
{mw_success_note}
```python
{broken_feedback_code}
```

Traceback:
```
{error_traceback}
```

Fix the Feedback class so summarize_trajectory runs without error for any valid traj
as above and returns a short string summary. The class must be named Feedback.

Only return the full fixed class in python code surrounded in tags:
```python

```
Do not return any other code
"""
    return prompt


def build_refinement_context(
    policy_history, env_name, n_timesteps, omit_manual_from_last=False
):
    assert len(policy_history) > 0, "policy_history must be non-empty."
    use_mw_success = is_meta_world_env_name(env_name)
    round_and_means = []
    for item in policy_history:
        if use_mw_success:
            scores = item["episode_successes"]
        else:
            scores = item["episode_returns"]
        assert len(scores) > 0, "per-episode score list must be non-empty."
        mean_score = sum(scores) / len(scores)
        round_and_means.append((item["round_index"], mean_score))
    max_mean = max(m for _r, m in round_and_means)
    best_round_index = min(
        r for r, m in round_and_means if m == max_mean
    )

    blocks = []
    n_items = len(policy_history)
    for item_idx, item in enumerate(policy_history):
        round_index = item["round_index"]
        generated_policy = item["generated_policy"]
        if use_mw_success:
            rets = item["episode_returns"]
            sucs = item.get("episode_successes")
            score_heading = (
                "episode returns (sum of GT reward over the rollout) and episode successes "
                "(max over timesteps of info['success'] per episode), five episodes each:"
            )
            if sucs is not None:
                score_list = "returns: %s\nsuccesses: %s" % (rets, sucs)
            else:
                score_list = str(rets)
        else:
            score_list = item["episode_returns"]
            score_heading = "episode returns across 5 episodes:"
        hint_prefix = ""
        if round_index == best_round_index and len(policy_history) >= 10:
            hint_prefix = (
                "***Hint: This policy was your best generated policy so far**\n\n"
            )
        traj_fb = item.get("trajectory_feedback", "")
        traj_fb_block = ""
        if traj_fb:
            traj_fb_block = (
                f"\n\nAttempt {round_index} trajectory feedback "
                f"(Feedback.summarize_trajectory, one line per episode):\n{traj_fb}"
            )
        manual_fb = item.get("manual_reflection_followup", "")
        manual_fb_block = ""
        skip_manual = omit_manual_from_last and (item_idx == n_items - 1)
        if manual_fb and not skip_manual:
            manual_fb_block = (
                f"\n\nAttempt {round_index} manual reflection "
                f"(LLM comparison after this attempt):\n{manual_fb}"
            )
        block = f"""Attempt {round_index} generated policy:
{hint_prefix}{generated_policy}

Attempt {round_index} {score_heading}
{score_list}{traj_fb_block}{manual_fb_block}
"""
        blocks.append(block)
    history_text = "\n\n".join(blocks)
    pcb = policy_class_base_for_env(env_name)
    policy_class_name = "%sPolicy" % pcb
    rf = reward_function_title_for_prompt(env_name)

    tail_text = f"""Re-implement a {policy_class_name} to attain higher episode returns.

The policy should maximize the expected sum of rewards with respect to the {rf} over {n_timesteps} timesteps.
The policy should be implemented in a class called {policy_class_name} with a function act(obs) that takes in the observation and returns a valid action.
Only return the generated class in python code surrounded in tags:
```python

```
Do not return any other code
"""
    return history_text, tail_text


def build_manual_reflection_prompt(
    env_name,
    horizon,
    reward_src,
    generated_policy,
    episode_returns,
    trajectory_feedback_text,
    prior_policy_history_items,
    current_round_index,
    episode_successes=None,
):
    traj_section = ""
    if trajectory_feedback_text:
        traj_section = (
            f"Trajectory feedback (Feedback.summarize_trajectory, brief per episode):\n"
            f"{trajectory_feedback_text}\n\n"
        )

    mean_return_latest = sum(episode_returns) / len(episode_returns)
    mw_extra = ""
    if is_meta_world_env_name(env_name) and episode_successes is not None:
        assert len(episode_successes) == len(episode_returns), (
            "episode_successes and episode_returns must align per episode."
        )
        mean_success_latest = sum(episode_successes) / len(episode_successes)
        mw_extra = (
            f"\nLatest policy episode task success (max over timesteps of info['success'] "
            f"per episode): {list(episode_successes)}, "
            f"mean episode success {mean_success_latest:.6f}.\n"
        )

    prior_lines = []
    for item in prior_policy_history_items:
        rets = item["episode_returns"]
        mean_ret = sum(rets) / len(rets)
        line = (
            f"  Attempt {item['round_index']}: episode returns {list(rets)}, "
            f"mean return {mean_ret:.6f}"
        )
        if is_meta_world_env_name(env_name):
            sucs = item.get("episode_successes")
            if sucs is not None:
                mean_suc = sum(sucs) / len(sucs)
                line += (
                    f"; episode successes {list(sucs)}, mean episode success {mean_suc:.6f}"
                )
        prior_lines.append(line)
    if prior_lines:
        prior_block = (
            "Earlier policy attempts in this run (for comparison):\n"
            + "\n".join(prior_lines)
            + "\n\n"
        )
    else:
        prior_block = (
            "Earlier policy attempts in this run: none (this was the first policy "
            "trained in this run).\n\n"
        )

    rf = reward_function_title_for_prompt(env_name)
    compare_intro = "Compare the latest policy's returns to the earlier attempts listed above. "
    low_metric = (
        f"If the latest mean return is clearly very low for the {horizon}-timestep "
        "rollout, or clearly far below the best prior mean return in this run, "
        "or otherwise obviously suboptimal, open instead with that this policy did "
        "poorly (or equivalent), rather than implying improvement. "
    )
    if is_meta_world_env_name(env_name) and episode_successes is not None:
        compare_intro = (
            "Compare the latest policy's episode returns and task success rates to the "
            "earlier attempts listed above (training ranks candidates by mean episode "
            "success, but cumulative reward also matters for shaping). "
        )
        low_metric = (
            f"If the latest mean return and/or mean episode success is clearly very low "
            f"for the {horizon}-timestep rollout, or clearly far below the best prior "
            "means on those metrics in this run, or otherwise obviously suboptimal, open "
            "instead with that this policy did poorly (or equivalent), rather than "
            "implying improvement. "
        )

    return (
        f"{rf}: {reward_src}\n\n"
        f"{traj_section}"
        f"Latest policy to assess (attempt {current_round_index}) source:\n"
        f"{generated_policy}\n\n"
        f"Latest policy (attempt {current_round_index}) episode returns "
        f"(sum of rewards over each rollout): {list(episode_returns)}, "
        f"mean return {mean_return_latest:.6f}.{mw_extra}\n"
        f"{prior_block}"
        f"{compare_intro}"
        "Reply in 3-4 sentences total. Your first sentence must open with a comparison: "
        "for example that this policy did better than certain prior attempts, worse than "
        "certain prior attempts, or performed similarly to them (name attempt numbers if "
        "helpful). "
        f"{low_metric}"
        f"Then briefly relate this to maximizing the expected sum of rewards under the "
        f"{rf} over {horizon} timesteps. "
        "Reply with only that explanation, no code."
    )
