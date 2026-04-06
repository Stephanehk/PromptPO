"""
Meta-World MT1: derive per-episode task success from rollout step records.

Assumes each step dict may include ``info`` from ``env.step``; success is
``max_t info["success"]`` over the episode (bool or numeric).
"""


def episode_success_from_episode_steps(episode_steps):
    max_s = 0.0
    for step in episode_steps:
        info = step.get("info") or {}
        if "success" not in info:
            continue
        s = info["success"]
        if isinstance(s, bool):
            max_s = max(max_s, 1.0 if s else 0.0)
        else:
            max_s = max(max_s, float(s))
    return max_s


def meta_world_episode_successes_from_trajectories(trajectories):
    return [episode_success_from_episode_steps(ep) for ep in trajectories]
