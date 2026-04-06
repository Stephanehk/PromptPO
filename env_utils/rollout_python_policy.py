"""
Roll out a Python policy class against a Gym-style environment.

This module is standalone and does not depend on CPL checkpoints.
Assumptions:
- The policy object exposes `act(observation)` and returns an action valid for env.step.
- The environment exposes `reset()` and `step(action)`.
- `reset()` returns either `obs` or `(obs, info)`.
- `step(action)` returns either:
  - `(next_obs, reward, done, info)` (legacy Gym), or
  - `(next_obs, reward, terminated, truncated, info)` (Gymnasium).

Each episode stops when the environment signals `terminated` or `truncated` (e.g. MuJoCo
fall / unhealthy state, Meta-World time limit, or env time limit), or when an optional
caller `max_steps` cap is reached—whichever comes first.
"""

import argparse
import os
import pickle

from .meta_world_env_names import (
    META_WORLD_GYM_MT1_ID,
    is_meta_world_env_name,
    meta_world_mt1_task_name,
)
from .mujoco_env_names import (
    is_mujoco_env_name,
    mujoco_gym_make_id,
    mujoco_supports_terminate_when_unhealthy,
)
from .point_maze_env_names import (
    is_point_maze_env_name,
    point_maze_gym_make_id,
    point_maze_max_episode_steps,
)
from .noise_world_env_names import (
    is_noise_world_env_name,
    noise_world_ensure_full_standard_board_codes,
    noise_world_grid_n,
    noise_world_prerequisite_pair,
    noise_world_seed,
)
from .point_maze_maps import point_maze_custom_maze_map
from direct_policy_learning.observations import (
    NoiseWorldObservation,
    MetaWorldButtonPressV3Observation,
    MetaWorldDoorOpenV3Observation,
    MetaWorldDrawerOpenV3Observation,
    MetaWorldPickPlaceV3Observation,
    MujocoAntObservation,
    MujocoHalfCheetahObservation,
    MujocoHopperObservation,
    MujocoHumanoidObservation,
    MujocoInvertedDoublePendulumObservation,
    MujocoReacherObservation,
    MujocoSwimmerObservation,
    PointMazeLargeV3Observation,
    PointMazeLargeDiverseGV3Observation,
)
# from direct_policy_learning.policy_scratch_pad import ThresholdStagePolicy, TrafficPolicy

_MUJOCO_OBS_BY_ENV = {
    "ant_v5": MujocoAntObservation,
    "halfcheetah_v5": MujocoHalfCheetahObservation,
    "hopper_v5": MujocoHopperObservation,
    "humanoid_v5": MujocoHumanoidObservation,
    "swimmer_v5": MujocoSwimmerObservation,
    "reacher_v5": MujocoReacherObservation,
    "inverted_double_pendulum_v5": MujocoInvertedDoublePendulumObservation,
}

_META_WORLD_OBS_BY_ENV = {
    "button_press_v3": MetaWorldButtonPressV3Observation,
    "pick_place_v3": MetaWorldPickPlaceV3Observation,
    "door_open_v3": MetaWorldDoorOpenV3Observation,
    "drawer_open_v3": MetaWorldDrawerOpenV3Observation,
}

_POINT_MAZE_OBS_BY_ENV = {
    "point_maze_large_v3": PointMazeLargeV3Observation,
    "point_maze_large_diverse_g_v3": PointMazeLargeDiverseGV3Observation,
}

_NOISE_WORLD_OBS_BY_ENV = {
    "noise_world_board_0": NoiseWorldObservation,
    "noise_world_board_1": NoiseWorldObservation,
    "noise_world_board_2": NoiseWorldObservation,
    "noise_world_board_3": NoiseWorldObservation,
    "noise_world_board_4": NoiseWorldObservation,
    "noise_world_board_5": NoiseWorldObservation,
    "noise_world_board_6": NoiseWorldObservation,
}

_ROLLOUT_OBS_BY_ENV = dict(_MUJOCO_OBS_BY_ENV)
_ROLLOUT_OBS_BY_ENV.update(_META_WORLD_OBS_BY_ENV)
_ROLLOUT_OBS_BY_ENV.update(_POINT_MAZE_OBS_BY_ENV)
_ROLLOUT_OBS_BY_ENV.update(_NOISE_WORLD_OBS_BY_ENV)


def _assert_valid_rollout_env_name(env_name):
    if env_name in ("traffic", "glucose", "pandemic"):
        return
    assert (
        is_mujoco_env_name(env_name)
        or is_meta_world_env_name(env_name)
        or is_point_maze_env_name(env_name)
        or is_noise_world_env_name(env_name)
    ), (
        "env_name must be traffic, glucose, pandemic, a supported mujoco *_v5 id, "
        "a supported Meta-World MT1 env id, a supported point_maze_*_v3 id, "
        "or a noise_world_board_* id"
    )


def _parse_reset_output(reset_output):
    """
    Return `(obs, info)` from env.reset output.

    Assumption validated:
    - reset output is either obs, or a tuple/list of length 2 `(obs, info)`.
    """
    if isinstance(reset_output, (tuple, list)):
        assert len(reset_output) == 2, (
            "env.reset() tuple/list output must have length 2: (obs, info)"
        )
        obs, info = reset_output
        return obs, info
    return reset_output, {}


def _parse_step_output(step_output):
    """
    Normalize env.step output to `(next_obs, reward, terminated, truncated, info)`.

    Assumptions validated:
    - step output is tuple/list with length 4 or 5.
    - Legacy length-4 `done` means the episode ended (treated as `terminated`; truncated
      is unknown and set to False).

    Episode should end when `terminated or truncated` (caller may also cap with max_steps).
    """
    assert isinstance(step_output, (tuple, list)), (
        "env.step(action) must return a tuple/list"
    )
    assert len(step_output) in (4, 5), (
        "env.step(action) output must have length 4 or 5"
    )
    if len(step_output) == 4:
        next_obs, reward, done, info = step_output
        return next_obs, reward, bool(done), False, info
    next_obs, reward, terminated, truncated, info = step_output
    return next_obs, reward, bool(terminated), bool(truncated), info


def _rollout_traffic_episode(env, policy, gt_reward_set, max_steps):
    """
    Roll out one traffic episode with TrafficObservation wrappers.
    """
    from flow_reward_misspecification.flow.envs.traffic_obs_wrapper import TrafficObservation
    import numpy as np

    _obs_np, _last_info = env.reset()
    obs = TrafficObservation()
    obs.update_obs_with_sim_state(
        env,
        np.zeros(env.action_space.shape, dtype=np.float32),
        {"crash": False},
    )

    step_idx = 0
    episode_return = 0.0
    episode_steps = []

    while True:
        if max_steps is not None and step_idx >= max_steps:
            break
        action = policy.act(obs)
        _next_obs_np, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        next_obs = TrafficObservation()
        next_obs.update_obs_with_sim_state(env, action, info)

        modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)
        episode_steps.append(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "modified_reward": modified_reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
            }
        )
        episode_return += float(modified_reward)
        obs = next_obs
        step_idx += 1
        if terminated or truncated:
            break

    return episode_steps, episode_return


def _rollout_glucose_episode(env, policy, gt_reward_set, max_steps):
    """
    Roll out one glucose episode with GlucoseObservation wrappers.
    """
    from bgp.simglucose.envs.glucose_obs_wrapper import GlucoseObservation

    _obs_np, _info = env.reset()
    obs = GlucoseObservation()
    obs.update_obs_with_sim_state(env)

    step_idx = 0
    episode_return = 0.0
    episode_steps = []

    while True:
        if max_steps is not None and step_idx >= max_steps:
            break
        action = policy.act(obs)
        _next_obs_np, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        next_obs = GlucoseObservation()
        next_obs.update_obs_with_sim_state(env)

        modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)
        episode_steps.append(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "modified_reward": modified_reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
            }
        )
        episode_return += float(modified_reward)
        obs = next_obs
        step_idx += 1
        if terminated or truncated:
            break

    return episode_steps, episode_return


def _rollout_mujoco_episode(env, policy, gt_reward_set, max_steps, env_name):
    """
    Roll out one episode with the observation wrapper for this env_name.

    env_name is a supported mujoco *_v5 id, Meta-World MT1 id, point_maze_large_v3,
    or a noise_world_board_* id.
    """
    assert env_name in _ROLLOUT_OBS_BY_ENV, "unsupported rollout env_name: %s" % env_name
    obs_ctor = _ROLLOUT_OBS_BY_ENV[env_name]
    reset_out = env.reset()
    obs_np, _info = _parse_reset_output(reset_out)
    obs = obs_ctor()
    obs.fill_from_env(env, obs_np)

    step_idx = 0
    episode_return = 0.0
    episode_steps = []

    while True:
        if max_steps is not None and step_idx >= max_steps:
            break
        action = policy.act(obs)
        step_out = env.step(action)
        next_obs_np, reward, terminated, truncated, info = _parse_step_output(step_out)
        done = bool(terminated) or bool(truncated)
        next_obs = obs_ctor()
        next_obs.fill_from_env(env, next_obs_np)

        modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)
        episode_steps.append(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "modified_reward": modified_reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
            }
        )
        episode_return += float(modified_reward)
        obs = next_obs
        step_idx += 1
        if terminated or truncated:
            break

    return episode_steps, episode_return


def _rollout_pandemic_episode(env, policy, gt_reward_set, max_steps):
    """
    Roll out one pandemic episode with native observation objects.
    """
    obs, _obs_np, _info = env.reset_keep_obs_obj()

    step_idx = 0
    episode_return = 0.0
    episode_steps = []

    while True:
        if max_steps is not None and step_idx >= max_steps:
            break
        action = policy.act(obs)
        next_obs, _next_obs_np, reward, terminated, truncated, info = env.step_keep_obs_obj(
            action
        )
        done = bool(terminated) or bool(truncated)

        modified_reward = gt_reward_set.calculate_reward(obs, action, next_obs)
        episode_steps.append(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "reward": reward,
                "modified_reward": modified_reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "info": info,
            }
        )
        episode_return += float(modified_reward)
        obs = next_obs
        step_idx += 1
        if terminated or truncated:
            break

    return episode_steps, episode_return


def rollout_python_policy(
    env_name, env, policy, gt_reward_set, num_episodes=1, max_steps=None
):
    """
    Roll out `policy` on `env` for `num_episodes`.

    Inputs:
    - env_name: traffic, glucose, pandemic, a supported mujoco *_v5 id, Meta-World MT1,
      point_maze_large_v3, or noise_world_board_*.
    - env: Gym-style environment with reset/step.
    - policy: object with `act(observation) -> action`.
    - gt_reward_set: synth-human reward object with calculate_reward(obs, action, next_obs).
    - num_episodes: number of episodes to run.
    - max_steps: optional per-episode step cap. If None, the episode runs until the
      environment returns `terminated` or `truncated`.

    Output:
    - dict with:
      - `trajectories`: list of episodes; each episode is a list of per-step dicts
        (each step includes `terminated`, `truncated`, and `done` = either flag).
      - `episode_returns`: sum of rewards for each episode
    """
    _assert_valid_rollout_env_name(env_name)
    assert hasattr(policy, "act"), "policy must define act(observation)"
    assert callable(policy.act), "policy.act must be callable"
    assert hasattr(gt_reward_set, "calculate_reward"), (
        "gt_reward_set must define calculate_reward(obs, action, next_obs)"
    )
    assert isinstance(num_episodes, int) and num_episodes > 0, (
        "num_episodes must be a positive integer"
    )
    assert max_steps is None or (isinstance(max_steps, int) and max_steps > 0), (
        "max_steps must be None or a positive integer"
    )

    trajectories = []
    episode_returns = []

    for _episode_idx in range(num_episodes):
        if env_name == "traffic":
            episode_steps, episode_return = _rollout_traffic_episode(
                env=env, policy=policy, gt_reward_set=gt_reward_set, max_steps=max_steps
            )
        elif env_name == "glucose":
            episode_steps, episode_return = _rollout_glucose_episode(
                env=env, policy=policy, gt_reward_set=gt_reward_set, max_steps=max_steps
            )
        elif (
            is_mujoco_env_name(env_name)
            or is_meta_world_env_name(env_name)
            or is_point_maze_env_name(env_name)
            or is_noise_world_env_name(env_name)
        ):
            episode_steps, episode_return = _rollout_mujoco_episode(
                env=env,
                policy=policy,
                gt_reward_set=gt_reward_set,
                max_steps=max_steps,
                env_name=env_name,
            )
        else:
            episode_steps, episode_return = _rollout_pandemic_episode(
                env=env, policy=policy, gt_reward_set=gt_reward_set, max_steps=max_steps
            )
        trajectories.append(episode_steps)
        episode_returns.append(episode_return)

    return {
        "trajectories": trajectories,
        "episode_returns": episode_returns,
    }


def _make_env(
    env_name,
    exp_tag,
    glucose_universal,
    pandemic_town_size,
    pandemic_obs_history_size,
    pandemic_num_days_in_obs,
):
    """
    Build env from known project configs like rollout_policy.py.

    Assumptions validated:
    - env_name is traffic/glucose/pandemic, a supported mujoco *_v5 id, Meta-World MT1,
      point_maze_large_v3, or noise_world_board_*.
    - Constructed env supports reset/step.
    """
    _assert_valid_rollout_env_name(env_name)

    if env_name == "traffic":
        from flow.utils.registry import make_create_env
        from utils.traffic_config import get_config

        env_configs = get_config(exp_tag)
        create_env, _env_nm = make_create_env(
            params=env_configs["flow_params_default"],
            reward_specification=env_configs["reward_specification"],
            reward_fun=env_configs["reward_fun"],
            reward_scale=env_configs["reward_scale"],
        )
        env = create_env()
    elif env_name == "glucose":
        from bgp.simglucose.envs.simglucose_gym_env import SimglucoseEnv
        from utils.glucose_config import get_config

        env = SimglucoseEnv(config=get_config(universal=glucose_universal))
    elif is_mujoco_env_name(env_name):
        import gymnasium as gym

        gym_id = mujoco_gym_make_id(env_name)
        if mujoco_supports_terminate_when_unhealthy(env_name):
            env = gym.make(gym_id, terminate_when_unhealthy=True)
        else:
            env = gym.make(gym_id)
    elif is_meta_world_env_name(env_name):
        import metaworld  # registers Meta-World Gymnasium environments
        import gymnasium as gym

        task = meta_world_mt1_task_name(env_name)
        env = gym.make(META_WORLD_GYM_MT1_ID, env_name=task, seed=0)
    elif is_point_maze_env_name(env_name):
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        gym_id = point_maze_gym_make_id(env_name)
        max_ep = point_maze_max_episode_steps(env_name)
        env = gym.make(
            gym_id,
            maze_map=point_maze_custom_maze_map(env_name),
            continuing_task=False,
            max_episode_steps=max_ep,
        )
    elif is_noise_world_env_name(env_name):
        import gymnasium as gym

        from direct_policy_learning.no_prior_envs.noise_world_env import register_noise_world

        register_noise_world()
        n = noise_world_grid_n(env_name)
        seed = noise_world_seed(env_name)
        env = gym.make(
            "NoiseWorld-v0",
            n=n,
            seed=seed,
            prerequisite_pair=noise_world_prerequisite_pair(env_name),
            ensure_all_standard_board_codes=noise_world_ensure_full_standard_board_codes(
                env_name
            ),
        )
    else:
        from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
        from utils.pandemic_config import get_config

        env_config = get_config(town_size=pandemic_town_size)
        env = PandemicPolicyGymEnv(
            config=env_config,
            obs_history_size=pandemic_obs_history_size,
            num_days_in_obs=pandemic_num_days_in_obs,
        )

    assert hasattr(env, "reset") and callable(env.reset), (
        "constructed env must implement reset()"
    )
    assert hasattr(env, "step") and callable(env.step), (
        "constructed env must implement step(action)"
    )
    return env


def _make_gt_reward_set(env_name, gt_rf):
    """
    Build synth-human GT reward set and select the specific reward id.
    """
    if env_name == "traffic":
        from utils.traffic_gt_rew_fns import SynthHumanTrafficRewardFunction

        gt_reward_set = SynthHumanTrafficRewardFunction()
    elif env_name == "glucose":
        from utils.glucose_gt_rew_fns import SynthHumanGlucoseRewardFunction

        gt_reward_set = SynthHumanGlucoseRewardFunction()
    elif is_mujoco_env_name(env_name):
        if env_name == "ant_v5":
            from direct_policy_learning.reward_functions.ant_gt_rew_fns import AntRewardFunction

            gt_reward_set = AntRewardFunction()
        elif env_name == "halfcheetah_v5":
            from direct_policy_learning.reward_functions.halfcheetah_gt_rew_fns import (
                HalfcheetahRewardFunction,
            )

            gt_reward_set = HalfcheetahRewardFunction()
        elif env_name == "hopper_v5":
            from direct_policy_learning.reward_functions.hopper_gt_rew_fns import HopperRewardFunction

            gt_reward_set = HopperRewardFunction()
        elif env_name == "humanoid_v5":
            from direct_policy_learning.reward_functions.humanoid_gt_rew_fns import (
                HumanoidRewardFunction,
            )

            gt_reward_set = HumanoidRewardFunction()
        elif env_name == "swimmer_v5":
            from direct_policy_learning.reward_functions.swimmer_gt_rew_fns import (
                SwimmerRewardFunction,
            )

            gt_reward_set = SwimmerRewardFunction()
        elif env_name == "reacher_v5":
            from direct_policy_learning.reward_functions.reacher_gt_rew_fns import ReacherRewardFunction

            gt_reward_set = ReacherRewardFunction()
        elif env_name == "inverted_double_pendulum_v5":
            from direct_policy_learning.reward_functions.inverted_double_pendulum_gt_rew_fns import (
                InvertedDoublePendulumRewardFunction,
            )

            gt_reward_set = InvertedDoublePendulumRewardFunction()
        else:
            assert False, "unhandled mujoco env_name: %r" % (env_name,)
    elif is_meta_world_env_name(env_name):
        if env_name == "button_press_v3":
            from direct_policy_learning.reward_functions.button_press_v3_gt_rew_fns import (
                ButtonPressV3RewardFunction,
            )

            gt_reward_set = ButtonPressV3RewardFunction()
        elif env_name == "pick_place_v3":
            from direct_policy_learning.reward_functions.pick_place_v3_gt_rew_fns import (
                PickPlaceV3RewardFunction,
            )

            gt_reward_set = PickPlaceV3RewardFunction()
        elif env_name == "door_open_v3":
            from direct_policy_learning.reward_functions.door_open_v3_gt_rew_fns import (
                DoorOpenV3RewardFunction,
            )

            gt_reward_set = DoorOpenV3RewardFunction()
        elif env_name == "drawer_open_v3":
            from direct_policy_learning.reward_functions.drawer_open_v3_gt_rew_fns import (
                DrawerOpenV3RewardFunction,
            )

            gt_reward_set = DrawerOpenV3RewardFunction()
        else:
            assert False, "unhandled meta_world env_name: %r" % (env_name,)
    elif is_point_maze_env_name(env_name):
        if env_name == "point_maze_large_v3":
            from direct_policy_learning.reward_functions.point_maze_large_v3_gt_rew_fns import (
                PointMazeLargeV3RewardFunction,
            )

            gt_reward_set = PointMazeLargeV3RewardFunction()
        elif env_name == "point_maze_large_diverse_g_v3":
            from direct_policy_learning.reward_functions.point_maze_large_diverse_g_v3_gt_rew_fns import (
                PointMazeLargeDiverseGV3RewardFunction,
            )

            gt_reward_set = PointMazeLargeDiverseGV3RewardFunction()
        else:
            assert False, "unhandled point maze env_name: %r" % (env_name,)
    elif is_noise_world_env_name(env_name):
        from direct_policy_learning.reward_functions.noise_world_gt_rew_fns import (
            NoiseWorldRewardFunction,
        )

        gt_reward_set = NoiseWorldRewardFunction(env_name=env_name)
    else:
        from utils.pandemic_gt_rew_fns import SynthHumanPandemicRewardFunction

        gt_reward_set = SynthHumanPandemicRewardFunction()
    gt_reward_set.set_specific_reward(int(gt_rf))
    return gt_reward_set


def main():
    parser = argparse.ArgumentParser(
        description="Roll out the placeholder Policy class in this file."
    )
    parser.add_argument(
        "--env_name",
        required=True,
        choices=(
            "traffic",
            "glucose",
            "pandemic",
            "ant_v5",
            "halfcheetah_v5",
            "hopper_v5",
            "humanoid_v5",
            "swimmer_v5",
            "reacher_v5",
            "inverted_double_pendulum_v5",
            "button_press_v3",
            "pick_place_v3",
            "door_open_v3",
            "drawer_open_v3",
            "point_maze_large_v3",
            "point_maze_large_diverse_g_v3",
            "noise_world_board_0",
            "noise_world_board_1",
            "noise_world_board_2",
            "noise_world_board_3",
            "noise_world_board_4",
            "noise_world_board_5",
            "noise_world_board_6",
        ),
        help="Environment to roll out in.",
    )
    # parser.add_argument(
    #     "--default_action",
    #     type=float,
    #     default=0.0,
    #     help="Constant action used by placeholder Policy.act(observation).",
    # )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to roll out.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per episode.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="rollout_data/python_policy_rollout.pkl",
        help="Where to pickle rollout output.",
    )
    parser.add_argument("--exp_tag", type=str, default="singleagent_merge_bus")
    parser.add_argument("--glucose_universal", action="store_true", default=False)
    parser.add_argument("--pandemic_town_size", type=str, default="tiny")
    parser.add_argument("--pandemic_obs_history_size", type=int, default=3)
    parser.add_argument("--pandemic_num_days_in_obs", type=int, default=8)
    parser.add_argument(
        "--gt_rf",
        type=str,
        default="0",
        help="Synth-human GT reward id used to compute reported returns.",
    )
    args = parser.parse_args()

    env = _make_env(
        env_name=args.env_name,
        exp_tag=args.exp_tag,
        glucose_universal=args.glucose_universal,
        pandemic_town_size=args.pandemic_town_size,
        pandemic_obs_history_size=args.pandemic_obs_history_size,
        pandemic_num_days_in_obs=args.pandemic_num_days_in_obs,
    )
    gt_reward_set = _make_gt_reward_set(args.env_name, args.gt_rf)
    policy = TrafficPolicy()
    # policy = ThresholdStagePolicy()
    rollout = rollout_python_policy(
        env_name=args.env_name,
        env=env,
        policy=policy,
        gt_reward_set=gt_reward_set,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(rollout, f)

    mean_return = sum(rollout["episode_returns"]) / len(rollout["episode_returns"])
    print(f"episode_returns={rollout['episode_returns']}")
    print(
        f"Saved rollout to {args.save_path}. "
        f"episodes={args.num_episodes} gt_rf={args.gt_rf} mean_return={mean_return:.6f}"
    )
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
