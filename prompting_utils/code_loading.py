"""
Execute generated Python for Policy and Feedback classes and resolve the policy type.
"""

import numpy as np

from direct_policy_learning.prompting_utils.env_kinds import policy_class_base_for_env


def is_policy_candidate_type(cls):
    if not isinstance(cls, type):
        return False
    act_attr = getattr(cls, "act", None)
    return callable(act_attr)


def policy_class_name_excluded(cls):
    name_lower = cls.__name__.lower()
    return "observation" in name_lower


def resolve_policy_class(namespace, env_name):
    expected_class_name = "%sPolicy" % policy_class_base_for_env(env_name)
    lowercase_class_map = {
        name.lower(): value
        for name, value in namespace.items()
        if isinstance(name, str) and isinstance(value, type)
    }
    expected_lower = expected_class_name.lower()
    if expected_lower in lowercase_class_map:
        return lowercase_class_map[expected_lower], expected_class_name

    with_act = [
        cls
        for cls in namespace.values()
        if is_policy_candidate_type(cls) and not policy_class_name_excluded(cls)
    ]
    assert len(with_act) > 0, (
        f"Generated code must define class {expected_class_name} "
        f"(case-insensitive), or exactly one other class with act(obs). "
        f"No act-bearing classes found. Namespace keys: {sorted(namespace.keys())}."
    )
    assert len(with_act) == 1, (
        f"Generated code must define class {expected_class_name} "
        f"(case-insensitive). Without it, there must be exactly one class "
        f"with callable act(obs); found {len(with_act)}: "
        f"{[c.__name__ for c in with_act]}."
    )
    return with_act[0], with_act[0].__name__


def materialize_policy_class(code, env_name):
    namespace = {}
    exec(code, namespace)

    policy_cls, resolved_name = resolve_policy_class(namespace, env_name)
    policy = policy_cls()
    assert hasattr(policy, "act"), f"{resolved_name} must define act(obs)."
    assert callable(policy.act), f"{resolved_name}.act must be callable."
    return policy, resolved_name


def materialize_feedback_class(code):
    namespace = {"np": np}
    exec(code, namespace)
    assert "Feedback" in namespace, "Generated code must define class Feedback."
    feedback_cls = namespace["Feedback"]
    assert isinstance(feedback_cls, type), "Feedback must be a class."
    feedback_obj = feedback_cls()
    assert hasattr(feedback_obj, "summarize_trajectory"), (
        "Feedback must define summarize_trajectory(self, traj)."
    )
    assert callable(feedback_obj.summarize_trajectory), (
        "Feedback.summarize_trajectory must be callable."
    )
    return feedback_obj
