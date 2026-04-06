"""
Filesystem anchors for the direct_policy_learning package.

``PACKAGE_ROOT`` is the directory that contains ``reward_functions/``, ``env_contexts/``,
and ``generated_policies/`` (sibling of this ``prompting_utils`` package).
"""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(_PACKAGE_DIR)
GENERATED_POLICIES_DIR = os.path.join(PACKAGE_ROOT, "generated_policies")
