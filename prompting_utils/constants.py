"""API versions, Gemini defaults, and retry limits for prompting-based training."""

RESUME_STATE_VERSION = 1

GEMINI_PROJECT = "hai-gcp-llm-data"
GEMINI_LOCATION = "global"

MAX_ROLLOUT_FIX_ATTEMPTS = 10
MAX_SUMMARIZE_RESTART_ATTEMPTS = 3
MAX_FEEDBACK_TRAJECTORY_REPAIR_ATTEMPTS = 3
