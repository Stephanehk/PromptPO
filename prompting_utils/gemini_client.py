"""
Vertex AI Gemini client for text prompts (single-turn generate_content).

Retries with backoff on empty output or API errors (same behavior as the original
training script).
"""

import time

from google import genai

from direct_policy_learning.prompting_utils.constants import (
    GEMINI_LOCATION,
    GEMINI_PROJECT,
)


def call_gemini(prompt, model_name):
    client = genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location=GEMINI_LOCATION,
    )
    max_retries = 5
    attempt_idx = 0
    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            output = response.text.strip()
            assert output, "Gemini returned empty output."
            return output
        except Exception as exc:
            if attempt_idx >= max_retries:
                raise
            retry_trial = attempt_idx + 1
            sleep_seconds = 30 * retry_trial
            print(
                f"Gemini call failed: {exc!r}. "
                f"retry_trial={retry_trial}/{max_retries} "
                f"sleep_seconds={sleep_seconds}"
            )
            time.sleep(sleep_seconds)
            attempt_idx += 1
