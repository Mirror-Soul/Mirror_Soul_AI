from __future__ import annotations

import os
from typing import Any

import httpx


class CloneTrainingCallbackError(RuntimeError):
    pass


def notify_personality_training_complete(
    clone_id: int,
    *,
    base_url: str | None = None,
    secret: str | None = None,
    timeout_seconds: float = 10.0,
    http_client: Any | None = None,
) -> bool:
    resolved_base_url = (
        base_url or os.getenv("CLONE_TRAINING_CALLBACK_BASE_URL") or ""
    ).strip()
    resolved_secret = (
        secret or os.getenv("CLONE_TRAINING_CALLBACK_SECRET") or ""
    ).strip()

    if not resolved_base_url and not resolved_secret:
        return False
    if not resolved_base_url or not resolved_secret:
        raise CloneTrainingCallbackError(
            "CLONE_TRAINING_CALLBACK_BASE_URL and "
            "CLONE_TRAINING_CALLBACK_SECRET must be configured together"
        )
    if clone_id <= 0:
        raise ValueError("clone_id must be a positive integer")

    url = (
        f"{resolved_base_url.rstrip('/')}"
        f"/internal/clone-training/{clone_id}/personality/complete"
    )
    headers = {"X-Clone-Training-Callback-Secret": resolved_secret}

    try:
        if http_client is not None:
            response = http_client.post(url, headers=headers)
        else:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.post(url, headers=headers)
    except httpx.HTTPError as exc:
        raise CloneTrainingCallbackError(
            "Personality completion callback request failed"
        ) from exc

    if not 200 <= response.status_code < 300:
        raise CloneTrainingCallbackError(
            "Personality completion callback failed with HTTP "
            f"{response.status_code}"
        )
    return True
