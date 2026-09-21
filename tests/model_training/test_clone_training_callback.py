import unittest
from unittest.mock import Mock, patch

import httpx

from model_training.clone_training_callback import (
    CloneTrainingCallbackError,
    notify_personality_training_complete,
)


class CloneTrainingCallbackTest(unittest.TestCase):
    def test_skips_callback_when_not_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            delivered = notify_personality_training_complete(12)

        self.assertFalse(delivered)

    def test_posts_completion_without_body(self):
        response = Mock(status_code=204)
        client = Mock()
        client.post.return_value = response

        delivered = notify_personality_training_complete(
            12,
            base_url="http://backend.internal:8080/",
            secret="callback-secret",
            http_client=client,
        )

        self.assertTrue(delivered)
        client.post.assert_called_once_with(
            "http://backend.internal:8080/internal/clone-training/12/personality/complete",
            headers={
                "X-Clone-Training-Callback-Secret": "callback-secret"
            },
        )

    def test_rejects_partial_configuration_without_leaking_secret(self):
        with self.assertRaises(CloneTrainingCallbackError) as context:
            notify_personality_training_complete(
                12,
                base_url="http://backend.internal:8080",
                secret="",
            )

        self.assertNotIn("callback-secret", str(context.exception))

    def test_raises_sanitized_error_for_http_failure(self):
        request = httpx.Request(
            "POST",
            "http://backend.internal:8080/internal/clone-training/12/personality/complete",
        )
        client = Mock()
        client.post.side_effect = httpx.ConnectError("connection failed", request=request)

        with self.assertRaises(CloneTrainingCallbackError) as context:
            notify_personality_training_complete(
                12,
                base_url="http://backend.internal:8080",
                secret="callback-secret",
                http_client=client,
            )

        self.assertEqual(
            str(context.exception),
            "Personality completion callback request failed",
        )

    def test_raises_for_non_success_response(self):
        client = Mock()
        client.post.return_value = Mock(status_code=401)

        with self.assertRaisesRegex(CloneTrainingCallbackError, "HTTP 401"):
            notify_personality_training_complete(
                12,
                base_url="http://backend.internal:8080",
                secret="callback-secret",
                http_client=client,
            )


if __name__ == "__main__":
    unittest.main()
