import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import httpx
except ImportError:
    class _HttpError(Exception):
        pass

    class _ConnectError(_HttpError):
        def __init__(self, message, *, request=None):
            super().__init__(message)
            self.request = request

    class _Request:
        def __init__(self, method, url):
            self.method = method
            self.url = url

    httpx = SimpleNamespace(
        HTTPError=_HttpError,
        ConnectError=_ConnectError,
        Request=_Request,
        Client=object,
    )
    sys.modules["httpx"] = httpx

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

    def test_posts_profile_score_components_when_available(self):
        response = Mock(status_code=204)
        client = Mock()
        client.post.return_value = response

        notify_personality_training_complete(
            12,
            score_components={
                "profileScore": 64.25,
                "dataReliabilityScore": 91.5,
                "penaltyScore": 1.5,
                "meaningfulAnswerCount": 5,
            },
            base_url="http://backend.internal:8080",
            secret="callback-secret",
            http_client=client,
        )

        client.post.assert_called_once_with(
            "http://backend.internal:8080/internal/clone-training/12/personality/complete",
            headers={
                "X-Clone-Training-Callback-Secret": "callback-secret"
            },
            json={
                "calculationVersion": "clone-similarity-v1",
                "profileScore": 64.25,
                "dataReliabilityScore": 91.5,
                "penaltyScore": 1.5,
            },
        )

    def test_posts_source_revision_so_retrained_scores_replace_old_ones(self):
        response = Mock(status_code=200)
        client = Mock()
        client.post.return_value = response

        notify_personality_training_complete(
            12,
            score_components={
                "profileScore": 71.0,
                "dataReliabilityScore": 92.0,
                "penaltyScore": 0.0,
            },
            source_revision=3,
            base_url="http://backend.internal:8080",
            secret="callback-secret",
            http_client=client,
        )

        self.assertEqual(
            client.post.call_args.kwargs["json"],
            {
                "calculationVersion": "clone-similarity-v1",
                "profileScore": 71.0,
                "dataReliabilityScore": 92.0,
                "penaltyScore": 0.0,
                "sourceRevision": 3,
            },
        )

    def test_rejects_non_positive_source_revision(self):
        client = Mock()
        with self.assertRaises(ValueError):
            notify_personality_training_complete(
                12,
                score_components={"profileScore": 1.0, "dataReliabilityScore": 1.0},
                source_revision=0,
                base_url="http://backend.internal:8080",
                secret="callback-secret",
                http_client=client,
            )
        client.post.assert_not_called()

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
