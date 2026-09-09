import json
import unittest

from model_training.face_training.result_message import (
    FaceTrainingResultMessage,
    failure_detail,
    publish_face_training_result,
)


class _FakeSqsClient:
    def __init__(self) -> None:
        self.calls = []

    def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return {"MessageId": "result-message-1"}


class FaceTrainingResultMessageTests(unittest.TestCase):
    def test_publishes_standard_queue_message(self) -> None:
        client = _FakeSqsClient()
        message = FaceTrainingResultMessage(
            job_id=12,
            user_uuid="16dc9bb9-e097-415f-9241-8dee558d858b",
            clone_id=3,
            status="COMPLETED",
            attempt_number=2,
            result={"profileStatus": "READY_FOR_RENDERING"},
        )

        message_id = publish_face_training_result(
            client,
            queue_url="https://sqs.example/face-results",
            message=message,
        )

        self.assertEqual(message_id, "result-message-1")
        sent = client.calls[0]
        payload = json.loads(sent["MessageBody"])
        self.assertEqual(payload["eventType"], "FACE_PROFILE_BUILD_STATUS")
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["attemptNumber"], 2)
        self.assertNotIn("MessageGroupId", sent)

    def test_adds_fifo_identifiers(self) -> None:
        client = _FakeSqsClient()
        message = FaceTrainingResultMessage(
            job_id=12,
            user_uuid="16dc9bb9-e097-415f-9241-8dee558d858b",
            clone_id=3,
            status="FAILED",
            attempt_number=1,
            error={"code": "Failure", "message": "failed"},
        )

        publish_face_training_result(
            client,
            queue_url="https://sqs.example/face-results.fifo",
            message=message,
        )

        sent = client.calls[0]
        self.assertEqual(sent["MessageGroupId"], message.user_uuid)
        self.assertEqual(
            sent["MessageDeduplicationId"],
            "face-12-failed-1",
        )

    def test_sanitizes_and_limits_failure_message(self) -> None:
        detail = failure_detail(ValueError("secret\n" + "x" * 1200))

        self.assertEqual(detail["code"], "ValueError")
        self.assertNotIn("\n", detail["message"])
        self.assertLessEqual(len(detail["message"]), 1000)
        self.assertIs(detail["retryable"], True)


if __name__ == "__main__":
    unittest.main()
