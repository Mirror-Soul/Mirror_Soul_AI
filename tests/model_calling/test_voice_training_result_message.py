import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-only")

from model_calling.voice_training.result_message import (
    VoiceTrainingResultMessage,
    publish_voice_training_result,
)
from model_calling.voice_training.worker import (
    DownloadedAudio,
    VoiceTrainingWorkerError,
    _fallback_voice_score,
    _handle_sqs_message,
    _parse_message,
    _completed_results,
    run_worker,
)


REQUEST = {
    "jobType": "VOICE_TRAINING",
    "source": "ONBOARDING_INTERVIEW",
    "jobId": 17,
    "userUuid": "a9c7159b-cc3f-40c9-b999-060637c2389c",
    "bucket": "private-bucket",
    "audioObjectKeys": ["voice/sample-1.m4a"],
}
RESULT_QUEUE = "https://sqs.ap-northeast-2.amazonaws.com/447343943160/mirrorsoul-voice-training-result-queue"


class VoiceTrainingResultTest(unittest.TestCase):
    def setUp(self):
        _completed_results.clear()

    def test_request_requires_audio_keys_but_not_clone_id(self):
        request = {
            key: value for key, value in REQUEST.items() if key != "audioObjectKeys"
        }
        with self.assertRaises(VoiceTrainingWorkerError):
            _parse_message(json.dumps(request))

        request = {**REQUEST, "audioObjectKeys": []}
        with self.assertRaises(VoiceTrainingWorkerError):
            _parse_message(json.dumps(request))

        parsed = _parse_message(json.dumps(REQUEST))
        self.assertEqual(parsed.job_id, 17)

    def test_invalid_request_is_retained(self):
        self.assertFalse(
            _handle_sqs_message(
                SimpleNamespace(),
                Mock(),
                {"Body": json.dumps({"jobId": 17})},
                result_queue_url=RESULT_QUEUE,
            )
        )

    def test_result_message_contains_backend_contract(self):
        message = VoiceTrainingResultMessage(
            job_id=17,
            user_uuid=REQUEST["userUuid"],
            status="COMPLETED",
            attempt_number=2,
            result={"elevenlabsVoiceId": "voice-71", "voiceScore": 62.0},
        )
        client = Mock()
        client.send_message.return_value = {"MessageId": "result-17"}

        self.assertEqual(
            publish_voice_training_result(client, queue_url=RESULT_QUEUE, message=message),
            "result-17",
        )
        payload = json.loads(client.send_message.call_args.kwargs["MessageBody"])
        self.assertEqual(client.send_message.call_args.kwargs["QueueUrl"], RESULT_QUEUE)
        self.assertEqual(payload["eventType"], "VOICE_TRAINING_STATUS")
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["jobId"], 17)
        self.assertNotIn("cloneId", payload)
        self.assertNotIn("attemptNumber", payload)
        self.assertEqual(payload["result"]["elevenlabsVoiceId"], "voice-71")
        self.assertNotIn("error", payload)

    def test_completed_result_validates_score_and_optional_intro_audio(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            VoiceTrainingResultMessage(
                job_id=17,
                user_uuid=REQUEST["userUuid"],
                status="COMPLETED",
                attempt_number=1,
                result={"elevenlabsVoiceId": "voice-71", "voiceScore": 100.01},
            ).to_dict()

        with self.assertRaisesRegex(ValueError, "introAudio missing"):
            VoiceTrainingResultMessage(
                job_id=17,
                user_uuid=REQUEST["userUuid"],
                status="COMPLETED",
                attempt_number=1,
                result={
                    "elevenlabsVoiceId": "voice-71",
                    "voiceScore": 62.123,
                    "introAudio": {"bucket": "voice-bucket"},
                },
            ).to_dict()

        payload = VoiceTrainingResultMessage(
            job_id=17,
            user_uuid=REQUEST["userUuid"],
            status="COMPLETED",
            attempt_number=1,
            result={
                "elevenlabsVoiceId": "voice-71",
                "voiceScore": 62.123,
                "introAudio": {
                    "bucket": "voice-bucket",
                    "objectKey": "voice-intros/user/job-17.mp3",
                    "contentType": "audio/mpeg",
                    "sizeBytes": 123456,
                    "durationMs": 12000,
                },
            },
        ).to_dict()
        self.assertEqual(payload["result"]["voiceScore"], 62.12)

    def test_success_publishes_processing_then_completed(self):
        sqs = Mock()
        sqs.send_message.return_value = {"MessageId": "sent"}
        audio = DownloadedAudio("sample.wav", b"normalized", "audio/wav")
        with (
            patch(
                "model_calling.voice_training.worker._download_audio",
                return_value=audio,
            ),
            patch(
                "model_calling.voice_training.worker._prepare_voice_training_audio",
                return_value=[audio],
            ),
            patch(
                "model_calling.voice_training.worker.clone_user_voice_from_files",
                new=AsyncMock(return_value="voice-71"),
            ),
            patch(
                "model_calling.voice_training.worker._evaluate_actual_voice_similarity",
                return_value=88.25,
            ),
        ):
            should_delete = _handle_sqs_message(
                SimpleNamespace(),
                sqs,
                {"Body": json.dumps(REQUEST), "Attributes": {"ApproximateReceiveCount": "1"}},
                result_queue_url=RESULT_QUEUE,
            )

        self.assertTrue(should_delete)
        payloads = [json.loads(call.kwargs["MessageBody"]) for call in sqs.send_message.call_args_list]
        self.assertEqual([payload["status"] for payload in payloads], ["PROCESSING", "COMPLETED"])
        self.assertEqual(payloads[-1]["result"]["voiceScore"], 88.25)
        self.assertEqual(
            set(payloads[-1]["result"]),
            {"elevenlabsVoiceId", "voiceScore"},
        )

    def test_failure_publishes_failed_result(self):
        sqs = Mock()
        sqs.send_message.return_value = {"MessageId": "sent"}
        with patch(
            "model_calling.voice_training.worker._download_audio",
            side_effect=VoiceTrainingWorkerError(
                "voice_input_quality_failed: insufficient valid samples"
            ),
        ):
            should_delete = _handle_sqs_message(
                SimpleNamespace(),
                sqs,
                {"Body": json.dumps(REQUEST)},
                result_queue_url=RESULT_QUEUE,
            )

        self.assertTrue(should_delete)
        payload = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
        self.assertEqual(payload["status"], "FAILED")
        self.assertFalse(payload["error"]["retryable"])

    def test_retryable_failure_retains_request_until_final_attempt(self):
        for attempt, expected_delete, expected_retryable in (
            (1, False, True),
            (3, True, False),
        ):
            with self.subTest(attempt=attempt):
                sqs = Mock()
                sqs.send_message.return_value = {"MessageId": "sent"}
                with patch(
                    "model_calling.voice_training.worker._download_audio",
                    side_effect=RuntimeError("temporary S3 failure"),
                ):
                    should_delete = _handle_sqs_message(
                        SimpleNamespace(),
                        sqs,
                        {
                            "Body": json.dumps(REQUEST),
                            "Attributes": {
                                "ApproximateReceiveCount": str(attempt)
                            },
                        },
                        result_queue_url=RESULT_QUEUE,
                    )
                payload = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
                self.assertEqual(should_delete, expected_delete)
                self.assertEqual(payload["error"]["retryable"], expected_retryable)

    def test_failed_result_delivery_retains_request(self):
        sqs = Mock()
        sqs.send_message.side_effect = [
            {"MessageId": "processing"},
            RuntimeError("SQS unavailable"),
        ]
        with patch(
            "model_calling.voice_training.worker._download_audio",
            side_effect=RuntimeError("temporary S3 failure"),
        ):
            should_delete = _handle_sqs_message(
                SimpleNamespace(),
                sqs,
                {"Body": json.dumps(REQUEST)},
                result_queue_url=RESULT_QUEUE,
            )
        self.assertFalse(should_delete)

    def test_redelivery_reuses_result_by_job_id(self):
        sqs = Mock()
        sqs.send_message.return_value = {"MessageId": "sent"}
        result = {"elevenlabsVoiceId": "voice-71", "voiceScore": 62.0}
        with patch(
            "model_calling.voice_training.worker._process_voice_training_message",
            return_value=result,
        ) as process:
            for _ in range(2):
                self.assertTrue(
                    _handle_sqs_message(
                        SimpleNamespace(),
                        sqs,
                        {"Body": json.dumps(REQUEST)},
                        result_queue_url=RESULT_QUEUE,
                    )
                )
        process.assert_called_once()

    def test_completed_result_delivery_failure_retains_request(self):
        sqs = Mock()
        sqs.send_message.side_effect = [
            {"MessageId": "processing"},
            RuntimeError("SQS unavailable"),
        ]
        with patch(
            "model_calling.voice_training.worker._process_voice_training_message",
            return_value={"elevenlabsVoiceId": "voice-71", "voiceScore": 62.0},
        ):
            should_delete = _handle_sqs_message(
                SimpleNamespace(),
                sqs,
                {"Body": json.dumps(REQUEST)},
                result_queue_url=RESULT_QUEUE,
            )
        self.assertFalse(should_delete)

    def test_fallback_score_matches_existing_sample_proxy(self):
        self.assertEqual(_fallback_voice_score(5), 62.0)
        self.assertEqual(_fallback_voice_score(20), 95.0)

    def test_worker_requires_result_queue_before_receiving(self):
        with patch.dict(
            os.environ,
            {"AWS_SQS_VOICE_TRAINING_QUEUE_URL": "request"},
        ):
            with patch.dict(os.environ, {"AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL": ""}):
                with self.assertRaisesRegex(
                    VoiceTrainingWorkerError, "AWS_SQS_VOICE_TRAINING_RESULT_QUEUE_URL"
                ):
                    run_worker(once=True)


if __name__ == "__main__":
    unittest.main()
