"""End-to-end checks for the unified RAG store on a real ChromaDB collection."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-only")

import chromadb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from model_training import services  # noqa: E402
from model_training.rag_documents import (  # noqa: E402
    INTERVIEW_MEMORY,
    PROFILE_SNAPSHOT,
    build_document_id,
)
from tools import migrate_rag_v2  # noqa: E402

TOPICS = ("카페", "갈등", "여행", "음악")


def fake_embedding(text: str) -> list[float]:
    """Orthogonal unit vectors per topic so distances are predictable."""
    vector = [0.0] * (len(TOPICS) + 1)
    for index, topic in enumerate(TOPICS):
        if topic in text:
            vector[index] = 1.0
            return vector
    vector[-1] = 1.0
    return vector


def fake_embeddings(texts):
    return [fake_embedding(text) for text in texts]


SAMPLE_CAFE = {
    "questionId": 1,
    "questionCategory": "취미",
    "questionText": "쉬는 날에는 무엇을 하나요?",
    "transcript": "친구들과 새로운 카페를 찾아다녀요.",
}
SAMPLE_CONFLICT = {
    "questionId": 2,
    "questionCategory": "관계",
    "questionText": "갈등이 생기면 어떻게 하나요?",
    "transcript": "갈등이 생기면 먼저 상대방 이야기를 들어요.",
}
SAMPLE_TRAVEL = {
    "questionId": 3,
    "questionCategory": "취향",
    "questionText": "가고 싶은 곳은?",
    "transcript": "혼자 여행 가는 걸 좋아해요.",
}


class RagUnificationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        client = chromadb.PersistentClient(path=self._tmp.name)
        self.collection = client.get_or_create_collection(f"rag-{id(self)}")
        self._patches = [
            patch.object(services, "collection", self.collection),
            patch.object(services, "create_embeddings", side_effect=fake_embeddings),
            patch.object(services, "create_embedding", side_effect=fake_embedding),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self._patches):
            item.stop()
        self._tmp.cleanup()

    # helpers -----------------------------------------------------------
    def save_profile(self, user_id="member-a", samples=None, **kwargs):
        params = {
            "user_id": user_id,
            "ai_profile_id": "clone-16",
            "clone_id": 16,
            "mbti": "ENFP",
            "interview_samples": [SAMPLE_CAFE, SAMPLE_CONFLICT]
            if samples is None
            else samples,
        }
        params.update(kwargs)
        return services.add_member_profile_to_rag(**params)

    def save_sample(self, sample, user_id="member-a", ai_profile_id="clone-16"):
        return services.add_training_sample_to_rag(
            user_id=user_id,
            ai_profile_id=ai_profile_id,
            question_id=sample.get("questionId"),
            question_category=sample.get("questionCategory"),
            question_text=sample.get("questionText"),
            transcript=sample["transcript"],
            mbti="ENFP",
            description="자기소개",
        )

    def stored(self, user_id="member-a"):
        result = self.collection.get(
            where={"userId": user_id}, include=["documents", "metadatas"]
        )
        return dict(zip(result["ids"], zip(result["documents"], result["metadatas"])))


class StorageTests(RagUnificationTestCase):
    def test_saving_same_profile_twice_does_not_grow_collection(self):
        self.save_profile()
        first = self.stored()
        self.save_profile()
        second = self.stored()

        self.assertEqual(len(first), 3)
        self.assertEqual(set(first), set(second))
        self.assertIn(build_document_id("member-a", PROFILE_SNAPSHOT, "clone-16"), first)

    def test_same_interview_via_profiles_and_samples_is_stored_once(self):
        self.save_sample(SAMPLE_CAFE)
        self.save_profile()
        self.save_sample(SAMPLE_CAFE)

        interview_ids = [
            doc_id
            for doc_id, (_, metadata) in self.stored().items()
            if metadata["sourceType"] == INTERVIEW_MEMORY
        ]
        self.assertEqual(
            sorted(interview_ids),
            [
                "member-a:interview_memory:question-1",
                "member-a:interview_memory:question-2",
            ],
        )
        # /samples after /profiles keeps the profile sync ownership.
        _, metadata = self.stored()["member-a:interview_memory:question-1"]
        self.assertTrue(metadata["profileManaged"])
        self.assertEqual(metadata["cloneId"], 16)

    def test_samples_retry_is_idempotent(self):
        first = self.save_sample(SAMPLE_TRAVEL)
        second = self.save_sample(SAMPLE_TRAVEL)
        self.assertEqual(first["documentId"], second["documentId"])
        self.assertEqual(first["sampleId"], "question-3")
        self.assertEqual(len(self.stored()), 1)

    def test_interview_order_change_keeps_document_ids(self):
        self.save_profile(samples=[SAMPLE_CAFE, SAMPLE_CONFLICT, SAMPLE_TRAVEL])
        first = set(self.stored())
        self.save_profile(samples=[SAMPLE_TRAVEL, SAMPLE_CAFE, SAMPLE_CONFLICT])
        self.assertEqual(first, set(self.stored()))

    def test_legacy_request_without_question_id_gets_stable_id(self):
        legacy = {"questionText": "좋아하는 음악은?", "transcript": "재즈 음악을 좋아해요."}
        self.save_profile(samples=[legacy, SAMPLE_CAFE])
        first = set(self.stored())
        self.save_profile(samples=[SAMPLE_CAFE, dict(legacy)])
        self.assertEqual(first, set(self.stored()))
        self.assertEqual(len(first), 3)

    def test_stale_cleanup_only_touches_same_profile_documents(self):
        # Another member, an independent /samples memory, another profile key
        # and a future memory type must all survive the profile update.
        self.save_profile(user_id="member-b")
        self.save_sample(SAMPLE_TRAVEL)
        self.save_profile(ai_profile_id="clone-99", samples=[SAMPLE_TRAVEL])
        services.upsert_rag_documents(
            [
                services.RagDocument(
                    user_id="member-a",
                    source_type="conversation_memory",
                    source_id="call-1-turn-3",
                    text="통화 중 음악 이야기를 했다.",
                    profile_key="clone-16",
                )
            ]
        )
        self.save_profile()  # questions 1, 2
        before_b = set(self.stored("member-b"))

        self.save_profile(samples=[SAMPLE_CAFE])  # question 2 removed

        ids = set(self.stored())
        self.assertNotIn("member-a:interview_memory:question-2", ids)
        self.assertIn("member-a:interview_memory:question-1", ids)
        self.assertIn("member-a:conversation_memory:call-1-turn-3", ids)
        self.assertIn("member-a:profile_snapshot:clone-99", ids)
        self.assertEqual(before_b, set(self.stored("member-b")))
        # question-3 belongs to clone-99's profile sync now (shared question
        # id), it was not part of clone-16's sync and must survive.
        self.assertIn("member-a:interview_memory:question-3", ids)

    def test_independent_sample_survives_profile_resync(self):
        self.save_sample(SAMPLE_TRAVEL)
        self.save_profile()
        self.save_profile(samples=[SAMPLE_CAFE])
        self.assertIn("member-a:interview_memory:question-3", self.stored())

    def test_profile_resync_replaces_legacy_profile_documents_only(self):
        self.collection.add(
            ids=["member_profile_member-a_clone-16", "legacy-other-key", "training_sample_x"],
            documents=["[회원 핵심 프로필] 옛 요약", "다른 프로필", "옛 개별 답변"],
            embeddings=[fake_embedding("x")] * 3,
            metadatas=[
                {"userId": "member-a", "sourceType": "member_profile_summary", "profileKey": "clone-16"},
                {"userId": "member-a", "sourceType": "member_profile_summary", "profileKey": "clone-1"},
                {"userId": "member-a", "sourceType": "interview_answer", "questionId": 7},
            ],
        )
        self.save_profile()
        ids = set(self.stored())
        self.assertNotIn("member_profile_member-a_clone-16", ids)
        self.assertIn("legacy-other-key", ids)
        self.assertIn("training_sample_x", ids)

    def test_metadata_uses_only_chroma_scalar_types(self):
        self.save_profile(interests=["카페"], values=["정직"], job="학생")
        self.save_sample(SAMPLE_TRAVEL)
        for _, metadata in self.stored().values():
            for key in (
                "userId", "sourceType", "sourceId", "profileKey",
                "schemaVersion", "updatedAt", "importance", "confidence",
            ):
                self.assertIn(key, metadata)
            for value in metadata.values():
                self.assertIsInstance(value, (str, int, float, bool))

    def test_profile_snapshot_contains_only_confirmed_fields(self):
        result = self.save_profile(
            name="홍길동",
            mbti="enfp",
            job="학생",
            interests=["카페"],
            values=["정직"],
            samples=[],
        )
        text = result["profileSummary"]
        self.assertIn("이름: 홍길동", text)
        self.assertIn("MBTI: ENFP", text)
        self.assertIn("직업: 학생", text)
        self.assertIn("- 정직", text)
        for missing in ("나이", "성별", "닉네임", "자기소개", "미입력"):
            self.assertNotIn(missing, text)


class SearchTests(RagUnificationTestCase):
    def test_search_isolated_per_member(self):
        self.save_profile(user_id="member-a")
        self.save_profile(user_id="member-b", samples=[SAMPLE_CAFE, SAMPLE_TRAVEL])
        memories = services.search_user_memories("member-a", "카페", top_k=10, max_distance=5)
        self.assertTrue(memories)
        for memory in memories:
            self.assertEqual(memory["metadata"]["userId"], "member-a")
            self.assertTrue(memory["documentId"].startswith("member-a:"))

    def test_profile_included_exactly_once_and_outside_top_k(self):
        self.save_profile(samples=[SAMPLE_CAFE, SAMPLE_CONFLICT, SAMPLE_TRAVEL])
        memories = services.search_user_memories("member-a", "카페", top_k=1, max_distance=0.5)
        self.assertEqual(
            [memory["documentId"] for memory in memories],
            [
                "member-a:profile_snapshot:clone-16",
                "member-a:interview_memory:question-1",
            ],
        )
        profiles = [m for m in memories if m["metadata"]["sourceType"] == PROFILE_SNAPSHOT]
        self.assertEqual(len(profiles), 1)

    def test_profile_included_even_without_relevant_memories(self):
        self.save_profile(samples=[SAMPLE_CAFE])
        memories = services.search_user_memories("member-a", "음악", top_k=5, max_distance=0.5)
        self.assertEqual(
            [memory["metadata"]["sourceType"] for memory in memories],
            [PROFILE_SNAPSHOT],
        )

    def test_far_documents_are_excluded(self):
        self.save_profile(samples=[SAMPLE_CAFE, SAMPLE_CONFLICT, SAMPLE_TRAVEL])
        memories = services.search_user_memories("member-a", "갈등", top_k=5, max_distance=0.75)
        memory_ids = [m["documentId"] for m in memories[1:]]
        self.assertEqual(memory_ids, ["member-a:interview_memory:question-2"])

    def test_legacy_documents_remain_searchable_and_deduplicated(self):
        self.collection.add(
            ids=["member_profile_member-a_default", "member_profile_interview_x", "training_sample_y"],
            documents=[
                "[회원 핵심 프로필] 옛 요약",
                "[인터뷰 질문] 카페 질문 옛 문서",
                "[사용자 답변] 여행 옛 개별 답변",
            ],
            embeddings=[
                fake_embedding("x"),
                fake_embedding("카페"),
                fake_embedding("여행"),
            ],
            metadatas=[
                {"userId": "member-a", "sourceType": "member_profile_summary", "profileKey": "default", "mbti": "INFP"},
                {"userId": "member-a", "sourceType": "member_profile_interview", "profileKey": "default", "questionId": 1},
                {"userId": "member-a", "sourceType": "interview_answer", "questionId": 3},
            ],
        )
        legacy_only = services.search_user_memories("member-a", "여행", top_k=5, max_distance=0.5)
        self.assertEqual(
            [m["documentId"] for m in legacy_only],
            ["member_profile_member-a_default", "training_sample_y"],
        )

        # A v2 sync for another profile key adds v2 copies of question 1.
        self.save_profile(samples=[SAMPLE_CAFE])
        memories = services.search_user_memories("member-a", "카페", top_k=5, max_distance=0.5)
        self.assertEqual(
            [m["documentId"] for m in memories],
            ["member-a:profile_snapshot:clone-16", "member-a:interview_memory:question-1"],
        )

    def test_search_handles_empty_collection_and_profile_only_member(self):
        self.assertEqual(
            services.search_user_memories("nobody", "카페", top_k=5, max_distance=5),
            [],
        )
        self.save_profile(samples=[])
        memories = services.search_user_memories("member-a", "카페", top_k=5, max_distance=5)
        self.assertEqual(
            [m["documentId"] for m in memories],
            ["member-a:profile_snapshot:clone-16"],
        )
        self.assertEqual(
            services.search_user_memories("nobody", "카페", top_k=5, max_distance=5),
            [],
        )

    def test_prompt_groups_v2_documents(self):
        from model_calling.services import format_retrieved_memories

        self.save_profile(samples=[SAMPLE_CAFE])
        memories = services.search_user_memories("member-a", "카페", top_k=5, max_distance=0.5)
        prompt = format_retrieved_memories(memories, max_chars=3000)
        self.assertIn("[확인된 회원 핵심 프로필]", prompt)
        self.assertIn("[회원이 직접 답한 인터뷰]", prompt)
        self.assertEqual(prompt.count("[회원 핵심 프로필]"), 1)
        short = format_retrieved_memories(memories, max_chars=80)
        self.assertLessEqual(len(short), 80)


class ProfileRouterCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi import FastAPI

        from model_training.routers import training

        self.training = training
        app = FastAPI()
        app.include_router(training.router)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.payload = {"userId": "member-a", "cloneId": 16, "interviewSamples": []}

    def test_callback_not_called_when_storage_fails(self):
        with (
            patch.object(self.training, "add_member_profile_to_rag", side_effect=RuntimeError("chroma down")),
            patch.object(self.training, "notify_personality_training_complete") as notify,
        ):
            response = self.client.post("/api/v1/training/profiles", json=self.payload)
        self.assertEqual(response.status_code, 500)
        notify.assert_not_called()

    def test_callback_called_after_storage_succeeds(self):
        manager = MagicMock()
        manager.store.return_value = {
            "documentId": "member-a:profile_snapshot:default",
            "status": "stored",
            "keywords": [],
            "profileSummary": "[회원 핵심 프로필]",
            "profileQuality": {"profileScore": 1.0},
        }
        manager.notify.return_value = True
        with (
            patch.object(self.training, "add_member_profile_to_rag", manager.store),
            patch.object(self.training, "notify_personality_training_complete", manager.notify),
        ):
            response = self.client.post("/api/v1/training/profiles", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([call[0] for call in manager.mock_calls if call[0] in {"store", "notify"}], ["store", "notify"])

    def test_backend_source_revision_is_passed_to_callback(self):
        stored = {
            "documentId": "member-a:profile_snapshot:default",
            "status": "stored",
            "keywords": [],
            "profileSummary": "[회원 핵심 프로필]",
            "profileQuality": {"profileScore": 70.0},
        }
        for payload, expected in (
            ({**self.payload, "sourceRevision": 4}, 4),
            (self.payload, None),
        ):
            with self.subTest(expected=expected):
                with (
                    patch.object(self.training, "add_member_profile_to_rag", return_value=stored),
                    patch.object(self.training, "notify_personality_training_complete", return_value=True) as notify,
                ):
                    response = self.client.post("/api/v1/training/profiles", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(notify.call_args.kwargs["source_revision"], expected)

    def test_invalid_source_revision_is_rejected(self):
        with patch.object(self.training, "add_member_profile_to_rag") as store:
            response = self.client.post(
                "/api/v1/training/profiles",
                json={**self.payload, "sourceRevision": 0},
            )
        self.assertEqual(response.status_code, 422)
        store.assert_not_called()

    def test_samples_endpoint_accepts_legacy_payload(self):
        with patch.object(
            self.training,
            "add_training_sample_to_rag",
            return_value={"documentId": "member-a:interview_memory:question-1", "sampleId": "question-1", "status": "stored"},
        ) as store:
            response = self.client.post(
                "/api/v1/training/samples",
                json={
                    "userId": "member-a",
                    "aiProfileId": "clone-16",
                    "questionId": 1,
                    "questionCategory": "취미",
                    "questionText": "q",
                    "transcript": "a",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(store.call_args.kwargs["source_id"])


class MigrationTests(RagUnificationTestCase):
    def _add_legacy(self):
        self.collection.add(
            ids=["member_profile_member-a_clone-16", "member_profile_interview_member-a_clone-16_001_1", "training_sample_a", "training_sample_b"],
            documents=["[회원 핵심 프로필] 요약", "카페 인터뷰", "카페 인터뷰 개별 저장", "여행 개별 답변"],
            embeddings=[fake_embedding("x"), fake_embedding("카페"), fake_embedding("카페"), fake_embedding("여행")],
            metadatas=[
                {"userId": "member-a", "sourceType": "member_profile_summary", "profileKey": "clone-16"},
                {"userId": "member-a", "sourceType": "member_profile_interview", "profileKey": "clone-16", "questionId": 1},
                {"userId": "member-a", "sourceType": "interview_answer", "aiProfileId": "clone-16", "questionId": 1, "sampleId": "sample_a"},
                {"userId": "member-a", "sourceType": "interview_answer", "aiProfileId": "clone-16", "questionId": 5},
            ],
        )

    def test_dry_run_writes_nothing(self):
        self._add_legacy()
        before = self.collection.count()
        result = migrate_rag_v2.run_migration(self.collection, apply=False)
        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["toCreate"], 3)
        self.assertEqual(result["duplicateLegacyDocuments"], 1)
        self.assertEqual(self.collection.count(), before)

    def test_apply_is_idempotent_and_keeps_legacy_by_default(self):
        self._add_legacy()
        first = migrate_rag_v2.run_migration(self.collection, apply=True)
        count = self.collection.count()
        second = migrate_rag_v2.run_migration(self.collection, apply=True)
        self.assertEqual(first["created"], 3)
        self.assertEqual(second["created"], 0)
        self.assertEqual(self.collection.count(), count)
        self.assertEqual(count, 7)
        self.assertIn("member-a:interview_memory:question-1", self.stored())
        self.assertIn("member-a:interview_memory:question-5", self.stored())
        _, metadata = self.stored()["member-a:interview_memory:question-1"]
        self.assertTrue(metadata["profileManaged"])

    def test_delete_legacy_requires_apply_and_keeps_duplicates(self):
        self._add_legacy()
        with self.assertRaises(ValueError):
            migrate_rag_v2.run_migration(self.collection, apply=False, delete_legacy=True)
        migrate_rag_v2.run_migration(self.collection, apply=True, delete_legacy=True)
        ids = set(self.stored())
        self.assertIn("training_sample_a", ids)  # duplicate kept
        self.assertNotIn("member_profile_member-a_clone-16", ids)
        self.assertEqual(self.collection.count(), 4)


if __name__ == "__main__":
    unittest.main()
