import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chromadb

from model_training import services


class _FakeCollection:
    def __init__(self) -> None:
        self.upsert_call = None
        self.deleted_ids = []

    def upsert(self, **kwargs) -> None:
        self.upsert_call = kwargs

    def get(self, **kwargs):
        current_ids = self.upsert_call["ids"][1:]
        current_metadatas = self.upsert_call["metadatas"][1:]
        return {
            "ids": [*current_ids, "stale-interview", "independent-sample"],
            "metadatas": [
                *current_metadatas,
                {
                    "userId": "member-uuid",
                    "sourceType": "member_profile_interview",
                    "profileKey": "clone-14",
                },
                {
                    "userId": "member-uuid",
                    "sourceType": "interview_answer",
                    "profileKey": "clone-14",
                },
            ],
        }

    def delete(self, *, ids) -> None:
        self.deleted_ids.extend(ids)


class _FakeSearchCollection:
    def __init__(self, *, with_summary: bool = True) -> None:
        self.with_summary = with_summary
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["near-memory", "far-memory", "unknown-distance"]],
            "documents": [["가까운 기억", "관련 없는 기억", "거리 없는 기억"]],
            "metadatas": [[
                {"userId": "member-uuid", "sourceType": "member_profile_interview", "questionId": 1},
                {"userId": "member-uuid", "sourceType": "member_profile_interview", "questionId": 2},
                {"userId": "member-uuid", "sourceType": "conversation_memory"},
            ]],
            "distances": [[0.31, 1.12, None]],
        }

    def get(self, **kwargs):
        if not self.with_summary:
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": ["profile-summary"],
            "documents": ["회원 핵심 프로필"],
            "metadatas": [
                {"userId": "member-uuid", "sourceType": "member_profile_summary"}
            ],
        }


class ProfileServiceTests(unittest.TestCase):
    def test_profile_update_batches_embeddings_and_replaces_stale_memories(self):
        collection = _FakeCollection()

        with (
            patch.object(services, "collection", collection),
                patch.object(
                    services,
                    "create_embeddings",
                    side_effect=lambda texts: [
                        [float(index)] for index, _ in enumerate(texts)
                    ],
            ) as create_embeddings,
        ):
            result = services.add_member_profile_to_rag(
                user_id="member-uuid",
                ai_profile_id="clone-14",
                age=29,
                gender="male",
                mbti="ENFP",
                description="새로운 사람을 만나는 것을 좋아합니다.",
                interests=[],
                interview_topics=[],
                interview_samples=[
                    {
                        "questionId": 1,
                        "questionCategory": "취미",
                        "questionText": "쉬는 날에는 무엇을 하나요?",
                        "transcript": "친구들과 새로운 카페를 찾아다녀요.",
                    },
                    {
                        "questionId": 2,
                        "questionCategory": "관계",
                        "questionText": "갈등이 생기면 어떻게 하나요?",
                        "transcript": "먼저 상대방 이야기를 들어보려고 해요.",
                    },
                ],
                keyword_limit=12,
            )

        self.assertEqual(len(collection.upsert_call["ids"]), 3)
        self.assertEqual(len(collection.upsert_call["embeddings"]), 3)
        create_embeddings.assert_called_once()
        self.assertEqual(collection.deleted_ids, ["stale-interview"])
        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["documentId"], collection.upsert_call["ids"][0])

    def test_profile_update_removes_stale_chroma_documents_only(self):
        with tempfile.TemporaryDirectory() as directory:
            client = chromadb.PersistentClient(path=directory)
            collection = client.get_or_create_collection("profile-test")
            collection.upsert(
                ids=["independent-sample"],
                documents=["별도 저장된 인터뷰 답변"],
                embeddings=[[0.0, 1.0]],
                metadatas=[
                    {
                        "userId": "member-uuid",
                        "sourceType": "interview_answer",
                        "profileKey": "clone-14",
                    }
                ],
            )

            with (
                patch.object(services, "collection", collection),
                patch.object(
                    services,
                    "create_embeddings",
                    side_effect=lambda texts: [
                        [1.0, float(index)] for index, _ in enumerate(texts)
                    ],
                ),
            ):
                services.add_member_profile_to_rag(
                    user_id="member-uuid",
                    ai_profile_id="clone-14",
                    mbti="ENFP",
                    interview_samples=[
                        {"questionId": 1, "transcript": "첫 번째 답변"},
                        {"questionId": 2, "transcript": "두 번째 답변"},
                    ],
                )
                services.add_member_profile_to_rag(
                    user_id="member-uuid",
                    ai_profile_id="clone-14",
                    mbti="ENFP",
                    interview_samples=[
                        {"questionId": 1, "transcript": "수정된 첫 번째 답변"},
                    ],
                )

            stored = collection.get(
                where={"userId": "member-uuid"},
                include=["documents", "metadatas"],
            )
            self.assertEqual(len(stored["ids"]), 3)
            self.assertIn("independent-sample", stored["ids"])
            interview_documents = [
                text
                for text, metadata in zip(stored["documents"], stored["metadatas"])
                if metadata["sourceType"] == "interview_memory"
            ]
            self.assertEqual(len(interview_documents), 1)
            self.assertIn("수정된 첫 번째 답변", interview_documents[0])

    def test_create_embeddings_restores_api_index_order(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[1.0]),
            ]
        )

        with patch.object(services, "openai_client", client):
            embeddings = services.create_embeddings(["first", "second"])

        self.assertEqual(embeddings, [[1.0], [2.0]])
        client.embeddings.create.assert_called_once_with(
            model=services.settings.EMBEDDING_MODEL,
            input=["first", "second"],
            encoding_format="float",
        )

    def test_memory_search_filters_results_above_distance_limit(self):
        collection = _FakeSearchCollection()
        with (
            patch.object(services, "collection", collection),
            patch.object(services, "create_embedding", return_value=[1.0, 0.0]),
        ):
            memories = services.search_user_memories(
                "member-uuid",
                "요즘 쉬는 날에는 뭐 해?",
                top_k=5,
                max_distance=0.75,
            )

        self.assertEqual(
            [memory["documentId"] for memory in memories],
            ["profile-summary", "near-memory", "unknown-distance"],
        )
        self.assertIsNone(memories[0]["distance"])
        self.assertEqual(memories[1]["distance"], 0.31)
        where = collection.query_kwargs["where"]["$and"]
        self.assertIn({"userId": "member-uuid"}, where)

    def test_memory_search_always_includes_stored_profile_summary(self):
        with (
            patch.object(services, "collection", _FakeSearchCollection()),
            patch.object(services, "create_embedding", return_value=[1.0, 0.0]),
        ):
            memories = services.search_user_memories(
                "member-uuid",
                "안녕",
                top_k=1,
                max_distance=0.75,
            )

        self.assertEqual(
            [memory["documentId"] for memory in memories],
            ["profile-summary", "near-memory"],
        )

    def test_memory_search_without_profile_returns_memories_only(self):
        with (
            patch.object(
                services, "collection", _FakeSearchCollection(with_summary=False)
            ),
            patch.object(services, "create_embedding", return_value=[1.0, 0.0]),
        ):
            memories = services.search_user_memories(
                "member-uuid", "안녕", top_k=5, max_distance=0.75
            )

        self.assertEqual(
            [memory["documentId"] for memory in memories],
            ["near-memory", "unknown-distance"],
        )


if __name__ == "__main__":
    unittest.main()
