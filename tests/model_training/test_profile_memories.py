import unittest

from model_training.profile_memories import (
    PROFILE_INTERVIEW_SOURCE_TYPE,
    build_member_profile_documents,
    find_stale_profile_interview_ids,
)
from model_training.rag_documents import (
    INTERVIEW_MEMORY,
    PROFILE_SNAPSHOT,
    RAG_SCHEMA_VERSION,
)


class ProfileMemoryDocumentTests(unittest.TestCase):
    def test_builds_profile_and_full_interview_memories(self) -> None:
        documents = build_member_profile_documents(
            user_id="member/uuid",
            ai_profile_id="clone-14",
            clone_id=14,
            age=29,
            gender="male",
            mbti="enfp",
            description="새로운 사람을 만나는 것을 좋아합니다.",
            keywords=["카페", "공감"],
            interview_samples=[
                {
                    "questionId": 10,
                    "questionCategory": "취미",
                    "questionText": "쉬는 날에는 무엇을 하나요?",
                    "transcript": "친구들과 새로운 카페를 찾아다녀요.",
                },
                {
                    "questionId": 11,
                    "questionCategory": "관계",
                    "questionText": "갈등이 생기면 어떻게 하나요?",
                    "transcript": "먼저 상대방의 이야기를 들어보려고 해요.",
                },
            ],
        )

        self.assertEqual(len(documents), 3)
        self.assertEqual(PROFILE_INTERVIEW_SOURCE_TYPE, INTERVIEW_MEMORY)
        snapshot = documents[0]
        self.assertEqual(snapshot.metadata["sourceType"], PROFILE_SNAPSHOT)
        self.assertEqual(snapshot.document_id, "member/uuid:profile_snapshot:clone-14")
        self.assertIn("MBTI: ENFP", snapshot.text)
        self.assertIn("자기소개: 새로운 사람을 만나는 것을 좋아합니다.", snapshot.text)

        interview = documents[1]
        self.assertEqual(interview.metadata["sourceType"], INTERVIEW_MEMORY)
        self.assertEqual(
            interview.document_id,
            "member/uuid:interview_memory:question-10",
        )
        self.assertIn("쉬는 날에는 무엇을 하나요?", interview.text)
        self.assertIn("친구들과 새로운 카페를 찾아다녀요.", interview.text)
        # Profile data lives in the snapshot instead of every interview.
        self.assertNotIn("새로운 사람을 만나는 것을 좋아합니다.", interview.text)
        self.assertEqual(interview.metadata["questionId"], 10)
        self.assertEqual(interview.metadata["cloneId"], 14)
        self.assertEqual(interview.metadata["schemaVersion"], RAG_SCHEMA_VERSION)
        self.assertTrue(interview.metadata["profileManaged"])

    def test_ignores_blank_interview_transcripts(self) -> None:
        documents = build_member_profile_documents(
            user_id="member",
            ai_profile_id=None,
            age=None,
            gender=None,
            mbti=None,
            description=None,
            keywords=[],
            interview_samples=[
                {"questionId": 1, "transcript": "   "},
                {"questionId": 2, "transcript": "기억할 답변"},
            ],
        )

        self.assertEqual(len(documents), 2)
        self.assertIn("기억할 답변", documents[1].text)

    def test_finds_only_stale_memories_for_same_profile(self) -> None:
        stale_ids = find_stale_profile_interview_ids(
            existing_ids=[
                "keep",
                "stale",
                "legacy-stale",
                "other-profile",
                "independent-sample",
                "legacy-sample",
                "other-user",
            ],
            existing_metadatas=[
                {
                    "userId": "member",
                    "sourceType": INTERVIEW_MEMORY,
                    "profileKey": "clone-14",
                    "profileManaged": True,
                },
                {
                    "userId": "member",
                    "sourceType": INTERVIEW_MEMORY,
                    "profileKey": "clone-14",
                    "profileManaged": True,
                },
                {
                    "userId": "member",
                    "sourceType": "member_profile_interview",
                    "profileKey": "clone-14",
                },
                {
                    "userId": "member",
                    "sourceType": INTERVIEW_MEMORY,
                    "profileKey": "clone-13",
                    "profileManaged": True,
                },
                {
                    "userId": "member",
                    "sourceType": INTERVIEW_MEMORY,
                    "profileKey": "clone-14",
                    "profileManaged": False,
                },
                {
                    "userId": "member",
                    "sourceType": "interview_answer",
                    "profileKey": "clone-14",
                },
                {
                    "userId": "someone-else",
                    "sourceType": INTERVIEW_MEMORY,
                    "profileKey": "clone-14",
                    "profileManaged": True,
                },
            ],
            user_id="member",
            ai_profile_id="clone-14",
            current_ids={"keep"},
        )

        self.assertEqual(stale_ids, ["stale", "legacy-stale"])


if __name__ == "__main__":
    unittest.main()
