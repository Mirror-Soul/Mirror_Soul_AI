import unittest

from model_training.profile_memories import (
    PROFILE_INTERVIEW_SOURCE_TYPE,
    build_member_profile_documents,
    find_stale_profile_interview_ids,
)


class ProfileMemoryDocumentTests(unittest.TestCase):
    def test_builds_profile_and_full_interview_memories(self) -> None:
        documents = build_member_profile_documents(
            user_id="member/uuid",
            ai_profile_id="clone-14",
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
        self.assertEqual(
            documents[0].metadata["sourceType"],
            "member_profile_summary",
        )
        self.assertEqual(
            documents[1].metadata["sourceType"],
            PROFILE_INTERVIEW_SOURCE_TYPE,
        )
        self.assertIn("쉬는 날에는 무엇을 하나요?", documents[1].text)
        self.assertIn("친구들과 새로운 카페를 찾아다녀요.", documents[1].text)
        self.assertIn("새로운 사람을 만나는 것을 좋아합니다.", documents[1].text)
        self.assertEqual(documents[1].metadata["questionId"], 10)

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
            existing_ids=["keep", "stale", "other-profile", "other-source"],
            existing_metadatas=[
                {
                    "userId": "member",
                    "sourceType": PROFILE_INTERVIEW_SOURCE_TYPE,
                    "profileKey": "clone-14",
                },
                {
                    "userId": "member",
                    "sourceType": PROFILE_INTERVIEW_SOURCE_TYPE,
                    "profileKey": "clone-14",
                },
                {
                    "userId": "member",
                    "sourceType": PROFILE_INTERVIEW_SOURCE_TYPE,
                    "profileKey": "clone-13",
                },
                {
                    "userId": "member",
                    "sourceType": "interview_answer",
                    "profileKey": "clone-14",
                },
            ],
            user_id="member",
            ai_profile_id="clone-14",
            current_ids={"keep"},
        )

        self.assertEqual(stale_ids, ["stale"])


if __name__ == "__main__":
    unittest.main()
