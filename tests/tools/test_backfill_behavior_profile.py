import io
import unittest
from contextlib import redirect_stdout

from model_training.behavior_profile import BehaviorTrait
from model_training.rag_documents import (
    BEHAVIOR_SECTION_HEADING,
    build_interview_memory_text,
    build_profile_snapshot_text,
)
from tools.backfill_behavior_profile import backfill_user, interview_sample_from_text


class _Collection:
    def __init__(self, profile_text, interview_texts):
        self.profile_text = profile_text
        self.interview_texts = interview_texts
        self.updates = []

    def get(self, *, where, include):
        types = where["$and"][1]["sourceType"]["$in"]
        if "profile_snapshot" in types:
            return {"ids": ["u:profile"], "documents": [self.profile_text], "metadatas": [{"userId": "u", "keywordCount": 1}]}
        return {"ids": [f"i{n}" for n, _ in enumerate(self.interview_texts)], "documents": self.interview_texts, "metadatas": [{}] * len(self.interview_texts)}

    def update(self, **kwargs):
        self.updates.append(kwargs)


INTERVIEW = build_interview_memory_text(
    question_category="스트레스",
    question_text="면접 전날 너무 떨리면 어떻게 할 건가요?",
    transcript="그냥 포기하고 도망가겠습니다.",
)


class BackfillTests(unittest.TestCase):
    def test_parses_stored_interview_text(self):
        sample = interview_sample_from_text(INTERVIEW)
        self.assertEqual(sample["questionText"], "면접 전날 너무 떨리면 어떻게 할 건가요?")
        self.assertEqual(sample["questionCategory"], "스트레스")
        self.assertEqual(sample["transcript"], "그냥 포기하고 도망가겠습니다.")

    def _run(self, apply):
        collection = _Collection(build_profile_snapshot_text(name="동빈", keywords=["개발"]), [INTERVIEW])
        seen = {}

        def extract(samples):
            seen["samples"] = samples
            return [BehaviorTrait("압박·스트레스", "포기하고 피한다", "도망가겠습니다")]

        with redirect_stdout(io.StringIO()):
            status = backfill_user(collection, "u", extract=extract, embed=lambda text: [0.1], apply=apply)
        return status, collection, seen

    def test_dry_run_writes_nothing(self):
        status, collection, seen = self._run(apply=False)
        self.assertEqual(status, "would_update")
        self.assertEqual(collection.updates, [])
        self.assertEqual(seen["samples"][0]["transcript"], "그냥 포기하고 도망가겠습니다.")

    def test_apply_updates_only_profile_document(self):
        status, collection, _ = self._run(apply=True)
        self.assertEqual(status, "updated")
        self.assertEqual(len(collection.updates), 1)
        update = collection.updates[0]
        self.assertEqual(update["ids"], ["u:profile"])
        self.assertIn(BEHAVIOR_SECTION_HEADING, update["documents"][0])
        self.assertEqual(update["metadatas"][0]["behaviorTraitCount"], 1)
        self.assertEqual(update["metadatas"][0]["keywordCount"], 1)

    def test_member_without_interviews_is_skipped(self):
        collection = _Collection("[회원 핵심 프로필]", [])
        status = backfill_user(collection, "u", extract=None, embed=None, apply=True)
        self.assertEqual(status, "no_interviews")


if __name__ == "__main__":
    unittest.main()
