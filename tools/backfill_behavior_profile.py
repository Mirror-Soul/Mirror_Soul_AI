"""Add the interview behavior profile to existing members' profile snapshots.

New RAG profile training writes the behavior profile automatically. Members
trained before that keep their old snapshot until the backend requests a new
training (interview added, personality updated). This script fills the gap
from the interview answers already stored in the RAG store.

Safe by default:

* Without ``--apply`` it only prints what it would write (dry run).
* Only the ``profile_snapshot`` document text, embedding and the
  ``behaviorTraitCount`` metadata are updated. Nothing is deleted.
* Re-running replaces the behavior section instead of appending to it.

Run on the AI API server (it owns the RAG store)::

    venv/bin/python -m tools.backfill_behavior_profile --user-id <member-uuid>
    venv/bin/python -m tools.backfill_behavior_profile --user-id <member-uuid> --apply
    venv/bin/python -m tools.backfill_behavior_profile --all --apply

Back up ``RAG_DB_PATH`` before running with ``--all --apply``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from model_training.rag_documents import (
    BEHAVIOR_SECTION_HEADING,
    INTERVIEW_SOURCE_TYPES,
    PROFILE_SNAPSHOT,
    replace_behavior_section,
)

_ANSWER_MARKER = "[사용자 답변]"


def interview_sample_from_text(text: str) -> dict[str, str]:
    """Split a stored interview memory text back into question and answer."""
    body = str(text or "")
    if _ANSWER_MARKER not in body:
        return {"transcript": body.strip()}
    head, answer = body.split(_ANSWER_MARKER, 1)
    question = ""
    category = ""
    for line in head.splitlines():
        line = line.strip()
        if line.startswith("질문:"):
            question = line[len("질문:"):].strip()
        elif line.startswith("카테고리:"):
            category = line[len("카테고리:"):].strip()
    return {
        "questionText": question,
        "questionCategory": category,
        "transcript": answer.strip(),
    }


def _get(collection: Any, user_id: str, source_types: list[str]) -> dict[str, Any]:
    return collection.get(
        where={"$and": [{"userId": user_id}, {"sourceType": {"$in": source_types}}]},
        include=["documents", "metadatas"],
    )


def profile_user_ids(collection: Any) -> list[str]:
    stored = collection.get(where={"sourceType": PROFILE_SNAPSHOT}, include=["metadatas"])
    return sorted(
        {
            str(metadata.get("userId"))
            for metadata in stored.get("metadatas") or []
            if metadata and metadata.get("userId")
        }
    )


def backfill_user(
    collection: Any,
    user_id: str,
    *,
    extract: Any,
    embed: Any,
    apply: bool,
) -> str:
    profiles = _get(collection, user_id, [PROFILE_SNAPSHOT])
    profile_ids = profiles.get("ids") or []
    if not profile_ids:
        return "no_profile"
    interviews = _get(collection, user_id, sorted(INTERVIEW_SOURCE_TYPES))
    samples = [
        interview_sample_from_text(text)
        for text in interviews.get("documents") or []
        if str(text or "").strip()
    ]
    if not samples:
        return "no_interviews"

    traits = extract(samples)
    lines = [trait.to_line() for trait in traits]
    print(f"\n== {user_id}: interviews={len(samples)} traits={len(lines)}")
    for line in lines:
        print(f"   - {line}")
    if not lines:
        return "empty"

    updated = 0
    for index, document_id in enumerate(profile_ids):
        old_text = (profiles.get("documents") or [""])[index] or ""
        new_text = replace_behavior_section(old_text, lines)
        if new_text == old_text:
            continue
        metadata = dict((profiles.get("metadatas") or [{}])[index] or {})
        metadata["behaviorTraitCount"] = len(lines)
        if apply:
            collection.update(
                ids=[document_id],
                documents=[new_text],
                embeddings=[embed(new_text)],
                metadatas=[metadata],
            )
        updated += 1
    if not updated:
        return "unchanged"
    return "updated" if apply else "would_update"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user-id", action="append", help="member uuid (repeatable)")
    target.add_argument("--all", action="store_true", help="every member with a profile")
    parser.add_argument("--apply", action="store_true", help="write changes")
    args = parser.parse_args(argv)

    # Imported here so --help works without opening the store.
    from model_training.behavior_profile import extract_behavior_traits
    from model_training.services import collection, create_embedding, openai_client

    user_ids = profile_user_ids(collection) if args.all else args.user_id
    print(
        f"behavior profile backfill: users={len(user_ids)} "
        f"mode={'APPLY' if args.apply else 'DRY-RUN'} section={BEHAVIOR_SECTION_HEADING}"
    )
    counts: dict[str, int] = {}
    for user_id in user_ids:
        try:
            status = backfill_user(
                collection,
                user_id,
                extract=lambda samples: extract_behavior_traits(samples, client=openai_client),
                embed=create_embedding,
                apply=args.apply,
            )
        except Exception as exc:  # noqa: BLE001 - continue with other members
            status = "error"
            print(f"\n== {user_id}: error {type(exc).__name__}: {exc}")
        counts[status] = counts.get(status, 0) + 1
    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not args.apply:
        print("dry run only. add --apply to write.")
    return 0 if "error" not in counts else 1


if __name__ == "__main__":
    sys.exit(main())
