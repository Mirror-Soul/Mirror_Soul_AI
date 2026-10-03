"""Copy legacy (schema v1) RAG documents to the unified schema v2 layout.

Safe by default:

* Without ``--apply`` the script only reports what it would do (dry run).
* Legacy documents are kept unless ``--delete-legacy`` is passed together
  with ``--apply``. Search already reads legacy documents, so keeping them is
  safe; v2 copies are preferred and duplicates are collapsed at read time.
* Existing embeddings and texts are reused, so no OpenAI call is made.
* Re-running is idempotent: documents whose v2 id already exists are skipped.

Usage::

    python -m tools.migrate_rag_v2 --dry-run
    python -m tools.migrate_rag_v2 --apply
    python -m tools.migrate_rag_v2 --apply --delete-legacy
    python -m tools.migrate_rag_v2 --dry-run --user-id <member-uuid>

Back up ``RAG_DB_PATH`` before running with ``--apply``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from model_training.rag_documents import (
    INTERVIEW_MEMORY,
    INTERVIEW_MEMORY_IMPORTANCE,
    LEGACY_INTERVIEW_ANSWER,
    LEGACY_PROFILE_INTERVIEW,
    LEGACY_PROFILE_SUMMARY,
    PROFILE_SNAPSHOT,
    PROFILE_SNAPSHOT_IMPORTANCE,
    RAG_SCHEMA_VERSION,
    USER_PROVIDED_CONFIDENCE,
    build_document_id,
    resolve_interview_source_id,
    resolve_profile_key,
    sanitize_metadata,
    utc_now_iso,
)

LEGACY_SOURCE_TYPES = (
    LEGACY_PROFILE_SUMMARY,
    LEGACY_PROFILE_INTERVIEW,
    LEGACY_INTERVIEW_ANSWER,
)
# Lower value wins when several legacy documents map to the same v2 id.
_LEGACY_PRIORITY = {
    LEGACY_PROFILE_SUMMARY: 0,
    LEGACY_PROFILE_INTERVIEW: 0,
    LEGACY_INTERVIEW_ANSWER: 1,
}
_BATCH_SIZE = 500


@dataclass
class PlannedDocument:
    legacy_id: str
    target_id: str
    text: str
    metadata: dict[str, Any]
    priority: int
    embedding: Any = None


@dataclass
class MigrationPlan:
    to_create: list[PlannedDocument] = field(default_factory=list)
    already_migrated: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    legacy_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "legacyDocuments": len(self.legacy_ids),
            "toCreate": len(self.to_create),
            "toCreateBySourceType": dict(
                Counter(item.metadata["sourceType"] for item in self.to_create)
            ),
            "alreadyMigrated": len(self.already_migrated),
            "duplicateLegacyDocuments": len(self.duplicates),
            "skipped": dict(self.skipped),
        }


def _target_for(
    document_id: str, text: str, metadata: Mapping[str, Any], now: str
) -> PlannedDocument | None:
    source_type = metadata.get("sourceType")
    user_id = str(metadata.get("userId") or "").strip()
    if not user_id:
        return None
    profile_key = resolve_profile_key(
        metadata.get("profileKey") or metadata.get("aiProfileId")
    )

    base = {
        key: value
        for key, value in metadata.items()
        if key not in {"sourceType", "sampleId", "interviewIndex"}
    }
    if source_type == LEGACY_PROFILE_SUMMARY:
        new_type = PROFILE_SNAPSHOT
        source_id = profile_key
        importance = PROFILE_SNAPSHOT_IMPORTANCE
        profile_managed = True
    else:
        new_type = INTERVIEW_MEMORY
        source_id = resolve_interview_source_id(
            question_id=metadata.get("questionId"),
            transcript=text,
        )
        importance = INTERVIEW_MEMORY_IMPORTANCE
        profile_managed = source_type == LEGACY_PROFILE_INTERVIEW

    base.update(
        {
            "userId": user_id,
            "sourceType": new_type,
            "sourceId": source_id,
            "profileKey": profile_key,
            "schemaVersion": RAG_SCHEMA_VERSION,
            "updatedAt": now,
            "importance": importance,
            "confidence": USER_PROVIDED_CONFIDENCE,
            "profileManaged": profile_managed,
            "migratedFrom": document_id,
            "legacySourceType": str(source_type),
        }
    )
    return PlannedDocument(
        legacy_id=document_id,
        target_id=build_document_id(user_id, new_type, source_id),
        text=text,
        metadata=sanitize_metadata(base),
        priority=_LEGACY_PRIORITY.get(str(source_type), 9),
    )


def plan_migration(
    records: Iterable[tuple[str, str, Mapping[str, Any] | None, Any]],
    existing_ids: set[str],
    *,
    user_id: str | None = None,
    now: str | None = None,
) -> MigrationPlan:
    """Build a migration plan from ``(id, text, metadata, embedding)`` rows."""
    timestamp = now or utc_now_iso()
    plan = MigrationPlan()
    chosen: dict[str, PlannedDocument] = {}
    for document_id, text, metadata, embedding in records:
        metadata = dict(metadata or {})
        if metadata.get("sourceType") not in LEGACY_SOURCE_TYPES:
            continue
        if metadata.get("schemaVersion") is not None:
            continue
        if user_id and metadata.get("userId") != user_id:
            continue
        plan.legacy_ids.append(document_id)
        if not str(text or "").strip():
            plan.skipped["emptyText"] += 1
            continue
        if embedding is None:
            plan.skipped["missingEmbedding"] += 1
            continue
        target = _target_for(document_id, str(text), metadata, timestamp)
        if target is None:
            plan.skipped["missingUserId"] += 1
            continue
        target.embedding = embedding
        if target.target_id in existing_ids:
            plan.already_migrated.append(document_id)
            continue
        current = chosen.get(target.target_id)
        if current is None or target.priority < current.priority:
            if current is not None:
                plan.duplicates.append(current.legacy_id)
            chosen[target.target_id] = target
        else:
            plan.duplicates.append(document_id)
    plan.to_create = list(chosen.values())
    return plan


def _iter_collection(collection, include: list[str]):
    offset = 0
    while True:
        batch = collection.get(limit=_BATCH_SIZE, offset=offset, include=include)
        ids = batch.get("ids") or []
        if not ids:
            return
        documents = batch.get("documents")
        metadatas = batch.get("metadatas")
        embeddings = batch.get("embeddings")
        for index, document_id in enumerate(ids):
            embedding = None if embeddings is None else embeddings[index]
            if embedding is not None and hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            yield (
                document_id,
                documents[index] if documents is not None else "",
                metadatas[index] if metadatas is not None else {},
                embedding,
            )
        offset += len(ids)


def run_migration(
    collection,
    *,
    apply: bool,
    delete_legacy: bool = False,
    user_id: str | None = None,
) -> dict[str, Any]:
    if delete_legacy and not apply:
        raise ValueError("--delete-legacy requires --apply")

    rows = list(
        _iter_collection(collection, ["documents", "metadatas", "embeddings"])
    )
    existing_ids = {row[0] for row in rows}
    plan = plan_migration(rows, existing_ids, user_id=user_id)
    result = {"mode": "apply" if apply else "dry-run", **plan.summary()}

    if apply and plan.to_create:
        for start in range(0, len(plan.to_create), _BATCH_SIZE):
            chunk = plan.to_create[start : start + _BATCH_SIZE]
            collection.upsert(
                ids=[item.target_id for item in chunk],
                documents=[item.text for item in chunk],
                embeddings=[item.embedding for item in chunk],
                metadatas=[item.metadata for item in chunk],
            )
    result["created"] = len(plan.to_create) if apply else 0

    deleted = 0
    if delete_legacy:
        # Duplicates that lost the priority check are kept: their text may
        # differ from the copied document, so they are only reported.
        covered = {item.legacy_id for item in plan.to_create}
        covered.update(plan.already_migrated)
        removable = [doc_id for doc_id in plan.legacy_ids if doc_id in covered]
        for start in range(0, len(removable), _BATCH_SIZE):
            collection.delete(ids=removable[start : start + _BATCH_SIZE])
        deleted = len(removable)
    result["deletedLegacy"] = deleted
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="write v2 documents")
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="with --apply, delete legacy documents that now have a v2 copy",
    )
    parser.add_argument("--user-id", help="limit the migration to one member")
    parser.add_argument("--db-path", help="defaults to RAG_DB_PATH")
    parser.add_argument("--collection", help="defaults to RAG_COLLECTION_NAME")
    args = parser.parse_args(argv)
    if args.delete_legacy and not args.apply:
        parser.error("--delete-legacy requires --apply")

    import chromadb

    from shared.config import settings

    client = chromadb.PersistentClient(path=args.db_path or settings.RAG_DB_PATH)
    collection = client.get_collection(args.collection or settings.RAG_COLLECTION_NAME)
    result = run_migration(
        collection,
        apply=args.apply,
        delete_legacy=args.delete_legacy,
        user_id=args.user_id,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
