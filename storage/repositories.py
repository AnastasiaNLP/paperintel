from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Sequence, get_args

from sqlalchemy import delete, exists, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from agents.agent_run_recorder import AgentRunPersistence
from api.in_memory_session_store import SessionNotFoundError
from api.session_store import SessionStore
from models.agent_runs import AgentRun
from models.artifacts import ComparisonArtifact, PaperWorkspace
from models.blob_artifacts import (
    BlobArtifact,
    BlobReference,
    BlobReferenceKind,
    BlobRetentionPolicy,
)
from models.blob_storage import StoredBlobObject
from models.discovery import CandidateStatus, SearchCandidate
from models.external_metadata import ArxivMetadataCacheEntry
from models.errors import StructuredError
from models.jobs import JobKind, JobStatus, WorkflowJob, pdf_blob_idempotency_key
from models.pdf_upload_errors import (
    PdfUploadExpiredError,
    PdfUploadNotFoundError,
    PdfUploadStateError,
)
from models.pdf_uploads import PdfUpload, PdfUploadStatus
from models.registered_pdf_errors import RegisteredPdfBlobNotAuthorizedError
from models.retrieval import PaperChunk, UpsertChunksResult
from models.session import Persona, Session, SessionPhase, Turn, TurnRole
from storage.mappers import (
    agent_run_to_orm,
    arxiv_metadata_cache_entry_to_orm,
    blob_artifact_to_orm,
    blob_reference_to_orm,
    comparison_artifact_to_orm,
    orm_to_agent_run,
    orm_to_arxiv_metadata_cache_entry,
    orm_to_blob_artifact,
    orm_to_blob_reference,
    orm_to_comparison_artifact,
    orm_to_paper_chunk,
    orm_to_workflow_job,
    orm_to_paper_workspace,
    orm_to_pdf_upload,
    orm_to_session,
    orm_to_search_candidate,
    orm_to_structured_error,
    orm_to_turn,
    paper_chunk_to_orm,
    paper_workspace_to_orm,
    pdf_upload_to_orm,
    search_candidate_to_orm,
    session_to_orm,
    structured_error_to_orm,
    turn_to_orm,
    workflow_job_to_orm,
)
from storage.models import (
    AgentRunORM,
    ArxivMetadataCacheORM,
    BlobArtifactORM,
    BlobReferenceORM,
    ComparisonArtifactORM,
    PaperChunkORM,
    PaperWorkspaceORM,
    PdfUploadORM,
    SearchCandidateORM,
    SessionORM,
    StructuredErrorORM,
    TurnORM,
    WorkflowJobORM,
)


class PostgresArxivMetadataCacheRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def get(self, arxiv_id: str) -> ArxivMetadataCacheEntry | None:
        with self.session_factory() as db:
            orm = db.get(ArxivMetadataCacheORM, arxiv_id)
            return orm_to_arxiv_metadata_cache_entry(orm) if orm is not None else None

    def save(self, entry: ArxivMetadataCacheEntry) -> ArxivMetadataCacheEntry:
        with self.session_factory() as db:
            orm = db.merge(arxiv_metadata_cache_entry_to_orm(entry))
            db.commit()
            db.refresh(orm)
            return orm_to_arxiv_metadata_cache_entry(orm)

    def record_success(self, entry: ArxivMetadataCacheEntry) -> ArxivMetadataCacheEntry:
        with self.session_factory() as db:
            orm = db.get(ArxivMetadataCacheORM, entry.arxiv_id)
            now = _utc_now()
            if orm is None:
                orm = arxiv_metadata_cache_entry_to_orm(
                    entry.model_copy(
                        update={
                            "fetched_at": entry.fetched_at or now,
                            "last_error_json": None,
                            "error_count": 0,
                            "updated_at": now,
                        }
                    )
                )
                db.add(orm)
            else:
                orm.title = entry.title
                orm.authors_json = entry.authors
                orm.abstract = entry.abstract
                orm.published_date = entry.published_date
                orm.categories_json = entry.categories
                orm.source_url = entry.source_url
                orm.fetched_at = entry.fetched_at or now
                orm.last_error_json = None
                orm.error_count = 0
                orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_arxiv_metadata_cache_entry(orm)

    def record_error(
        self,
        arxiv_id: str,
        *,
        error_json: dict,
    ) -> ArxivMetadataCacheEntry:
        with self.session_factory() as db:
            orm = db.get(ArxivMetadataCacheORM, arxiv_id)
            now = _utc_now()
            if orm is None:
                orm = ArxivMetadataCacheORM(
                    arxiv_id=arxiv_id,
                    last_error_json=error_json,
                    error_count=1,
                    updated_at=now,
                )
                db.add(orm)
            else:
                orm.last_error_json = error_json
                orm.error_count += 1
                orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_arxiv_metadata_cache_entry(orm)


class BlobArtifactNotFoundError(ValueError):
    def __init__(self, blob_id: str) -> None:
        super().__init__(f"Blob artifact not found: {blob_id}")
        self.blob_id = blob_id


class PostgresBlobArtifactRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_artifact(
        self,
        stored: StoredBlobObject,
        *,
        retention_policy: BlobRetentionPolicy = "durable",
        expires_at: datetime | None = None,
    ) -> BlobArtifact:
        candidate = BlobArtifact(
            kind=stored.kind,
            object_key=stored.object_key,
            bucket_name=stored.bucket_name,
            content_hash=stored.content_hash,
            content_type=stored.content_type,
            size_bytes=stored.size_bytes,
            storage_backend=stored.storage_backend,
            retention_policy=retention_policy,
            expires_at=expires_at,
        )
        with self.session_factory() as db:
            db.execute(
                pg_insert(BlobArtifactORM)
                .values(
                    id=candidate.id,
                    kind=candidate.kind,
                    object_key=candidate.object_key,
                    bucket_name=candidate.bucket_name,
                    content_hash=candidate.content_hash,
                    content_type=candidate.content_type,
                    size_bytes=candidate.size_bytes,
                    storage_backend=candidate.storage_backend,
                    retention_policy=candidate.retention_policy,
                    status=candidate.status,
                    expires_at=candidate.expires_at,
                    last_accessed_at=candidate.last_accessed_at,
                    deleted_at=candidate.deleted_at,
                    cleanup_metadata_json=candidate.cleanup_metadata,
                    created_at=candidate.created_at,
                    updated_at=candidate.updated_at,
                )
                .on_conflict_do_nothing(
                    constraint="uq_blob_artifacts_kind_content_hash"
                )
            )
            orm = (
                db.execute(
                    select(BlobArtifactORM)
                    .where(BlobArtifactORM.kind == stored.kind)
                    .where(BlobArtifactORM.content_hash == stored.content_hash)
                )
                .scalars()
                .one()
            )
            artifact = BlobArtifact.model_validate(
                {
                    **orm_to_blob_artifact(orm).model_dump(),
                    "object_key": stored.object_key,
                    "bucket_name": stored.bucket_name,
                    "content_type": stored.content_type,
                    "size_bytes": stored.size_bytes,
                    "storage_backend": stored.storage_backend,
                    "retention_policy": retention_policy,
                    "status": "active",
                    "expires_at": expires_at,
                    "deleted_at": None,
                    "cleanup_metadata": {},
                    "updated_at": _utc_now(),
                }
            )
            orm.object_key = artifact.object_key
            orm.bucket_name = artifact.bucket_name
            orm.content_type = artifact.content_type
            orm.size_bytes = artifact.size_bytes
            orm.storage_backend = artifact.storage_backend
            orm.retention_policy = artifact.retention_policy
            orm.status = artifact.status
            orm.expires_at = artifact.expires_at
            orm.deleted_at = artifact.deleted_at
            orm.cleanup_metadata_json = artifact.cleanup_metadata
            orm.updated_at = artifact.updated_at
            db.commit()
            db.refresh(orm)
            return orm_to_blob_artifact(orm)

    def get_artifact(self, blob_id: str) -> BlobArtifact | None:
        with self.session_factory() as db:
            orm = db.get(BlobArtifactORM, blob_id)
            return (
                orm_to_blob_artifact(orm)
                if orm is not None and orm.status == "active"
                else None
            )

    def get_by_kind_and_hash(self, kind: str, content_hash: str) -> BlobArtifact | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(BlobArtifactORM)
                    .where(BlobArtifactORM.kind == kind)
                    .where(BlobArtifactORM.content_hash == content_hash)
                    .where(BlobArtifactORM.status == "active")
                )
                .scalars()
                .first()
            )
            return orm_to_blob_artifact(orm) if orm is not None else None

    def get_by_object_key(self, object_key: str) -> BlobArtifact | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(BlobArtifactORM).where(
                        BlobArtifactORM.object_key == object_key
                    ).where(BlobArtifactORM.status == "active")
                )
                .scalars()
                .first()
            )
            return orm_to_blob_artifact(orm) if orm is not None else None

    def add_reference(
        self,
        blob_id: str,
        *,
        ref_kind: BlobReferenceKind,
        ref_id: str,
        metadata: dict | None = None,
    ) -> BlobReference:
        reference = BlobReference(
            blob_id=blob_id,
            ref_kind=ref_kind,
            ref_id=ref_id,
            metadata=metadata or {},
        )
        with self.session_factory() as db:
            self._require_active_orm_for_update(db, blob_id)
            db.execute(
                pg_insert(BlobReferenceORM)
                .values(
                    id=reference.id,
                    blob_id=reference.blob_id,
                    ref_kind=reference.ref_kind,
                    ref_id=reference.ref_id,
                    metadata_json=reference.metadata,
                    created_at=reference.created_at,
                )
                .on_conflict_do_update(
                    constraint="uq_blob_references_blob_kind_ref",
                    set_={
                        "status": "active",
                        "released_at": None,
                    },
                )
            )
            orm = (
                db.execute(
                    select(BlobReferenceORM)
                    .where(BlobReferenceORM.blob_id == blob_id)
                    .where(BlobReferenceORM.ref_kind == ref_kind)
                    .where(BlobReferenceORM.ref_id == ref_id)
                )
                .scalars()
                .one()
            )
            db.commit()
            return orm_to_blob_reference(orm)

    def has_active_reference(
        self,
        blob_id: str,
        *,
        ref_kind: BlobReferenceKind,
        ref_id: str,
    ) -> bool:
        with self.session_factory() as db:
            return (
                db.execute(
                    select(BlobReferenceORM.id)
                    .where(BlobReferenceORM.blob_id == blob_id)
                    .where(BlobReferenceORM.ref_kind == ref_kind)
                    .where(BlobReferenceORM.ref_id == ref_id)
                    .where(BlobReferenceORM.status == "active")
                )
                .scalars()
                .first()
                is not None
            )

    def list_references(self, blob_id: str) -> list[BlobReference]:
        with self.session_factory() as db:
            self._require_orm(db, blob_id)
            rows = (
                db.execute(
                    select(BlobReferenceORM)
                    .where(BlobReferenceORM.blob_id == blob_id)
                    .order_by(BlobReferenceORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_blob_reference(row) for row in rows]

    def list_artifacts_for_reference(
        self,
        *,
        ref_kind: BlobReferenceKind,
        ref_id: str,
    ) -> list[BlobArtifact]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(BlobArtifactORM)
                    .join(BlobReferenceORM)
                    .where(BlobArtifactORM.status == "active")
                    .where(BlobReferenceORM.ref_kind == ref_kind)
                    .where(BlobReferenceORM.ref_id == ref_id)
                    .where(BlobReferenceORM.status == "active")
                    .order_by(BlobReferenceORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_blob_artifact(row) for row in rows]

    def mark_accessed(self, blob_id: str) -> BlobArtifact:
        with self.session_factory() as db:
            self._require_orm(db, blob_id)
            db.execute(
                update(BlobArtifactORM)
                .where(BlobArtifactORM.id == blob_id)
                .values(
                    last_accessed_at=_utc_now(),
                    updated_at=BlobArtifactORM.updated_at,
                )
            )
            db.commit()
            orm = self._require_orm(db, blob_id)
            return orm_to_blob_artifact(orm)

    def release_reference(
        self,
        blob_id: str,
        *,
        ref_kind: BlobReferenceKind,
        ref_id: str,
    ) -> None:
        with self.session_factory() as db:
            db.execute(
                update(BlobReferenceORM)
                .where(BlobReferenceORM.blob_id == blob_id)
                .where(BlobReferenceORM.ref_kind == ref_kind)
                .where(BlobReferenceORM.ref_id == ref_id)
                .where(BlobReferenceORM.status == "active")
                .values(status="released", released_at=_utc_now())
            )
            db.commit()

    def _require_orm(self, db: DbSession, blob_id: str) -> BlobArtifactORM:
        orm = db.get(BlobArtifactORM, blob_id)
        if orm is None:
            raise BlobArtifactNotFoundError(blob_id)
        return orm

    def _require_active_orm_for_update(
        self, db: DbSession, blob_id: str
    ) -> BlobArtifactORM:
        orm = (
            db.execute(
                select(BlobArtifactORM)
                .where(BlobArtifactORM.id == blob_id)
                .where(BlobArtifactORM.status == "active")
                .with_for_update()
            )
            .scalars()
            .first()
        )
        if orm is None:
            raise BlobArtifactNotFoundError(blob_id)
        return orm


class PostgresBlobCleanupRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def list_expired_upload_candidates(
        self, *, now: datetime, limit: int
    ) -> list[PdfUpload]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PdfUploadORM)
                    .where(PdfUploadORM.status.in_(("initiated", "uploaded", "failed")))
                    .where(PdfUploadORM.expires_at <= now)
                    .order_by(PdfUploadORM.expires_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [orm_to_pdf_upload(row) for row in rows]

    def expire_next_upload(
        self, *, now: datetime, delete_object: Callable[[str], None]
    ) -> PdfUpload | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(PdfUploadORM)
                    .where(PdfUploadORM.status.in_(("initiated", "uploaded", "failed")))
                    .where(PdfUploadORM.expires_at <= now)
                    .order_by(PdfUploadORM.expires_at.asc())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .first()
            )
            if orm is None:
                return None
            delete_object(orm.object_key)
            previous_error = orm.error_json
            orm.status = "expired"
            orm.error_json = {
                "code": "upload_expired",
                "message": "Expired staging upload cleaned up.",
                "cleaned_at": now.isoformat(),
                "previous_error": previous_error,
            }
            orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_pdf_upload(orm)

    def list_ttl_blob_cleanup_candidates(
        self, *, cutoff: datetime, limit: int
    ) -> list[BlobArtifact]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(BlobArtifactORM)
                    .where(BlobArtifactORM.status == "active")
                    .where(BlobArtifactORM.retention_policy == "ttl")
                    .where(BlobArtifactORM.expires_at <= cutoff)
                    .where(~_active_blob_reference_exists())
                    .order_by(BlobArtifactORM.expires_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [orm_to_blob_artifact(row) for row in rows]

    def tombstone_next_ttl_blob(
        self,
        *,
        cutoff: datetime,
        now: datetime,
        delete_object: Callable[[str], None],
    ) -> BlobArtifact | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(BlobArtifactORM)
                    .where(BlobArtifactORM.status == "active")
                    .where(BlobArtifactORM.retention_policy == "ttl")
                    .where(BlobArtifactORM.expires_at <= cutoff)
                    .where(~_active_blob_reference_exists())
                    .order_by(BlobArtifactORM.expires_at.asc())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .first()
            )
            if orm is None:
                return None
            has_active_reference = (
                db.execute(
                    select(BlobReferenceORM.id)
                    .where(BlobReferenceORM.blob_id == orm.id)
                    .where(BlobReferenceORM.status == "active")
                    .limit(1)
                )
                .scalars()
                .first()
                is not None
            )
            if has_active_reference:
                return None
            delete_object(orm.object_key)
            orm.status = "deleted"
            orm.deleted_at = now
            orm.cleanup_metadata_json = {
                "code": "ttl_blob_deleted",
                "message": "Unreferenced TTL blob object deleted.",
                "cleaned_at": now.isoformat(),
            }
            orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_blob_artifact(orm)



InvalidPdfUploadTransitionError = PdfUploadStateError

class PostgresPdfUploadRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def create(self, upload: PdfUpload) -> PdfUpload:
        with self.session_factory() as db:
            db.add(pdf_upload_to_orm(upload))
            db.commit()
        return upload

    def get(self, upload_id: str) -> PdfUpload | None:
        with self.session_factory() as db:
            orm = db.get(PdfUploadORM, upload_id)
            return orm_to_pdf_upload(orm) if orm is not None else None

    def mark_uploaded(self, upload_id: str) -> PdfUpload:
        return self._transition(upload_id, allowed={"initiated"}, target_status="uploaded")

    def finalize(
        self,
        upload_id: str,
        *,
        blob_id: str,
        actual_sha256: str,
        size_bytes: int,
    ) -> PdfUpload:
        return self._transition(
            upload_id,
            allowed={"uploaded"},
            target_status="finalized",
            reject_expired=True,
            updates={
                "blob_id": blob_id,
                "actual_sha256": actual_sha256,
                "size_bytes": size_bytes,
                "finalized_at": _utc_now(),
                "error_json": None,
            },
        )

    def mark_failed(self, upload_id: str, *, error_json: dict) -> PdfUpload:
        return self._transition(
            upload_id,
            allowed={"initiated", "uploaded"},
            target_status="failed",
            updates={"error_json": error_json},
        )

    def mark_enqueued(self, upload_id: str) -> PdfUpload:
        return self._transition(upload_id, allowed={"finalized"}, target_status="enqueued")

    def _transition(
        self,
        upload_id: str,
        *,
        allowed: set[PdfUploadStatus],
        target_status: PdfUploadStatus,
        updates: dict | None = None,
        reject_expired: bool = False,
    ) -> PdfUpload:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(PdfUploadORM)
                    .where(PdfUploadORM.id == upload_id)
                    .with_for_update()
                )
                .scalars()
                .first()
            )
            if orm is None:
                raise PdfUploadNotFoundError(upload_id)
            if orm.status not in allowed:
                raise PdfUploadStateError(
                    upload_id=upload_id, status=orm.status, target_status=target_status
                )
            if reject_expired and orm.expires_at <= _utc_now():
                raise PdfUploadExpiredError(upload_id)
            values = {"status": target_status, "updated_at": _utc_now(), **(updates or {})}
            for name, value in values.items():
                setattr(orm, name, value)
            db.commit()
            db.refresh(orm)
            return orm_to_pdf_upload(orm)


class WorkflowJobNotFoundError(ValueError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Workflow job not found: {job_id}")
        self.job_id = job_id


class InvalidWorkflowJobTransitionError(ValueError):
    def __init__(self, *, job_id: str, status: str, target_status: str) -> None:
        super().__init__(
            f"Cannot transition workflow job {job_id} from {status} "
            f"to {target_status}."
        )
        self.job_id = job_id
        self.status = status
        self.target_status = target_status


class WorkflowJobLeaseLostError(InvalidWorkflowJobTransitionError):
    pass


class PostgresWorkflowJobRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def create(self, job: WorkflowJob) -> WorkflowJob:
        with self.session_factory() as db:
            db.add(workflow_job_to_orm(job))
            db.commit()
        return job

    def enqueue_pdf_blob(
        self,
        *,
        session_id: str,
        upload_id: str,
        paper_id: str | None,
        skip_arxiv_metadata_fetch: bool,
        pipeline_version: str,
    ) -> WorkflowJob:
        with self.session_factory() as db:
            upload = (
                db.execute(
                    select(PdfUploadORM)
                    .where(PdfUploadORM.id == upload_id)
                    .with_for_update()
                )
                .scalars()
                .first()
            )
            if upload is None or upload.session_id != session_id:
                raise PdfUploadNotFoundError(upload_id)
            blob_id = upload.blob_id
            if blob_id is None:
                raise PdfUploadStateError(
                    upload_id=upload.id,
                    status=upload.status,
                    target_status="enqueued",
                )
            job = WorkflowJob(
                session_id=session_id,
                kind="analyze_pdf_blob",
                input_json={
                    "blob_id": blob_id,
                    "upload_id": upload.id,
                    "paper_id": paper_id,
                    "skip_arxiv_metadata_fetch": skip_arxiv_metadata_fetch,
                    "pipeline_version": pipeline_version,
                },
                idempotency_key=pdf_blob_idempotency_key(
                    session_id=session_id,
                    blob_id=blob_id,
                    paper_id=paper_id,
                    pipeline_version=pipeline_version,
                ),
                pipeline_version=pipeline_version,
                max_attempts=3,
                retry_policy_json={
                    "base_delay_seconds": 5,
                    "max_delay_seconds": 300,
                },
            )

            existing = (
                db.execute(
                    select(WorkflowJobORM).where(
                        WorkflowJobORM.idempotency_key == job.idempotency_key
                    )
                )
                .scalars()
                .first()
            )
            if upload.status != "finalized" and not (
                upload.status == "enqueued" and existing is not None
            ):
                raise PdfUploadStateError(
                    upload_id=upload.id,
                    status=upload.status,
                    target_status="enqueued",
                )
            if not _has_active_blob_reference(
                db, blob_id=blob_id, ref_kind="session", ref_id=session_id
            ):
                raise RegisteredPdfBlobNotAuthorizedError(
                    session_id=session_id, blob_id=blob_id
                )

            if existing is None:
                db.execute(
                    pg_insert(WorkflowJobORM)
                    .values(_workflow_job_values(job))
                    .on_conflict_do_nothing(
                        constraint="uq_workflow_jobs_idempotency_key"
                    )
                )
                existing = (
                    db.execute(
                        select(WorkflowJobORM).where(
                            WorkflowJobORM.idempotency_key == job.idempotency_key
                        )
                    )
                    .scalars()
                    .one()
                )

            _add_blob_reference(
                db,
                blob_id=blob_id,
                ref_kind="workflow_job",
                ref_id=existing.id,
                metadata={"upload_id": upload.id},
            )
            if upload.status == "finalized":
                upload.status = "enqueued"
                upload.updated_at = _utc_now()
            db.commit()
            db.refresh(existing)
            return orm_to_workflow_job(existing)

    def get(self, job_id: str) -> WorkflowJob | None:
        with self.session_factory() as db:
            orm = db.get(WorkflowJobORM, job_id)
            return orm_to_workflow_job(orm) if orm is not None else None

    def list_for_session(self, session_id: str, limit: int = 50) -> list[WorkflowJob]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(WorkflowJobORM)
                    .where(WorkflowJobORM.session_id == session_id)
                    .order_by(WorkflowJobORM.created_at.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [orm_to_workflow_job(row) for row in rows]

    def claim_next(
        self,
        *,
        worker_id: str,
        kinds: list[JobKind] | None = None,
        lease_seconds: int = 90,
    ) -> WorkflowJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")
        with self.session_factory() as db:
            now = _utc_now()
            query = (
                select(WorkflowJobORM)
                .where(
                    or_(
                        (
                            (WorkflowJobORM.status == "queued")
                            & (
                                (WorkflowJobORM.next_attempt_at.is_(None))
                                | (WorkflowJobORM.next_attempt_at <= now)
                            )
                        ),
                        (
                            (WorkflowJobORM.status == "running")
                            & (
                                (WorkflowJobORM.lease_expires_at.is_(None))
                                | (WorkflowJobORM.lease_expires_at <= now)
                            )
                        ),
                    )
                )
                .order_by(WorkflowJobORM.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if kinds is not None:
                query = query.where(WorkflowJobORM.kind.in_(list(kinds)))

            orm = db.execute(query).scalars().first()
            if orm is None:
                return None

            reclaimed = orm.status == "running"
            orm.status = "running"
            orm.locked_by = worker_id
            orm.locked_at = now
            orm.heartbeat_at = now
            orm.lease_expires_at = now + timedelta(seconds=lease_seconds)
            orm.next_attempt_at = None
            if orm.started_at is None:
                orm.started_at = now
            orm.attempts += 1
            orm.updated_at = now
            if reclaimed and orm.cancel_requested_at is not None:
                _finish_workflow_job(
                    db,
                    orm,
                    status="canceled",
                    now=now,
                    error_json=_job_canceled_error(),
                )
            elif reclaimed and orm.attempts >= orm.max_attempts:
                _finish_workflow_job(
                    db,
                    orm,
                    status="failed",
                    now=now,
                    error_json={
                        "error": "retry_exhausted",
                        "message": "Workflow job retry budget exhausted during lease reclaim.",
                    },
                )
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def heartbeat(
        self, job_id: str, *, worker_id: str, lease_seconds: int = 90
    ) -> WorkflowJob:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            self._require_active_lease(orm, worker_id=worker_id, now=now)
            orm.heartbeat_at = now
            orm.lease_expires_at = now + timedelta(seconds=lease_seconds)
            orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def is_cancel_requested(self, job_id: str, *, worker_id: str) -> bool:
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            self._require_active_lease(orm, worker_id=worker_id, now=now)
            return orm.cancel_requested_at is not None

    def mark_succeeded(
        self, job_id: str, *, worker_id: str, result_json: dict
    ) -> WorkflowJob:
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            self._require_active_lease(orm, worker_id=worker_id, now=now)
            if orm.cancel_requested_at is not None:
                _finish_workflow_job(
                    db,
                    orm,
                    status="canceled",
                    now=now,
                    error_json=_job_canceled_error(),
                )
            else:
                _finish_workflow_job(
                    db, orm, status="succeeded", now=now, result_json=result_json
                )
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def record_failure(
        self,
        job_id: str,
        *,
        worker_id: str,
        error_json: dict,
        retryable: bool,
    ) -> WorkflowJob:
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            self._require_active_lease(orm, worker_id=worker_id, now=now)
            if orm.cancel_requested_at is not None:
                _finish_workflow_job(
                    db,
                    orm,
                    status="canceled",
                    now=now,
                    error_json=_job_canceled_error(),
                )
            elif retryable and orm.attempts < orm.max_attempts:
                orm.status = "queued"
                orm.error_json = error_json
                orm.next_attempt_at = now + timedelta(
                    seconds=_retry_delay_seconds(orm)
                )
                _clear_workflow_job_lock(orm)
                orm.updated_at = now
            else:
                _finish_workflow_job(
                    db, orm, status="failed", now=now, error_json=error_json
                )
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def mark_canceled(self, job_id: str) -> WorkflowJob:
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            self._require_status(orm, {"queued", "running"}, target_status="canceled")
            now = _utc_now()
            if orm.status == "running":
                orm.cancel_requested_at = orm.cancel_requested_at or now
                orm.updated_at = now
            else:
                _finish_workflow_job(
                    db,
                    orm,
                    status="canceled",
                    now=now,
                    error_json=_job_canceled_error(),
                )
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def complete_canceled(self, job_id: str, *, worker_id: str) -> WorkflowJob:
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            self._require_active_lease(orm, worker_id=worker_id, now=now)
            if orm.cancel_requested_at is None:
                raise InvalidWorkflowJobTransitionError(
                    job_id=orm.id, status=orm.status, target_status="canceled"
                )
            _finish_workflow_job(
                db,
                orm,
                status="canceled",
                now=now,
                error_json=_job_canceled_error(),
            )
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def mark_running(
        self, job_id: str, *, worker_id: str, lease_seconds: int = 90
    ) -> WorkflowJob:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive.")
        with self.session_factory() as db:
            orm = self._require_orm_for_update(db, job_id)
            now = _utc_now()
            if orm.status == "running" and orm.locked_by == worker_id:
                self._require_active_lease(orm, worker_id=worker_id, now=now)
                orm.heartbeat_at = now
                orm.lease_expires_at = now + timedelta(seconds=lease_seconds)
                orm.updated_at = now
                db.commit()
                db.refresh(orm)
                return orm_to_workflow_job(orm)
            self._require_status(orm, {"queued"}, target_status="running")
            orm.status = "running"
            orm.locked_by = worker_id
            orm.locked_at = now
            orm.heartbeat_at = now
            orm.lease_expires_at = now + timedelta(seconds=lease_seconds)
            if orm.started_at is None:
                orm.started_at = now
            orm.attempts += 1
            orm.updated_at = now
            db.commit()
            db.refresh(orm)
            return orm_to_workflow_job(orm)

    def mark_failed(
        self, job_id: str, *, worker_id: str, error_json: dict
    ) -> WorkflowJob:
        return self.record_failure(
            job_id, worker_id=worker_id, error_json=error_json, retryable=False
        )

    def _require_orm(self, db: DbSession, job_id: str) -> WorkflowJobORM:
        orm = db.get(WorkflowJobORM, job_id)
        if orm is None:
            raise WorkflowJobNotFoundError(job_id)
        return orm

    def _require_orm_for_update(self, db: DbSession, job_id: str) -> WorkflowJobORM:
        orm = (
            db.execute(
                select(WorkflowJobORM)
                .where(WorkflowJobORM.id == job_id)
                .with_for_update()
            )
            .scalars()
            .first()
        )
        if orm is None:
            raise WorkflowJobNotFoundError(job_id)
        return orm

    def _require_status(
        self,
        orm: WorkflowJobORM,
        allowed: set[JobStatus],
        *,
        target_status: JobStatus,
    ) -> None:
        if orm.status not in allowed:
            raise InvalidWorkflowJobTransitionError(
                job_id=orm.id,
                status=orm.status,
                target_status=target_status,
            )

    def _require_active_lease(
        self, orm: WorkflowJobORM, *, worker_id: str, now: datetime
    ) -> None:
        if (
            orm.status != "running"
            or orm.locked_by != worker_id
            or orm.lease_expires_at is None
            or orm.lease_expires_at <= now
        ):
            raise WorkflowJobLeaseLostError(
                job_id=orm.id, status=orm.status, target_status="running"
            )


def _workflow_job_values(job: WorkflowJob) -> dict:
    return {
        "id": job.id,
        "session_id": job.session_id,
        "kind": job.kind,
        "status": job.status,
        "input_json": job.input_json,
        "result_json": job.result_json,
        "error_json": job.error_json,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "idempotency_key": job.idempotency_key,
        "pipeline_version": job.pipeline_version,
        "next_attempt_at": job.next_attempt_at,
        "retry_policy_json": job.retry_policy_json,
        "locked_by": job.locked_by,
        "locked_at": job.locked_at,
        "lease_expires_at": job.lease_expires_at,
        "heartbeat_at": job.heartbeat_at,
        "cancel_requested_at": job.cancel_requested_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _has_active_blob_reference(
    db: DbSession,
    *,
    blob_id: str,
    ref_kind: BlobReferenceKind,
    ref_id: str,
) -> bool:
    return (
        db.execute(
            select(BlobReferenceORM.id)
            .where(BlobReferenceORM.blob_id == blob_id)
            .where(BlobReferenceORM.ref_kind == ref_kind)
            .where(BlobReferenceORM.ref_id == ref_id)
            .where(BlobReferenceORM.status == "active")
        )
        .scalars()
        .first()
        is not None
    )


def _add_blob_reference(
    db: DbSession,
    *,
    blob_id: str,
    ref_kind: BlobReferenceKind,
    ref_id: str,
    metadata: dict | None = None,
) -> None:
    db.execute(
        pg_insert(BlobReferenceORM)
        .values(
            id=BlobReference(blob_id=blob_id, ref_kind=ref_kind, ref_id=ref_id).id,
            blob_id=blob_id,
            ref_kind=ref_kind,
            ref_id=ref_id,
            metadata_json=metadata or {},
        )
        .on_conflict_do_nothing(constraint="uq_blob_references_blob_kind_ref")
    )


def _release_workflow_job_reference(
    db: DbSession, *, job_id: str, released_at: datetime
) -> None:
    db.execute(
        update(BlobReferenceORM)
        .where(BlobReferenceORM.ref_kind == "workflow_job")
        .where(BlobReferenceORM.ref_id == job_id)
        .where(BlobReferenceORM.status == "active")
        .values(status="released", released_at=released_at)
    )


def _active_blob_reference_exists():
    return exists(
        select(BlobReferenceORM.id)
        .where(BlobReferenceORM.blob_id == BlobArtifactORM.id)
        .where(BlobReferenceORM.status == "active")
    )


def _clear_workflow_job_lock(orm: WorkflowJobORM) -> None:
    orm.locked_by = None
    orm.locked_at = None
    orm.lease_expires_at = None
    orm.heartbeat_at = None


def _finish_workflow_job(
    db: DbSession,
    orm: WorkflowJobORM,
    *,
    status: JobStatus,
    now: datetime,
    result_json: dict | None = None,
    error_json: dict | None = None,
) -> None:
    orm.status = status
    orm.result_json = result_json
    orm.error_json = error_json
    orm.finished_at = now
    orm.next_attempt_at = None
    _clear_workflow_job_lock(orm)
    orm.updated_at = now
    _release_workflow_job_reference(db, job_id=orm.id, released_at=now)


def _retry_delay_seconds(orm: WorkflowJobORM) -> float:
    policy = orm.retry_policy_json or {}
    base_delay = policy.get("base_delay_seconds", 5)
    max_delay = policy.get("max_delay_seconds", 300)
    if not isinstance(base_delay, (int, float)) or base_delay < 0:
        base_delay = 5
    if not isinstance(max_delay, (int, float)) or max_delay < 0:
        max_delay = 300
    return min(float(max_delay), float(base_delay) * (2 ** max(orm.attempts - 1, 0)))


def _job_canceled_error() -> dict:
    return {"error": "job_canceled", "message": "Workflow job cancellation requested."}


class PostgresSessionStore(SessionStore):
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def create_session(
        self,
        *,
        persona: Persona = "engineer",
        original_query: str | None = None,
    ) -> Session:
        session = Session(persona=persona, original_query=original_query)
        with self.session_factory() as db:
            db.add(session_to_orm(session))
            db.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            return orm_to_session(orm) if orm is not None else None

    def require_session(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session

    def update_phase(self, session_id: str, phase: SessionPhase) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            orm.phase = phase
            db.commit()
            db.refresh(orm)
            return orm_to_session(orm)

    def add_active_paper(self, session_id: str, paper_id: str) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            current_ids = list(orm.active_paper_ids or [])
            if paper_id not in current_ids:
                orm.active_paper_ids = [*current_ids, paper_id]
                db.commit()
                db.refresh(orm)
            return orm_to_session(orm)

    def set_selected_candidate_ids(
        self,
        session_id: str,
        candidate_ids: list[str],
    ) -> Session:
        with self.session_factory() as db:
            orm = db.get(SessionORM, session_id)
            if orm is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            orm.selected_candidate_ids = list(dict.fromkeys(candidate_ids))
            db.commit()
            db.refresh(orm)
            return orm_to_session(orm)

    def append_turn(
        self,
        session_id: str,
        *,
        role: TurnRole,
        content: str,
        intent: str | None = None,
        referenced_paper_ids: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        error=None,
        metadata: dict | None = None,
    ) -> Turn:
        with self.session_factory() as db:
            if db.get(SessionORM, session_id) is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            error_id = None
            if isinstance(error, StructuredError):
                if error.session_id is None:
                    error = error.model_copy(update={"session_id": session_id})
                db.merge(structured_error_to_orm(error))
                error_id = error.id

            turn = Turn(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                referenced_paper_ids=referenced_paper_ids or [],
                artifact_refs=artifact_refs or [],
                error=error if isinstance(error, StructuredError) else None,
                metadata=metadata or {},
            )
            db.add(turn_to_orm(turn, error_id=error_id))
            db.commit()
            return turn

    def list_recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        with self.session_factory() as db:
            if db.get(SessionORM, session_id) is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            rows = (
                db.execute(
                    select(TurnORM)
                    .where(TurnORM.session_id == session_id)
                    .order_by(TurnORM.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [orm_to_turn(row) for row in reversed(rows)]


class PostgresAgentRunPersistence(AgentRunPersistence):
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def save(self, run: AgentRun) -> None:
        with self.session_factory() as db:
            db.merge(agent_run_to_orm(run))
            db.commit()

    def get(self, run_id: str) -> AgentRun | None:
        with self.session_factory() as db:
            orm = db.get(AgentRunORM, run_id)
            return orm_to_agent_run(orm) if orm is not None else None


class PostgresStructuredErrorRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def save(self, error: StructuredError) -> StructuredError:
        with self.session_factory() as db:
            db.merge(structured_error_to_orm(error))
            db.commit()
        return error

    def list_for_session(self, session_id: str) -> list[StructuredError]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(StructuredErrorORM)
                    .where(StructuredErrorORM.session_id == session_id)
                    .order_by(StructuredErrorORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_structured_error(row) for row in rows]


class PostgresPaperChunkRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_many(self, chunks: list[PaperChunk]) -> UpsertChunksResult:
        if not chunks:
            return UpsertChunksResult()

        chunk_ids = [chunk.id for chunk in chunks]
        with self.session_factory() as db:
            existing_ids = set(
                db.execute(
                    select(PaperChunkORM.id).where(PaperChunkORM.id.in_(chunk_ids))
                )
                .scalars()
                .all()
            )
            for chunk in chunks:
                db.merge(paper_chunk_to_orm(chunk))
            db.commit()

        updated = len(existing_ids)
        inserted = len(chunks) - updated
        return UpsertChunksResult(inserted=inserted, updated=updated, skipped=0)

    def list_for_paper(self, paper_id: str) -> list[PaperChunk]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperChunkORM)
                    .where(PaperChunkORM.paper_id == paper_id)
                    .order_by(PaperChunkORM.chunk_index.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_paper_chunk(row) for row in rows]

    def get_many_by_ids(self, chunk_ids: Sequence[str]) -> list[PaperChunk]:
        if not chunk_ids:
            return []

        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperChunkORM).where(PaperChunkORM.id.in_(list(chunk_ids)))
                )
                .scalars()
                .all()
            )

        chunks_by_id = {row.id: orm_to_paper_chunk(row) for row in rows}
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]


class PostgresSearchCandidateRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_many(self, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        if not candidates:
            return []

        with self.session_factory() as db:
            for candidate in candidates:
                db.merge(search_candidate_to_orm(candidate))
            db.commit()
        return candidates

    def list_for_discovery_turn(
        self,
        session_id: str,
        discovery_turn_id: str,
    ) -> list[SearchCandidate]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(SearchCandidateORM)
                    .where(SearchCandidateORM.session_id == session_id)
                    .where(SearchCandidateORM.discovery_turn_id == discovery_turn_id)
                    .order_by(SearchCandidateORM.display_rank.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_search_candidate(row) for row in rows]

    def list_latest_for_session(self, session_id: str) -> list[SearchCandidate]:
        with self.session_factory() as db:
            latest_turn_id = (
                db.execute(
                    select(SearchCandidateORM.discovery_turn_id)
                    .where(SearchCandidateORM.session_id == session_id)
                    .order_by(SearchCandidateORM.created_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if latest_turn_id is None:
                return []

            rows = (
                db.execute(
                    select(SearchCandidateORM)
                    .where(SearchCandidateORM.session_id == session_id)
                    .where(SearchCandidateORM.discovery_turn_id == latest_turn_id)
                    .order_by(SearchCandidateORM.display_rank.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_search_candidate(row) for row in rows]

    def get_many_by_ids(self, candidate_ids: Sequence[str]) -> list[SearchCandidate]:
        if not candidate_ids:
            return []

        requested_ids = list(dict.fromkeys(candidate_ids))
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(SearchCandidateORM).where(
                        SearchCandidateORM.id.in_(requested_ids)
                    )
                )
                .scalars()
                .all()
            )
            by_id = {row.id: orm_to_search_candidate(row) for row in rows}
            return [by_id[candidate_id] for candidate_id in requested_ids if candidate_id in by_id]

    def update_status(
        self,
        candidate_id: str,
        status: CandidateStatus,
    ) -> SearchCandidate | None:
        if status not in get_args(CandidateStatus):
            raise ValueError(f"Invalid search candidate status: {status}")

        with self.session_factory() as db:
            orm = db.get(SearchCandidateORM, candidate_id)
            if orm is None:
                return None
            orm.status = status
            db.commit()
            db.refresh(orm)
            return orm_to_search_candidate(orm)


class PostgresPaperWorkspaceRepository:
    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self.session_factory = session_factory

    def upsert_workspace(self, workspace: PaperWorkspace) -> PaperWorkspace:
        with self.session_factory() as db:
            existing = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == workspace.session_id)
                    .where(PaperWorkspaceORM.paper_id == workspace.paper_id)
                )
                .scalars()
                .first()
            )
            if existing is None:
                db.add(paper_workspace_to_orm(workspace))
                db.commit()
                return workspace

            existing.title = workspace.title
            existing.source_url = workspace.source_url
            existing.pipeline_stage = workspace.pipeline_stage
            existing.pipeline_version = workspace.pipeline_version
            existing.finalized_report_json = workspace.finalized_report_json
            existing.method_extraction_json = workspace.method_extraction_json
            existing.benchmarks_json = workspace.benchmarks_json
            existing.readiness_json = workspace.readiness_json
            existing.full_markdown_report = workspace.full_markdown_report
            db.commit()
            db.refresh(existing)
            return orm_to_paper_workspace(existing)

    def list_workspaces(self, session_id: str) -> list[PaperWorkspace]:
        with self.session_factory() as db:
            rows = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == session_id)
                    .order_by(PaperWorkspaceORM.created_at.asc())
                )
                .scalars()
                .all()
            )
            return [orm_to_paper_workspace(row) for row in rows]

    def get_workspace(
        self,
        session_id: str,
        paper_id: str,
    ) -> PaperWorkspace | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(PaperWorkspaceORM)
                    .where(PaperWorkspaceORM.session_id == session_id)
                    .where(PaperWorkspaceORM.paper_id == paper_id)
                )
                .scalars()
                .first()
            )
            return orm_to_paper_workspace(orm) if orm is not None else None

    def save_comparison(
        self,
        artifact: ComparisonArtifact,
    ) -> ComparisonArtifact:
        with self.session_factory() as db:
            db.merge(comparison_artifact_to_orm(artifact))
            db.commit()
        return artifact

    def latest_comparison(self, session_id: str) -> ComparisonArtifact | None:
        with self.session_factory() as db:
            orm = (
                db.execute(
                    select(ComparisonArtifactORM)
                    .where(ComparisonArtifactORM.session_id == session_id)
                    .order_by(ComparisonArtifactORM.created_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            return orm_to_comparison_artifact(orm) if orm is not None else None


def clear_foundation_tables(db: DbSession) -> None:
    db.execute(delete(PdfUploadORM))
    db.execute(delete(BlobReferenceORM))
    db.execute(delete(BlobArtifactORM))
    db.execute(delete(WorkflowJobORM))
    db.execute(delete(ArxivMetadataCacheORM))
    db.execute(delete(ComparisonArtifactORM))
    db.execute(delete(PaperWorkspaceORM))
    db.execute(delete(SearchCandidateORM))
    db.execute(delete(PaperChunkORM))
    db.execute(delete(TurnORM))
    db.execute(delete(AgentRunORM))
    db.execute(delete(StructuredErrorORM))
    db.execute(delete(SessionORM))
    db.commit()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
