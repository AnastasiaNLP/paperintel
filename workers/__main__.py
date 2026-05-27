from __future__ import annotations

import argparse
import logging
from uuid import uuid4

from api.app_factory import create_paperintel_service
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresWorkflowJobRepository
from workers.workflow_worker import (
    SUPPORTED_JOB_KINDS,
    WorkflowJobExecutor,
    WorkflowWorker,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperIntel workflow jobs.")
    parser.add_argument("--once", action="store_true", help="Process at most one job.")
    parser.add_argument(
        "--worker-id",
        default=f"worker-{uuid4().hex[:12]}",
        help="Stable worker identifier for job locks.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds to sleep when no job is available in daemon mode.",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=sorted(SUPPORTED_JOB_KINDS),
        help="Limit worker to a supported job kind. May be passed multiple times.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="WORKFLOW_WORKER %(levelname)s %(name)s %(message)s",
    )

    from config.settings import settings

    service = create_paperintel_service(enable_health_checks=False)
    engine = make_engine(settings.postgres_url)
    session_factory = make_session_factory(engine)
    repository = PostgresWorkflowJobRepository(session_factory)
    worker = WorkflowWorker(
        repository=repository,
        executor=WorkflowJobExecutor(service),
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        kinds=args.kind if args.kind is not None else sorted(SUPPORTED_JOB_KINDS),
    )
    try:
        if args.once:
            worker.run_once()
        else:
            worker.run_forever()
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
