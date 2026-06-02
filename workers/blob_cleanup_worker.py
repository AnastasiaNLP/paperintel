from __future__ import annotations

import argparse
import logging
import time

from api.app_factory import create_paperintel_service
from services.blob_cleanup import BlobCleanupService
from storage.db import make_engine, make_session_factory
from storage.repositories import PostgresBlobCleanupRepository


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperIntel blob cleanup.")
    parser.add_argument("--once", action="store_true", help="Run one cleanup batch.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Seconds to sleep between cleanup batches in daemon mode.",
    )
    parser.add_argument("--upload-batch-size", type=int, default=100)
    parser.add_argument("--blob-batch-size", type=int, default=100)
    parser.add_argument("--blob-grace-period-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="BLOB_CLEANUP_WORKER %(levelname)s %(name)s %(message)s",
    )

    from config.settings import settings

    service = create_paperintel_service(enable_health_checks=False)
    if service.blob_store is None:
        raise RuntimeError("Blob storage is not configured.")
    engine = make_engine(settings.postgres_url)
    cleanup = BlobCleanupService(
        repository=PostgresBlobCleanupRepository(make_session_factory(engine)),
        blob_store=service.blob_store,
        upload_expiry_batch_size=args.upload_batch_size,
        blob_batch_size=args.blob_batch_size,
        blob_grace_period_seconds=args.blob_grace_period_seconds,
    )
    try:
        while True:
            summary = cleanup.run_once(dry_run=args.dry_run)
            LOGGER.info("Blob cleanup batch: %s", summary.model_dump(mode="json"))
            if args.once:
                break
            time.sleep(args.interval_seconds)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
