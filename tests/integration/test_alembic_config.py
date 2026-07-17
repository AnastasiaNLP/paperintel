from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_alembic_config_points_to_project_migrations():
    config = Config("alembic.ini")

    assert config.get_main_option("script_location") == "alembic"
    assert "paperintel" in config.get_main_option("sqlalchemy.url")


def test_alembic_has_single_head_revision():
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    assert heads == ["20260625_0012"]
    assert Path("alembic/versions/20260504_0001_initial_session_schema.py").exists()
    assert Path("alembic/versions/20260511_0002_paper_chunks.py").exists()
    assert Path("alembic/versions/20260517_0003_search_candidates.py").exists()
    assert Path("alembic/versions/20260518_0004_artifact_workspaces.py").exists()
    assert Path("alembic/versions/20260527_0005_workflow_jobs.py").exists()
    assert Path("alembic/versions/20260527_0006_arxiv_metadata_cache.py").exists()
    assert Path("alembic/versions/20260601_0007_blob_artifacts.py").exists()
    assert Path("alembic/versions/20260602_0008_async_pdf_foundation.py").exists()
    assert Path("alembic/versions/20260602_0009_blob_cleanup_tombstones.py").exists()
    assert Path("alembic/versions/20260608_0010_provider_rate_limits.py").exists()
    assert Path("alembic/versions/20260608_0011_provider_circuit_breakers.py").exists()
    assert Path("alembic/versions/20260625_0012_benchmark_candidate_pool.py").exists()
