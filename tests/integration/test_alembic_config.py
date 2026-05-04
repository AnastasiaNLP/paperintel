from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_alembic_config_points_to_project_migrations():
    config = Config("alembic.ini")

    assert config.get_main_option("script_location") == "alembic"
    assert "paperintel" in config.get_main_option("sqlalchemy.url")


def test_alembic_has_single_initial_head_revision():
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    assert heads == ["20260504_0001"]
    assert Path("alembic/versions/20260504_0001_initial_session_schema.py").exists()
