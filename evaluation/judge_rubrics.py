from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUBRIC_DIR = Path("evaluation/rubrics")
EXPECTED_RUBRIC_IDS = {
    "action_reasoning",
    "comparison_balance",
    "comparison_grounding",
    "comparison_recommendation_justification",
    "implementation_difficulty",
    "qa_faithfulness",
    "recommended_action",
    "synthesis_citation_faithfulness",
    "synthesis_justification",
    "synthesis_persona_fit",
}


class RubricLoaderError(ValueError):
    """Raised when judge rubrics cannot be loaded or validated."""


@dataclass(frozen=True)
class JudgeRubric:
    rubric_id: str
    path: Path
    text: str
    sha256: str


def load_judge_rubrics(
    rubric_dir: str | Path = DEFAULT_RUBRIC_DIR,
) -> dict[str, JudgeRubric]:
    path = Path(rubric_dir)
    try:
        files = sorted(
            file
            for file in path.glob("*.md")
            if file.name != "README.md"
        )
    except OSError as exc:
        raise RubricLoaderError(f"Could not read rubric directory: {path}") from exc

    rubrics = {
        file.stem: _load_rubric(file)
        for file in files
    }
    missing = EXPECTED_RUBRIC_IDS - set(rubrics)
    if missing:
        raise RubricLoaderError(
            "Missing judge rubrics: " + ",".join(sorted(missing))
        )
    return rubrics


def _load_rubric(path: Path) -> JudgeRubric:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RubricLoaderError(f"Could not read rubric file: {path}") from exc

    if not text.strip():
        raise RubricLoaderError(f"Rubric file is empty: {path}")

    return JudgeRubric(
        rubric_id=path.stem,
        path=path,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
