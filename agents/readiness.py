import copy
import json
import logging
import re
from pathlib import Path
from typing import Optional

import anthropic
import httpx

from agents.error_utils import paper_error
from config.settings import settings
from models.schemas import ProductionReadiness
from models.state import PaperIntelState
from tools.github_checker import check_repo, check_requirements
from tools.paper_resources_client import get_resources

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "readiness_prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Specific generic framework repos to skip, not an owner-wide blacklist.
# This skips "pytorch/pytorch" and "huggingface/transformers", but still allows
# paper repos such as "facebookresearch/llama", "openai/whisper", and
# "google-research/bert".
_GENERIC_FRAMEWORK_REPOS = {
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("tensorflow", "models"),
    ("google-research", "google-research"),
    ("huggingface", "transformers"),
    ("huggingface", "diffusers"),
    ("huggingface", "datasets"),
    ("huggingface", "accelerate"),
    ("huggingface", "peft"),
    ("huggingface", "tokenizers"),
    ("scikit-learn", "scikit-learn"),
    ("numpy", "numpy"),
    ("pandas-dev", "pandas"),
    ("scipy", "scipy"),
    ("keras-team", "keras"),
    ("matplotlib", "matplotlib"),
    ("apache", "spark"),
    ("apache", "flink"),
}

VALID_MATURITY_LEVELS = {"research_only", "experimental", "production_ready"}
UNAVAILABLE_GITHUB_STATUSES = {"source_unavailable", "forbidden", "rate_limited"}
MAX_GITHUB_CANDIDATES = 3
MAX_LLM_GITHUB_CANDIDATES = 2
MAX_LLM_HF_CANDIDATES = 2
PRODUCTION_READY_MIN_STARS = 500
RESOURCE_HINTS_CHARS = 2000
SNIPPETS_HEAD_CHARS = 3000
SNIPPETS_TAIL_CHARS = 2000
SNIPPETS_MAX_HINTS = 80


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_text_block(
    response: object,
    context: str,
) -> tuple[Optional[str], Optional[str]]:
    content = getattr(response, "content", None)
    if not content:
        return None, f"{context}: empty content"

    block = content[0]
    raw = getattr(block, "text", None)
    if not isinstance(raw, str) or not raw.strip():
        return None, f"{context}: non-text block"

    return raw.strip(), None


def _normalize_github_url(url: str) -> str:
    """Canonicalize GitHub URLs for dedup across trailing slash, .git, and case."""
    return url.rstrip("/").removesuffix(".git").lower()


def _extract_resource_snippets(raw_text: Optional[str]) -> str:
    """
    Sample the start and end of the paper plus keyword-matching lines.

    Availability/model-card sections often appear at the end of a paper, while
    method framing appears at the start. This keeps snippets compact without
    losing common code/model availability sections.
    """
    if not raw_text:
        return ""

    keywords = [
        "github.com",
        "huggingface.co",
        "code",
        "implementation",
        "repository",
        "available at",
        "released",
        "open-source",
        "open source",
        "model weights",
        "checkpoint",
        "weights",
        "GPU",
        "A100",
        "H100",
        "H800",
        "V100",
        "TPU",
        "inference",
        "latency",
        "throughput",
        "parameters",
        "billion",
    ]

    relevant = []
    seen = set()
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if any(keyword.lower() in lower for keyword in keywords) and stripped not in seen:
            seen.add(stripped)
            relevant.append(stripped)

    head = raw_text[:SNIPPETS_HEAD_CHARS]
    tail = (
        raw_text[-SNIPPETS_TAIL_CHARS:]
        if len(raw_text) > SNIPPETS_HEAD_CHARS + SNIPPETS_TAIL_CHARS
        else ""
    )
    hints = "\n".join(relevant[:SNIPPETS_MAX_HINTS])

    parts = [f"[Paper start]:\n{head}"]
    if tail:
        parts.append(f"[Paper end]:\n{tail}")
    parts.append(f"[Relevant lines]:\n{hints}")

    result = "\n\n".join(parts)
    cap = RESOURCE_HINTS_CHARS + SNIPPETS_HEAD_CHARS + SNIPPETS_TAIL_CHARS
    return result[:cap]


def _extract_github_urls(raw_text: str) -> list[str]:
    """Extract GitHub URLs and skip only specific generic framework repos."""
    pattern = (
        r"https?://(?:www\.)?github\.com/"
        r"([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-\.]+)"
        r"(?:\.git)?(?:[/#?][^\s]*)?"
    )
    matches = re.findall(pattern, raw_text)

    seen = set()
    candidates = []
    for owner, repo in matches:
        repo = repo.rstrip(".,);]}")
        owner_lower = owner.lower()
        repo_lower = repo.lower()

        if (owner_lower, repo_lower) in _GENERIC_FRAMEWORK_REPOS:
            continue

        url = f"https://github.com/{owner}/{repo}".removesuffix(".git")
        key = _normalize_github_url(url)
        if key not in seen:
            seen.add(key)
            candidates.append(url)

    return candidates


def _detect_framework_mentions(raw_text: Optional[str], deps: list[str]) -> list[str]:
    frameworks = {
        "PyTorch": [r"\bpytorch\b", r"\btorch\b"],
        "TensorFlow": [r"\btensorflow\b"],
        "JAX": [r"\bjax\b", r"\bflax\b"],
        "HuggingFace Transformers": [r"\btransformers\b", r"\bhuggingface\b"],
        "vLLM": [r"\bvllm\b"],
        "LangChain": [r"\blangchain\b"],
        "LangGraph": [r"\blanggraph\b"],
        "ONNX": [r"\bonnx\b"],
        "DeepSpeed": [r"\bdeepspeed\b"],
        "FAISS": [r"\bfaiss\b"],
    }

    text_lower = (raw_text or "").lower()
    deps_lower = [dependency.lower() for dependency in deps]
    found = set()

    for name, patterns in frameworks.items():
        for pattern in patterns:
            if re.search(pattern, text_lower) or any(
                re.search(pattern, dependency) for dependency in deps_lower
            ):
                found.add(name)
                break

    return sorted(found)


def _collect_hf_evidence(arxiv_id: str) -> dict:
    try:
        result = get_resources(arxiv_id)
        source_available = result.get("source_available", True)

        if not result.get("paper_found"):
            return {
                "source_available": source_available,
                "model_id": None,
                "model_url": None,
                "downloads": 0,
                "likes": 0,
                "total_resources": 0,
            }

        resources = result.get("results", [])
        models = [item for item in resources if item.get("repo_type") == "model"]
        best = max(
            models,
            key=lambda item: (item.get("downloads") or 0, item.get("likes") or 0),
            default=None,
        )

        return {
            "source_available": source_available,
            "model_id": best["repo_id"] if best else None,
            "model_url": best["url"] if best else None,
            "downloads": (best.get("downloads") or 0) if best else 0,
            "likes": (best.get("likes") or 0) if best else 0,
            "total_resources": len(resources),
        }
    except Exception as exc:
        logger.warning("HF evidence failed: %s", exc)
        return {
            "source_available": False,
            "model_id": None,
            "model_url": None,
            "downloads": 0,
            "likes": 0,
            "total_resources": 0,
        }


def _check_single_github(url: str) -> dict:
    try:
        result = check_repo(url)
    except Exception as exc:
        logger.warning("GitHub check failed for %s: %s", url, exc)
        return {"url": url, "status": "source_unavailable", "source_available": False}

    status = result.get("status") or "unknown"
    if status in UNAVAILABLE_GITHUB_STATUSES or not result.get("source_available", True):
        return {"url": url, "status": status, "source_available": False}

    if status == "found" and result.get("result"):
        repo = result["result"]
        deps = []
        try:
            deps_result = check_requirements(url)
            deps = deps_result.get("dependencies", [])[:20]
        except Exception as deps_exc:
            logger.warning("Deps check failed for %s: %s", url, deps_exc)

        return {
            "url": url,
            "status": "found",
            "source_available": True,
            "stars": repo.get("stars", 0),
            "forks": repo.get("forks", 0),
            "last_commit": repo.get("last_commit_date"),
            "has_releases": repo.get("has_releases", False),
            "is_archived": repo.get("is_archived", False),
            "license": repo.get("license"),
            "language": repo.get("language"),
            "dependencies": deps,
        }

    return {"url": url, "status": status, "source_available": True}


def _collect_github_evidence(raw_text: Optional[str]) -> dict:
    if not raw_text:
        return {
            "source_available": True,
            "found": False,
            "candidates_checked": 0,
            "candidates": [],
            "best_repo": None,
        }

    candidate_urls = _extract_github_urls(raw_text)[:MAX_GITHUB_CANDIDATES]
    if not candidate_urls:
        return {
            "source_available": True,
            "found": False,
            "candidates_checked": 0,
            "candidates": [],
            "best_repo": None,
        }

    checked = []
    best_repo = None

    for url in candidate_urls:
        info = _check_single_github(url)

        candidate = {"url": url, "status": info.get("status", "unknown")}
        if "stars" in info:
            candidate["stars"] = info.get("stars", 0)
        checked.append(candidate)

        if info.get("status") == "found":
            if best_repo is None or info.get("stars", 0) > best_repo.get("stars", 0):
                best_repo = info

    all_unavailable = (
        bool(checked)
        and all(item.get("status") in UNAVAILABLE_GITHUB_STATUSES for item in checked)
    )

    return {
        "source_available": not all_unavailable,
        "found": best_repo is not None,
        "candidates_checked": len(checked),
        "candidates": checked,
        "best_repo": best_repo,
    }


def _verify_hf_model(model_id: str) -> dict:
    """
    Verify a HuggingFace model candidate and preserve reason for debugging.

    Returns:
      - verified=True, reason="ok" for 200 with valid model JSON
      - verified=False, reason="not_found" for 404 or other non-200 statuses
      - verified=False, reason="rate_limited" for 429
      - verified=False, reason="unreachable" for network/timeout failures
      - verified=False, reason="malformed" for invalid model id shape
      - verified=False, reason="bad_response" for 200 with unusable body
    """
    if not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", model_id):
        return {"verified": False, "reason": "malformed"}

    try:
        response = httpx.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
        )
    except Exception as exc:
        logger.warning("HF verify unreachable for %s: %s", model_id, exc)
        return {"verified": False, "reason": "unreachable"}

    if response.status_code == 429:
        logger.warning("HF verify rate-limited for %s", model_id)
        return {"verified": False, "reason": "rate_limited"}

    if response.status_code == 200:
        try:
            data = response.json()
        except Exception:
            return {"verified": False, "reason": "bad_response"}
        if isinstance(data, dict) and (data.get("id") or data.get("modelId")):
            return {"verified": True, "reason": "ok"}
        return {"verified": False, "reason": "bad_response"}

    return {"verified": False, "reason": "not_found"}


def _build_evidence_json(
    state: PaperIntelState,
    hf: dict,
    github: dict,
    snippets: str,
    framework_mentions: list[str],
) -> str:
    metadata = state.get("metadata")
    extraction = state.get("method_extraction")
    benchmarks = state.get("benchmarks", [])
    best_repo = github.get("best_repo") or {}

    evidence = {
        "paper": {
            "title": metadata.title if metadata else None,
            "abstract": metadata.abstract[:800] if metadata else None,
            "arxiv_id": metadata.arxiv_id if metadata else None,
        },
        "method": {
            "name": extraction.method_name if extraction else None,
            "description": extraction.description if extraction else None,
            "novelty_claim": extraction.novelty_claim if extraction else None,
            "key_components": extraction.key_components[:5] if extraction else [],
        },
        "benchmarks": [
            {"task": item.task, "metric": item.metric, "value": item.value}
            for item in benchmarks[:5]
        ],
        "deterministic_evidence": {
            "hf": {
                "model_id": hf.get("model_id"),
                "downloads": hf.get("downloads", 0),
                "total_resources": hf.get("total_resources", 0),
                "source_available": hf.get("source_available", True),
            },
            "github": {
                "found": github.get("found"),
                "best_repo": {
                    "url": best_repo.get("url"),
                    "stars": best_repo.get("stars", 0),
                    "has_releases": best_repo.get("has_releases", False),
                    "last_commit": best_repo.get("last_commit"),
                }
                if best_repo
                else None,
                "candidates_checked": github.get("candidates_checked", 0),
                "candidates": github.get("candidates", []),
                "source_available": github.get("source_available", True),
            },
            "framework_mentions": framework_mentions,
            "dependencies": best_repo.get("dependencies", []),
        },
        "snippets": snippets,
    }

    return json.dumps(evidence, ensure_ascii=False, indent=2)


def _call_llm(evidence_json: str) -> tuple[Optional[str], Optional[str]]:
    try:
        response = _client.messages.create(
            model=settings.haiku_model,
            max_tokens=800,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": evidence_json}],
        )
        return _extract_text_block(response, "Readiness LLM")
    except Exception as exc:
        logger.exception("Readiness LLM failed")
        return None, f"Readiness LLM failed: {exc}"


def _call_llm_repair(bad_json: str) -> tuple[Optional[str], Optional[str]]:
    try:
        response = _client.messages.create(
            model=settings.haiku_model,
            max_tokens=800,
            system=(
                "You are a JSON repair specialist. Return ONLY a valid JSON object. "
                'The first character must be "{". The last character must be "}". '
                "No prose, markdown, or explanation."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Fix invalid JSON and return only the JSON object:\n\n"
                        f"{bad_json[:3000]}"
                    ),
                }
            ],
        )
        return _extract_text_block(response, "Readiness repair")
    except Exception as exc:
        return None, f"Repair failed: {exc}"


def _parse_claims(raw_json: str) -> tuple[Optional[dict], Optional[str]]:
    cleaned = _strip_fences(raw_json)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    if not isinstance(data, dict):
        return None, f"Expected object, got {type(data).__name__}"

    return data, None


def _verify_claims(claims: dict, hf: dict, github: dict) -> dict:
    best_repo = github.get("best_repo") or {}
    verified_github = copy.deepcopy(best_repo) if best_repo else None
    verified_hf_model = hf.get("model_id")

    already_checked = {
        _normalize_github_url(item.get("url", ""))
        for item in github.get("candidates", [])
    }

    github_attempts = []
    for url in _as_string_list(claims.get("candidate_code_urls"))[:MAX_LLM_GITHUB_CANDIDATES]:
        if "github.com/" not in url:
            github_attempts.append({"url": url, "outcome": "not_github"})
            continue

        normalized_url = _normalize_github_url(url)
        if normalized_url in already_checked:
            github_attempts.append({"url": url, "outcome": "already_checked"})
            continue

        info = _check_single_github(url)
        attempt_outcome = info.get("status", "unknown")
        if info.get("status") == "found":
            if verified_github is None or info.get("stars", 0) > verified_github.get("stars", 0):
                verified_github = info
                logger.info(
                    "LLM GitHub candidate verified: %s (%d stars)",
                    url,
                    info.get("stars", 0),
                )
        github_attempts.append({"url": url, "outcome": attempt_outcome})

    hf_attempts: list[dict] = []
    if not verified_hf_model:
        for model_id in _as_string_list(claims.get("candidate_hf_models"))[
            :MAX_LLM_HF_CANDIDATES
        ]:
            outcome = _verify_hf_model(model_id)
            hf_attempts.append({"model_id": model_id, **outcome})
            if outcome.get("verified"):
                verified_hf_model = model_id
                logger.info("LLM HF candidate verified: %s", model_id)
                break

    return {
        "verified_github": verified_github,
        "verified_hf_model": verified_hf_model,
        "github_attempts": github_attempts,
        "hf_attempts": hf_attempts,
    }


def _append_downgrade_reason(reasoning: str, note: str) -> str:
    reasoning = reasoning.strip()
    if reasoning:
        return f"{reasoning} {note}"
    return note


def _is_inference_gpu_requirement(gpu_req: str) -> bool:
    """
    Keep the requirement if it explicitly mentions inference.
    Drop it only when the text is training-only.
    """
    text = gpu_req.lower()

    inference_hints = [
        "inference",
        "serving",
        "serve",
        "deploy",
        "deployment",
        "runtime",
        "per token",
        "per-token",
        "tok/s",
    ]
    training_hints = [
        "training",
        "train",
        "trained",
        "pretraining",
        "pre-training",
        "fine-tun",
        "finetun",
    ]

    has_inference = any(hint in text for hint in inference_hints)
    has_training = any(hint in text for hint in training_hints)

    if has_inference:
        return True
    if has_training:
        return False
    return True


def _normalize(
    claims: dict,
    verified: dict,
    hf: dict,
    framework_mentions: list[str],
) -> tuple[Optional[ProductionReadiness], Optional[str]]:
    verified_github = verified.get("verified_github") or {}
    verified_hf_model = verified.get("verified_hf_model")

    has_open_code = bool(verified_github.get("url")) or bool(verified_hf_model)
    code_url = verified_github.get("url")

    llm_frameworks = _as_string_list(claims.get("framework_integrations"))
    all_frameworks = sorted(set(framework_mentions) | set(llm_frameworks))

    det_deps = verified_github.get("dependencies", [])
    llm_deps = _as_string_list(claims.get("additional_dependencies"))
    dependencies = sorted(set(det_deps) | set(llm_deps))

    maturity_level = str(claims.get("maturity_level") or "research_only")
    if maturity_level not in VALID_MATURITY_LEVELS:
        maturity_level = "research_only"

    maturity_reasoning = str(claims.get("maturity_reasoning") or "")

    if not has_open_code and maturity_level in {"experimental", "production_ready"}:
        maturity_level = "research_only"
        maturity_reasoning = _append_downgrade_reason(
            maturity_reasoning,
            "Normalization downgraded maturity to research_only because no verified GitHub repo or HuggingFace model was available after candidate verification.",
        )

    # Production-ready requires the stars floor plus at least one additional
    # strong signal: releases or verified HF model. This keeps production_ready
    # reachable for mature code repos without HF models while preventing low-
    # evidence artifacts from being marked production-ready.
    if has_open_code and maturity_level == "production_ready":
        stars = verified_github.get("stars", 0)
        has_releases = bool(verified_github.get("has_releases"))
        hf_ok = bool(verified_hf_model)

        if stars < PRODUCTION_READY_MIN_STARS or not (has_releases or hf_ok):
            maturity_level = "experimental"
            maturity_reasoning = _append_downgrade_reason(
                maturity_reasoning,
                (
                    f"Normalization downgraded maturity to experimental: "
                    f"production_ready requires >= {PRODUCTION_READY_MIN_STARS} stars "
                    f"plus at least one of releases or verified HF model; "
                    f"observed stars={stars}, releases={has_releases}, hf_model={hf_ok}."
                ),
            )

    if has_open_code and maturity_level == "research_only":
        maturity_level = "experimental"
        maturity_reasoning = _append_downgrade_reason(
            maturity_reasoning,
            "Upgraded to experimental: verified open code or HF model confirmed after candidate verification.",
        )

    gpu_req = claims.get("min_gpu_requirement") or None
    if gpu_req and not _is_inference_gpu_requirement(gpu_req):
        logger.info("Dropping min_gpu_requirement because it is training-only: %r", gpu_req)
        gpu_req = None

    try:
        return (
            ProductionReadiness(
                has_open_code=has_open_code,
                code_url=code_url,
                huggingface_model=verified_hf_model,
                framework_integrations=all_frameworks,
                min_gpu_requirement=gpu_req,
                estimated_inference_cost=claims.get("estimated_inference_cost") or None,
                dependencies=dependencies,
                maturity_level=maturity_level,
                maturity_reasoning=maturity_reasoning,
            ),
            None,
        )
    except Exception as exc:
        return None, f"ProductionReadiness validation error: {exc}"


def readiness_agent(state: PaperIntelState) -> dict:
    logger.info("Readiness agent started")

    metadata = state.get("metadata")
    arxiv_id = metadata.arxiv_id if metadata else None
    raw_text = state.get("raw_text")

    hf = (
        _collect_hf_evidence(arxiv_id)
        if arxiv_id
        else {
            "source_available": True,
            "model_id": None,
            "model_url": None,
            "downloads": 0,
            "likes": 0,
            "total_resources": 0,
        }
    )

    github = _collect_github_evidence(raw_text)
    best_repo = github.get("best_repo") or {}
    snippets = _extract_resource_snippets(raw_text)
    framework_mentions = _detect_framework_mentions(
        raw_text,
        best_repo.get("dependencies", []),
    )

    logger.info(
        "Deterministic: hf_model=%s github_found=%s stars=%s frameworks=%s",
        hf.get("model_id"),
        github.get("found"),
        best_repo.get("stars", 0),
        framework_mentions,
    )

    evidence_json = _build_evidence_json(state, hf, github, snippets, framework_mentions)
    raw, llm_error = _call_llm(evidence_json)
    if llm_error:
        return paper_error(state, llm_error, "readiness")

    claims, parse_error = _parse_claims(raw or "")
    if parse_error:
        repaired, repair_error = _call_llm_repair(raw or "")
        if repair_error:
            return paper_error(
                state,
                f"Readiness parse failed: {parse_error}; repair: {repair_error}",
                "readiness",
            )
        claims, parse_error = _parse_claims(repaired or "")

    if parse_error or claims is None:
        return paper_error(
            state,
            f"Readiness parse failed after repair: {parse_error}",
            "readiness",
        )

    verified = _verify_claims(claims, hf, github)
    logger.info(
        "Verified: github_url=%s hf_model=%s github_attempts=%d hf_attempts=%d",
        (verified.get("verified_github") or {}).get("url"),
        verified.get("verified_hf_model"),
        len(verified.get("github_attempts", [])),
        len(verified.get("hf_attempts", [])),
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("GitHub attempts: %s", verified.get("github_attempts"))
        logger.debug("HF attempts: %s", verified.get("hf_attempts"))

    result, norm_error = _normalize(claims, verified, hf, framework_mentions)
    if norm_error:
        return paper_error(state, norm_error, "readiness")

    logger.info(
        "Readiness complete: maturity=%s has_code=%s hf=%s",
        result.maturity_level,
        result.has_open_code,
        result.huggingface_model,
    )

    return {
        "production_readiness": result,
        "processing_stage": "report",
    }
