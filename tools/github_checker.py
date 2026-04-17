import base64
import logging
import re
import time
from typing import Literal, Optional, TypedDict

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
RATE_LIMIT_DELAY = 1.0

_headers = {"Accept": "application/vnd.github+json"}
if settings.github_token:
    _headers["Authorization"] = f"Bearer {settings.github_token}"

_timeout = httpx.Timeout(30.0, connect=5.0)
_client = httpx.Client(
    timeout=_timeout,
    follow_redirects=True,
    headers=_headers,
)

RepoStatus = Literal[
    "found",
    "not_found",
    "source_unavailable",
    "rate_limited",
    "invalid_url",
    "forbidden",
]


class RepoInfo(TypedDict):
    owner: str
    repo: str
    url: str
    stars: int
    forks: int
    open_issues: int
    language: Optional[str]
    license: Optional[str]
    readme_exists: Optional[bool]
    last_commit_date: Optional[str]
    topics: list[str]
    is_archived: bool
    has_releases: Optional[bool]
    default_branch: str
    partial_data: bool


class RepoResponse(TypedDict):
    source_available: bool
    repo_found: Optional[bool]
    status: RepoStatus
    reason: Optional[str]
    result: Optional[RepoInfo]


class DependenciesResponse(TypedDict):
    source_available: bool
    repo_found: Optional[bool]
    status: RepoStatus
    reason: Optional[str]
    dependencies: list[str]
    source_file: Optional[str]


def _rate_limit() -> None:
    time.sleep(RATE_LIMIT_DELAY)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


_retry = retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)


def _parse_github_url(url: str) -> Optional[tuple[str, str]]:
    """
    Supports:
    - https://github.com/owner/repo
    - https://github.com/owner/repo.git
    - http://www.github.com/owner/repo
    - git@github.com:owner/repo.git

    Does not support:
    - github.com without scheme
    - GitHub Enterprise
    """
    url = url.strip()

    ssh = re.match(r"git@github\.com:([^/\s]+)/([^/\s]+?)(?:\.git)?$", url)
    if ssh:
        return ssh.group(1), ssh.group(2)

    https = re.match(
        r"https?://(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?(?:[/#?].*)?$",
        url,
    )
    if https:
        return https.group(1), https.group(2)

    return None


def _classify_http_error(status_code: int) -> RepoStatus:
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limited"
    if status_code in (401, 403):
        return "forbidden"
    return "source_unavailable"


def _classify_exception(exc: Exception) -> RepoStatus:
    if isinstance(exc, httpx.HTTPStatusError):
        return _classify_http_error(exc.response.status_code)
    return "source_unavailable"


def _raise_for_retryable(response: httpx.Response, context: str) -> None:
    status = response.status_code
    if status == 429:
        logger.warning("GitHub rate limit hit: %s", context)
        response.raise_for_status()
    if status >= 500:
        logger.error("GitHub server error %s: %s", status, context)
        response.raise_for_status()


@_retry
def _request(path: str, *, context: str) -> httpx.Response:
    t0 = time.perf_counter()
    response = _client.get(f"{GITHUB_API_URL}{path}")
    latency = time.perf_counter() - t0
    logger.info("GitHub %s latency: %.2fs status: %d", context, latency, response.status_code)
    _raise_for_retryable(response, context)
    _rate_limit()
    return response


def _safe_json(response: httpx.Response) -> dict:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except ValueError:
        logger.warning("GitHub returned invalid JSON")
        return {}


def check_repo(url: str) -> RepoResponse:
    parsed = _parse_github_url(url)
    if not parsed:
        logger.warning("Invalid GitHub URL: %s", url)
        return {
            "source_available": True,
            "repo_found": None,
            "status": "invalid_url",
            "reason": "cannot_parse_url",
            "result": None,
        }

    owner, repo = parsed
    logger.info("GitHub check_repo: %s/%s", owner, repo)

    try:
        response = _request(f"/repos/{owner}/{repo}", context=f"repo {owner}/{repo}")
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
        logger.exception("GitHub repo request failed: %s/%s", owner, repo)
        status = _classify_exception(exc)
        return {
            "source_available": status != "source_unavailable",
            "repo_found": None,
            "status": status,
            "reason": str(exc),
            "result": None,
        }

    if response.status_code != 200:
        status = _classify_http_error(response.status_code)
        return {
            "source_available": status not in ("rate_limited", "source_unavailable"),
            "repo_found": False if status == "not_found" else None,
            "status": status,
            "reason": f"http_{response.status_code}",
            "result": None,
        }

    data = _safe_json(response)
    if not data:
        return {
            "source_available": True,
            "repo_found": None,
            "status": "source_unavailable",
            "reason": "invalid_response",
            "result": None,
        }

    default_branch = data.get("default_branch", "main")
    partial_data = False

    last_commit_date: Optional[str] = None
    try:
        commit_resp = _request(
            f"/repos/{owner}/{repo}/commits/{default_branch}",
            context=f"last commit {owner}/{repo}",
        )
        if commit_resp.status_code == 200:
            last_commit_date = (
                _safe_json(commit_resp).get("commit", {}).get("author", {}).get("date")
            )
        else:
            partial_data = True
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
        logger.warning("Could not fetch last commit for %s/%s", owner, repo)
        partial_data = True

    readme_exists: Optional[bool] = None
    try:
        readme_resp = _request(
            f"/repos/{owner}/{repo}/readme",
            context=f"readme {owner}/{repo}",
        )
        readme_exists = readme_resp.status_code == 200
        if readme_resp.status_code not in (200, 404):
            partial_data = True
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
        logger.warning("Could not check readme for %s/%s", owner, repo)
        partial_data = True

    has_releases: Optional[bool] = None
    try:
        releases_resp = _request(
            f"/repos/{owner}/{repo}/releases?per_page=1",
            context=f"releases {owner}/{repo}",
        )
        if releases_resp.status_code == 200:
            releases_data = releases_resp.json()
            has_releases = isinstance(releases_data, list) and len(releases_data) > 0
        else:
            partial_data = True
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
        logger.warning("Could not check releases for %s/%s", owner, repo)
        partial_data = True

    license_name: Optional[str] = None
    if isinstance(data.get("license"), dict):
        license_name = data["license"].get("spdx_id") or data["license"].get("name")

    result: RepoInfo = {
        "owner": owner,
        "repo": repo,
        "url": f"https://github.com/{owner}/{repo}",
        "stars": int(data.get("stargazers_count") or 0),
        "forks": int(data.get("forks_count") or 0),
        "open_issues": int(data.get("open_issues_count") or 0),
        "language": data.get("language"),
        "license": license_name,
        "readme_exists": readme_exists,
        "last_commit_date": last_commit_date,
        "topics": data.get("topics") or [],
        "is_archived": bool(data.get("archived", False)),
        "has_releases": has_releases,
        "default_branch": default_branch,
        "partial_data": partial_data,
    }

    return {
        "source_available": True,
        "repo_found": True,
        "status": "found",
        "reason": None,
        "result": result,
    }


def check_requirements(url: str) -> DependenciesResponse:
    parsed = _parse_github_url(url)
    if not parsed:
        return {
            "source_available": True,
            "repo_found": None,
            "status": "invalid_url",
            "reason": "cannot_parse_url",
            "dependencies": [],
            "source_file": None,
        }

    owner, repo = parsed
    logger.info("GitHub check_requirements: %s/%s", owner, repo)

    try:
        repo_response = _request(f"/repos/{owner}/{repo}", context=f"repo {owner}/{repo}")
    except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
        logger.warning("Could not verify repo %s/%s before dependency check: %s", owner, repo, exc)
        status = _classify_exception(exc)
        return {
            "source_available": status != "source_unavailable",
            "repo_found": None,
            "status": status,
            "reason": "repo_check_failed",
            "dependencies": [],
            "source_file": None,
        }

    if repo_response.status_code != 200:
        status = _classify_http_error(repo_response.status_code)
        return {
            "source_available": status not in ("rate_limited", "source_unavailable"),
            "repo_found": False if status == "not_found" else None,
            "status": status,
            "reason": f"repo_http_{repo_response.status_code}",
            "dependencies": [],
            "source_file": None,
        }

    candidates = [
        ("requirements.txt", _parse_requirements_txt),
        ("pyproject.toml", _parse_pyproject_toml),
        ("setup.py", _parse_setup_py),
    ]

    last_error_status: Optional[RepoStatus] = None

    for filename, parser in candidates:
        try:
            response = _request(
                f"/repos/{owner}/{repo}/contents/{filename}",
                context=f"{filename} {owner}/{repo}",
            )
            if response.status_code == 200:
                content_b64 = _safe_json(response).get("content", "")
                if content_b64:
                    content = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
                    deps = parser(content)
                    if deps:
                        logger.info(
                            "Found %d deps in %s for %s/%s",
                            len(deps), filename, owner, repo,
                        )
                        return {
                            "source_available": True,
                            "repo_found": True,
                            "status": "found",
                            "reason": None,
                            "dependencies": deps,
                            "source_file": filename,
                        }
            elif response.status_code != 404:
                last_error_status = _classify_http_error(response.status_code)

        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            logger.warning("Could not fetch %s for %s/%s: %s", filename, owner, repo, exc)
            last_error_status = _classify_exception(exc)

    if last_error_status in ("source_unavailable", "rate_limited", "forbidden"):
        return {
            "source_available": last_error_status != "source_unavailable",
            "repo_found": None,
            "status": last_error_status,
            "reason": "api_error_during_dependency_check",
            "dependencies": [],
            "source_file": None,
        }

    return {
        "source_available": True,
        "repo_found": True,
        "status": "not_found",
        "reason": "no_dependency_file_found",
        "dependencies": [],
        "source_file": None,
    }


def _parse_requirements_txt(content: str) -> list[str]:
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "-e", "http://", "https://", "git+")):
            continue
        name = re.split(r"[>=<!;\s\[]", line)[0].strip()
        if name and re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
            deps.append(name)
    return deps


def _parse_pyproject_toml(content: str) -> list[str]:
    """
    Supports PEP 621 and Poetry via tomllib (Python 3.11+).
    """
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            logger.warning("tomllib not available, skipping pyproject.toml")
            return []

    try:
        data = tomllib.loads(content)
    except Exception:
        logger.warning("Failed to parse pyproject.toml")
        return []

    deps: list[str] = []

    for dep in data.get("project", {}).get("dependencies", []):
        name = re.split(r"[>=<!;\s\[]", str(dep))[0].strip()
        if name:
            deps.append(name)

    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    if isinstance(poetry_deps, dict):
        for name in poetry_deps:
            if name.lower() != "python":
                deps.append(name)

    return deps


def _parse_setup_py(content: str) -> list[str]:
    """
    Heuristic regex for install_requires in setup.py.
    Covers only the simple case: install_requires=[...].
    """
    deps = []
    match = re.search(r"install_requires\s*=\s*\[([^\]]+)\]", content, re.DOTALL)
    if match:
        for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            name = re.split(r"[>=<!;\s\[]", item)[0].strip()
            if name:
                deps.append(name)
    return deps
