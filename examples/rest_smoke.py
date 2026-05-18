"""Smoke test PaperIntel through the REST API.

Prerequisites:
- Postgres and Qdrant running: docker compose up -d postgres qdrant
- REST API running: see docs/QUICKSTART.md

Usage:
    python examples/rest_smoke.py
"""

from __future__ import annotations

import sys
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"
PAPER_URL = "https://arxiv.org/abs/1706.03762"
QUESTION = "What is the main contribution of this paper?"


def main() -> int:
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5)
    except httpx.ConnectError:
        print(f"ERROR: API is not reachable at {BASE_URL}")
        print("Start it first: see docs/QUICKSTART.md")
        return 1

    if health.status_code == 503:
        print("WARNING: API is up but degraded:")
        print(health.text)
    elif health.status_code >= 400:
        print(f"ERROR: health check failed with HTTP {health.status_code}")
        print(health.text)
        return 1

    print("Creating session...")
    session_response = httpx.post(
        f"{BASE_URL}/sessions",
        json={"persona": "engineer"},
        timeout=10,
    )
    session_response.raise_for_status()
    session_id = session_response.json()["id"]
    print(f"  session_id = {session_id}")

    print("Analyzing paper (~1 minute)...")
    start = time.time()
    analyze_response = httpx.post(
        f"{BASE_URL}/sessions/{session_id}/analyze",
        json={"paper_url": PAPER_URL},
        timeout=180,
    )
    if not _check_response(analyze_response, "analyze paper"):
        return 1
    print(f"  done in {time.time() - start:.0f}s")

    print("Asking question...")
    answer_response = httpx.post(
        f"{BASE_URL}/sessions/{session_id}/ask",
        json={"question": QUESTION},
        timeout=120,
    )
    if not _check_response(answer_response, "ask question"):
        return 1
    answer = answer_response.json()

    print("\n--- Answer ---")
    print(answer["response_text"])
    print("\nReferenced papers:", ", ".join(answer.get("referenced_paper_ids") or []))
    print("Citations:", len(answer.get("citations") or []))

    print("\nLoading persisted workspaces...")
    workspaces_response = httpx.get(
        f"{BASE_URL}/sessions/{session_id}/workspaces",
        timeout=10,
    )
    if not _check_response(workspaces_response, "load persisted workspaces"):
        return 1
    workspaces = workspaces_response.json().get("workspaces") or []
    print("Workspaces:", len(workspaces))
    if workspaces:
        first = workspaces[0]
        print("First workspace:", first.get("paper_id"), first.get("title"))
    return 0


def _check_response(response: httpx.Response, action: str) -> bool:
    if response.status_code < 400:
        return True
    print(f"ERROR: failed to {action}; HTTP {response.status_code}")
    try:
        print(response.json())
    except ValueError:
        print(response.text)
    return False


if __name__ == "__main__":
    sys.exit(main())
