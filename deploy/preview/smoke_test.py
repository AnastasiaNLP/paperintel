"""Exercise one authenticated discovery, queued paper analysis, and QA path."""

from __future__ import annotations

import os
import sys
import time

import httpx


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    base_url = os.environ.get("PAPERINTEL_BASE_URL", "").rstrip("/")
    token = os.environ.get("PAPERINTEL_API_AUTH_TOKEN", "")
    timeout_seconds = int(os.environ.get("PAPERINTEL_SMOKE_TIMEOUT_SECONDS", "1200"))
    require(bool(base_url), "Set PAPERINTEL_BASE_URL, including https://.")
    require(bool(token), "Set PAPERINTEL_API_AUTH_TOKEN.")
    require(timeout_seconds >= 60, "PAPERINTEL_SMOKE_TIMEOUT_SECONDS must be >= 60.")

    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        health = client.get("/health")
        require(
            health.status_code == 200,
            f"Health check failed: {health.status_code} {health.text}",
        )
        health_body = health.json()
        require(
            health_body.get("status") == "healthy",
            f"Dependencies are unhealthy: {health_body}",
        )
        print("health: ok")

        created = client.post("/sessions", json={"persona": "engineer"})
        require(
            created.status_code == 200,
            f"Session creation failed: {created.status_code} {created.text}",
        )
        session_id = created.json()["id"]
        print(f"session: {session_id}")

        discovery = client.post(
            f"/sessions/{session_id}/jobs/discover",
            json={"topic": "retrieval augmented generation recent research"},
        )
        require(
            discovery.status_code == 202,
            "Discovery enqueue failed: "
            f"{discovery.status_code} {discovery.text}",
        )
        discovery_job = poll_job(client, discovery.json()["id"], deadline)
        discovery_result = discovery_job.get("result_json") or {}
        require(
            discovery_result.get("phase") == "selection",
            f"Discovery did not reach selection: {discovery_job}",
        )
        candidate_count = discovery_result.get("discovery_candidate_count") or 0
        require(candidate_count > 0, "Discovery returned no candidates.")
        print(f"discovery: ok ({candidate_count} candidates)")

        selected = client.post(
            f"/sessions/{session_id}/select",
            json={"selection": "use 1"},
        )
        require(
            selected.status_code == 200,
            f"Candidate selection failed: {selected.status_code} {selected.text}",
        )
        selected_ids = selected.json().get("selected_candidate_ids") or []
        require(bool(selected_ids), f"No candidate selected: {selected.json()}")
        print(f"selection: ok ({len(selected_ids)} paper)")

        analysis = client.post(f"/sessions/{session_id}/jobs/analyze-selected")
        require(
            analysis.status_code == 202,
            f"Analysis enqueue failed: {analysis.status_code} {analysis.text}",
        )
        analysis_job = poll_job(client, analysis.json()["id"], deadline)
        analysis_result = analysis_job.get("result_json") or {}
        require(
            analysis_result.get("phase") == "qa",
            f"Paper analysis did not reach QA: {analysis_job}",
        )
        workspaces = client.get(f"/sessions/{session_id}/workspaces")
        require(
            workspaces.status_code == 200,
            f"Workspace query failed: {workspaces.status_code} {workspaces.text}",
        )
        workspace_rows = workspaces.json().get("workspaces") or []
        require(
            any(row.get("has_finalized_report") for row in workspace_rows),
            "No finalized report was persisted.",
        )
        print("queued analysis and persisted report: ok")

        qa = client.post(
            f"/sessions/{session_id}/ask",
            json={"question": "What is the paper's main contribution?"},
        )
        require(qa.status_code == 200, f"QA request failed: {qa.status_code} {qa.text}")
        qa_result = qa.json()
        require(
            bool(qa_result.get("response_text", "").strip()),
            "QA response is empty.",
        )
        require(
            bool(qa_result.get("referenced_paper_ids")),
            "QA response is not grounded in a paper.",
        )
        print("grounded QA: ok")

    print(f"PREVIEW_SMOKE_OK session={session_id}")
    return 0


def poll_job(client: httpx.Client, job_id: str, deadline: float) -> dict:
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        require(
            response.status_code == 200,
            f"Job status failed: {response.status_code} {response.text}",
        )
        payload = response.json()
        status = payload.get("status")
        if status == "succeeded":
            return payload
        if status in {"failed", "canceled"}:
            raise RuntimeError(
                f"Job {job_id} ended as {status}: {payload.get('error_json')}"
            )
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} did not finish before the smoke-test deadline.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREVIEW_SMOKE_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
