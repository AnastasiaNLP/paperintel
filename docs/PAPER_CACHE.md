# Paper Cache

PaperIntel reuses completed paper analysis across sessions by cloning durable
analysis artifacts. This is a workspace-level cache, not a separate
`paper_cache` table.

## Reusable Identity

Reusable analysis is keyed by:

- arXiv URL analysis: `(paper_id, pipeline_version)`
- registered PDF analysis: `(source_pdf_hash, pipeline_version)`

`pipeline_version` is the cache invalidation key. Prompt, model, schema,
extraction, or report-contract changes that make old results incompatible must
use a new `pipeline_version` before reuse is allowed.

## Reuse Activation Contract

A cache hit must make the target session operationally equivalent to a
successful analysis:

- clone a ready `PaperWorkspace` into the target session
- clone retrieval chunks into the target session
- upsert the target-session chunk payloads into Qdrant before the paper becomes
  active
- add the paper id to `session.active_paper_ids`
- set the session phase to `qa`
- for PDF-derived reuse, preserve the session/workspace blob references
- return a normal analysis-shaped result with `paper_workspace:<id>` artifact
  refs

The cloned workspace may include finalized report JSON, markdown report, method
extraction JSON, benchmark JSON, and readiness JSON. Cache reuse does not clone
turns, selected candidates, comparisons, synthesis outputs, or other
conversation-specific state.

## Public Result Metadata

Reusable analysis is exposed through structured result metadata:

```json
{
  "metadata": {
    "analysis_reused": true,
    "reuse_source": "paper_id"
  }
}
```

`reuse_source` is `"paper_id"` for arXiv URL reuse and `"pdf_hash"` for
registered PDF reuse. REST responses and workflow job `result_json` preserve
this metadata. MCP job output includes neutral summary lines for these fields;
normal MCP analysis text remains analysis-shaped and does not mention internal
cache mechanics.

## Miss and Conflict Policy

Cache lookup is opportunistic for cross-session reuse. A miss or missing cache
dependencies falls back to normal analysis. Same-session incompatible workspace
state remains an explicit conflict and must not be overwritten silently.

Incomplete cache sources and empty cloned chunk sets are explicit failures for
the cache path. They are not treated as successful reuse because retrieval would
not be operationally equivalent to a fresh analysis.

## Current Limitations

- There is no separate `paper_cache` table or global cache manifest.
- Cache invalidation is manual through `pipeline_version`.
- There is no automated cleanup for cloned workspaces, cloned chunks, Qdrant
  vectors, or blob references.
- There is no distributed coordination around concurrent reuse attempts beyond
  the repository idempotency/conflict rules.
- Cached reports assume report generation is not persona-specific. If report
  generation becomes persona-specific, this contract must be revised before
  report reuse remains enabled.

## Verification Checklist

Run the unit transport/service checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_paperintel_service.py \
  tests/unit/test_workflow_worker.py \
  tests/unit/test_rest_schemas.py \
  tests/unit/test_mcp_tools.py -q
```

Run REST/MCP integration checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/integration/test_rest_api.py \
  tests/integration/test_mcp_server.py -q
```

With Postgres available through `PAPERINTEL_TEST_DATABASE_URL`, run the
repository cache checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/integration/test_postgres_repositories.py \
  -k 'paper_workspace_repository or paper_chunk_repository' -q
```
