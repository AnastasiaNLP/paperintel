# Chunking Strategy

Retrieval is built around durable paper chunks and Qdrant vectors. Chunking runs
after the analysis report has been finalized:

```text
ingestion -> extraction -> benchmark -> readiness -> report
-> evidence_critic -> report_finalize -> chunk_and_index
-> [next paper | comparator | END]
```

Indexing is non-fatal. If `chunk_and_index` fails, the paper has still been
analyzed and finalized.

## Embedding Contract

- Model: `text-embedding-3-small`
- Dimensions: `1536`
- The chunk metadata stores the expected model and dimensions so Qdrant
  collections can be validated before indexing.

## Paper Identity

`paper_id` is the canonical paper identifier, normally the arXiv id without a
version suffix, for example `2310.06825`.

This is intentionally not a session-scoped UUID. Chunks should be reusable
across sessions and pipeline reruns. If arXiv metadata is unavailable, the
chunking service may use a deterministic fallback derived from the input URL or
paper index, but that fallback is treated as lower-quality provenance.

## Chunk Types

`ChunkType` is part of the retrieval domain model:

- `abstract`
- `text`
- `table`
- `equation`
- `figure`
- `caption`
- `reference`

The current chunking service emits `abstract` and `text` chunks. Tables,
equations, figures, and captions are represented in the contract for future
extraction but are not parsed yet.

## Size And Overlap

The chunking service uses character-based chunking to avoid adding tokenizer
dependencies.

- Target chunk size: `2400` characters
- Overlap: `300` characters
- Minimum chunk size: `200` characters

The service splits on paragraph boundaries when possible, then falls back to
character windows for long paragraphs. This gives deterministic chunks without
requiring model-specific tokenizers.

## Page And Section Tracking

When `text_by_page` is available, the chunking service chunks per page and
preserves:

- `page_start`
- `page_end`
- `char_start`
- `char_end`

Section detection is conservative. A short line with title-like capitalization
or numbering can become `section_title`. The section title is stored as chunk
location metadata and can later be used by citations.

## Evidence Artifacts

`EvidenceArtifact` supports tables, equations, figures, page images, PDFs, and
raw text through `storage_ref`. The current implementation does not upload
artifacts to S3 or MinIO. The field exists so retrieval chunks can later connect
to durable objects without changing the retrieval contract.

## Deterministic IDs

Chunk ids are deterministic:

```text
{paper_id}:chunk:{chunk_index}
```

This makes reruns idempotent and lets later storage/vector layers upsert safely.
