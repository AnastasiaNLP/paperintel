# Evaluation Rubrics

These rubrics are the versioned source of truth for the future LLM-judge
evaluation layer. They are executable prompt artifacts: a judge runner should
load these files directly rather than reimplementing the criteria from prose
documentation.

## Scope

The deterministic runner remains the CI gate. It checks stable facts and
coverage:

- benchmark rows
- readiness fields
- method/report keyword coverage

The judge runner is a gauge. It should report approximate quality signals for
free-text reasoning and citation faithfulness, but it should not fail normal CI
based on routine score variance.

## Scoring Scale

All rubrics use a `0.0` to `1.0` score:

- `0.0`: unusable or contradicted by evidence
- `0.25`: mostly unsupported or materially misleading
- `0.50`: partially supported, incomplete, or ambiguous
- `0.75`: mostly supported with minor omissions
- `1.0`: well supported, specific, and faithful to evidence

Rubric runners may emit intermediate scores, but reports should preserve the raw
score and rubric filename/version context.

## Evidence Policy

Unless a specific rubric says otherwise, judge decisions must be based only on:

- the persisted workspace artifact being evaluated;
- the paper text/evidence snippets supplied to the judge;
- the golden record fields relevant to the task.

The judge must not reward later ecosystem knowledge, popularity, or production
adoption unless that information is present in the supplied evidence.

## Non-Determinism Policy

LLM-judge scores are expected to vary across runs. They are useful for trend
tracking and qualitative review, not strict CI gating. If a future workflow adds
blocking thresholds, those thresholds must be broad and explicitly documented.

