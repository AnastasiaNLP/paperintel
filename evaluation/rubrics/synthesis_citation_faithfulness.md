# Rubric: Synthesis Citation Faithfulness

Evaluate whether synthesis citations support the claims they are attached to
and only cite selected papers.

## Inputs

- selected paper ids
- durable `PaperWorkspace` artifacts
- synthesis citations
- synthesis report and rendered response

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: citations accurately support the cited claims and refer only to
  selected papers.
- `0.75`: citations are mostly accurate with minor vague support.
- `0.50`: citations point to selected papers but only weakly support claims.
- `0.25`: citations are often mismatched, vague, or misleading.
- `0.0`: citations cite unrelated papers or materially contradict evidence.

## Criteria

The judge should not reward fluent synthesis if citations do not ground the
claims in the supplied artifacts.
