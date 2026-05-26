# Rubric: Comparison Balance

Evaluate whether a Comparison Analyst output covers all compared papers fairly.

## Inputs

- selected paper ids
- durable `PaperWorkspace` artifacts
- `comparison_report_json`
- rendered comparison markdown

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: all compared papers are represented fairly, with clear strengths,
  weaknesses, and trade-offs.
- `0.75`: all papers are covered, but one paper receives thinner treatment.
- `0.50`: coverage is uneven or one paper is only mentioned superficially.
- `0.25`: comparison strongly favors or ignores papers without evidence basis.
- `0.0`: one or more selected papers are effectively absent from the comparison.

## Criteria

The judge should reward balanced treatment across the selected papers. It should
not require a winner when evidence supports `no_clear_winner`.
