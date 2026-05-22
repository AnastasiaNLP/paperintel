# Rubric: Recommended Action

Evaluate whether the engineer-facing `recommended_action` is justified by the
paper evidence and the extracted artifacts.

## Inputs

- `engineer_report.recommended_action`
- `engineer_report.action_reasoning`
- `method_extraction`
- `benchmarks`
- `production_readiness`
- relevant paper evidence supplied to the judge

## Valid Actions

- `implement_now`
- `prototype`
- `watch`
- `skip`

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: action is strongly justified by readiness, benchmark evidence,
  implementation risk, and paper limitations.
- `0.75`: action is reasonable but misses one material caveat.
- `0.50`: action is plausible but under-justified or too generic.
- `0.25`: action conflicts with important readiness or benchmark evidence.
- `0.0`: action is unsupported, contradicted, or relies on external knowledge.

## Criteria

The judge should reward:

- conservative recommendations when code, benchmarks, or reproducibility are
  weak;
- stronger recommendations when open artifacts, clear implementation paths, and
  relevant benchmark evidence exist;
- explicit handling of paper limitations and deployment risk.

The judge should penalize:

- `implement_now` without strong evidence of usable artifacts and benchmark
  support;
- `skip` when the paper provides clear usable artifacts and directly relevant
  results;
- recommendations based on popularity or later ecosystem adoption not present in
  the supplied evidence.

