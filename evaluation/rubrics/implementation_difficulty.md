# Rubric: Implementation Difficulty

Evaluate whether `engineer_report.implementation_difficulty` is consistent with
the method, dependencies, readiness artifacts, and stated limitations.

## Inputs

- `engineer_report.implementation_difficulty`
- `engineer_report.practical_implications`
- `method_extraction`
- `production_readiness`
- relevant paper evidence supplied to the judge

## Valid Difficulty Labels

- `trivial`
- `moderate`
- `significant`
- `research_only`

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: difficulty label is well calibrated and explicitly supported.
- `0.75`: label is reasonable with minor missing nuance.
- `0.50`: label is plausible but weakly justified.
- `0.25`: label is materially too easy or too hard for the evidence.
- `0.0`: label is contradicted by the evidence.

## Criteria

The judge should consider:

- whether open code or framework integration exists;
- how much low-level systems work, training, data preparation, or infrastructure
  is required;
- whether the method requires specialized hardware or custom kernels;
- whether the paper leaves essential implementation details unresolved.

The judge should not infer difficulty from later ecosystem tooling unless that
tooling is included in the supplied evidence.

