# Rubric: Action Reasoning

Evaluate whether `engineer_report.action_reasoning` explains the recommendation
with evidence-grounded, engineer-useful reasoning.

## Inputs

- `engineer_report.recommended_action`
- `engineer_report.action_reasoning`
- `method_extraction`
- `benchmarks`
- `production_readiness`
- relevant paper evidence supplied to the judge

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: reasoning clearly connects action to evidence, readiness, benchmarks,
  implementation constraints, and limitations.
- `0.75`: reasoning is mostly grounded but misses one important factor.
- `0.50`: reasoning is partly grounded but generic or incomplete.
- `0.25`: reasoning is mostly vague, unsupported, or misses major evidence.
- `0.0`: reasoning is misleading, contradicted, or based on external knowledge.

## Criteria

The judge should reward reasoning that:

- cites concrete artifact facts such as code availability, framework integration,
  benchmark coverage, dependencies, and limitations;
- explains trade-offs rather than only restating the action;
- distinguishes "promising research" from "ready to implement".

The judge should penalize reasoning that:

- makes confident production claims without artifact support;
- ignores missing benchmarks or missing code;
- uses broad praise without actionable engineering implications.

