# Rubric: Comparison Grounding

Evaluate whether comparison trade-offs are grounded in durable PaperIntel
artifacts rather than unsupported narrative claims.

## Inputs

- selected paper ids
- durable `PaperWorkspace` artifacts
- extracted methods, benchmarks, readiness, and engineer reports
- comparison output

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: trade-offs are directly supported by supplied artifact fields.
- `0.75`: mostly grounded, with minor unsupported generalization.
- `0.50`: partially grounded but missing important evidence links.
- `0.25`: many claims are unsupported or overstate weak evidence.
- `0.0`: comparison relies mainly on outside knowledge or invented claims.

## Criteria

The judge should check benchmark values, method names, readiness claims, and
limitations against the supplied artifacts only.
