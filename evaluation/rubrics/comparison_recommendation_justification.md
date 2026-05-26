# Rubric: Comparison Recommendation Justification

Evaluate whether comparison recommendations or winner logic are justified by
the available evidence.

## Inputs

- selected paper ids
- durable `PaperWorkspace` artifacts
- comparison recommendations
- overall winner fields and winner basis

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: recommendations follow clearly from benchmark, readiness, and method
  evidence.
- `0.75`: recommendations are reasonable but miss a material caveat.
- `0.50`: recommendations are plausible but generic or thinly justified.
- `0.25`: recommendations conflict with important supplied evidence.
- `0.0`: winner or recommendations are unsupported, arbitrary, or invented.

## Criteria

The judge should reward explicit uncertainty when evidence is weak and should
penalize unsupported winner claims.
