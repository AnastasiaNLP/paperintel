# Rubric: Synthesis Persona Fit

Evaluate whether a Synthesis Agent response is tailored to the requested
persona.

## Inputs

- persona: `engineer`, `researcher`, or `techlead`
- durable `PaperWorkspace` artifacts
- synthesis report and rendered response

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: response consistently matches the requested persona's decision needs.
- `0.75`: mostly persona-appropriate with minor generic sections.
- `0.50`: partially tailored but could fit any audience.
- `0.25`: mostly mismatched to the requested persona.
- `0.0`: ignores or contradicts the requested persona.

## Criteria

Engineer synthesis should emphasize implementation and dependencies. Researcher
synthesis should emphasize novelty and evidence quality. Techlead synthesis
should emphasize maturity, adoption risk, ROI, and sequencing.
