# Rubric: QA Faithfulness

Evaluate whether a QA answer is faithful to the supplied evidence and citations.

## Inputs

- user question
- answer text
- cited evidence chunks
- citation metadata
- optional golden QA case keywords and required citation paper IDs

## Scoring

Score from `0.0` to `1.0`:

- `1.0`: answer is fully supported by cited evidence, directly answers the
  question, and uses citations accurately.
- `0.75`: answer is mostly supported with minor omissions or mild overstatement.
- `0.50`: answer is partially supported but incomplete, vague, or weakly cited.
- `0.25`: answer contains material unsupported claims or citation misuse.
- `0.0`: answer is mostly unsupported, contradicted by evidence, or cites the
  wrong paper.

## Criteria

The judge should check:

- whether each important answer claim is supported by cited chunks;
- whether citations point to the required paper IDs when specified;
- whether benchmark values, method names, and limitations are copied faithfully;
- whether the answer avoids adding outside facts not present in evidence.

The judge should not reward fluent answers that lack evidence support. It should
also not require facts that are outside the user question or golden QA case.

