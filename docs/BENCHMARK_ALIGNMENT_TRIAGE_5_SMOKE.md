# Benchmark Alignment Triage: 5-Paper Smoke

This report summarizes deterministic benchmark alignment failures for the 5-paper
manual-verified smoke set. It is based on exported PDF-run workspaces, not the
URL run that reused cached v0.1 artifacts.

## Source Artifacts

- Workspaces: `/tmp/paperintel_eval_smoke_workspaces_v02_pdf.jsonl`
- Eval summary: `/tmp/paperintel_eval_smoke_summary_v02_pdf.json`
- Alignment audit: `/tmp/paperintel_benchmark_alignment_audit_v02_pdf.json`

## Global Summary

| Metric | Value |
| --- | ---: |
| Papers | 5 |
| Expected benchmark rows | 38 |
| Matched rows | 0 |
| Actual v0.2-like rows exported | yes |

Failure counts:

| Failure class | Count |
| --- | ---: |
| wrong_row_selected | 23 |
| wrong_method_variant | 8 |
| right_value_metric_wrong_task_dataset | 4 |
| right_task_dataset_wrong_metric | 2 |
| value_mismatch | 1 |

## Paper Summary

| Paper | Expected | Actual | Dominant failure | Failure counts |
| --- | ---: | ---: | --- | --- |
| SELF-RAG (`2310.11511`) | 8 | 2 | wrong_method_variant | wrong_method_variant=8 |
| FlashAttention (`2205.14135`) | 8 | 3 | wrong_row_selected | wrong_row_selected=5, right_value_metric_wrong_task_dataset=2, right_task_dataset_wrong_metric=1 |
| QLoRA (`2305.14314`) | 8 | 4 | wrong_row_selected | wrong_row_selected=8 |
| LoRA (`2106.09685`) | 8 | 3 | wrong_row_selected | wrong_row_selected=5, value_mismatch=1, right_value_metric_wrong_task_dataset=1, right_task_dataset_wrong_metric=1 |
| Attention (`1706.03762`) | 6 | 2 | wrong_row_selected | wrong_row_selected=5, right_value_metric_wrong_task_dataset=1 |

## Representative Failures

| Paper | Failure class | Expected source | Actual source | Expected row | Actual row | Fix type | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SELF-RAG | wrong_method_variant | Experiments / Table 2 / page 10 | Experiments / Table 2 / page 18 | PopQA, Accuracy 55.8%, conditions `SELF-RAG 13B`, `test set` | Open-domain QA, Natural Questions, Exact Match 93.8, conditions `Llama2-7B`, `Retrieve` | row_selection + prompt | Same table label family, but extracted baseline/retrieval variant instead of SELF-RAG downstream result. |
| SELF-RAG | wrong_method_variant | Experiments / Table 2 / page 10 | Experiments / Table 2 / page 18 | TriviaQA, Accuracy 69.3%, conditions `SELF-RAG 13B`, `test set` | Open-domain QA, Natural Questions, Exact Match 93.8, conditions `Llama2-7B`, `Retrieve` | row_selection + prompt | Method/model variant filtering is missing; all 8 expected rows fail this way. |
| FlashAttention | wrong_row_selected | Experiments / Table 3 / page 8 | Benchmarking Attention / no table | GPT-2 attention runtime, Long Range Arena, Attention Speedup 7.6x, conditions `FlashAttention`, `GPT-2`, `A100 GPU` | GPT-2 training, Long Range Arena, Perplexity 0.7, conditions `FlashAttention` | row_selection + context | Extractor selected a text/perplexity row instead of speedup/runtime table row. |
| FlashAttention | right_task_dataset_wrong_metric | Experiments / Table 2 / page 7 | Benchmarking Attention / no table | GPT-2 training, GPT-2, Training Speedup 3.0x, conditions `FlashAttention`, `GPT-2 medium`, `OpenWebText` | GPT-2 training, Long Range Arena, Perplexity 0.7, conditions `FlashAttention` | row_selection + metric_priority | Task is close, but metric/value target is wrong. |
| QLoRA | wrong_row_selected | Results / Table 6 / page 10 | Evaluation / Table 5 / page 6 | Vicuna, Relative ChatGPT Score 99.3%, conditions `Guanaco 65B`, `GPT-4 evaluation` | MMLU benchmark evaluation, MMLU, Accuracy 63.4%, conditions `QLORA`, `65B` | row_selection + table_priority | Extractor selected MMLU table instead of Vicuna/GPT-4 evaluation target. |
| QLoRA | wrong_row_selected | Results / Table 1 / page 2 | Evaluation / Table 5 / page 6 | 65B finetuning memory, GPU Memory Requirement 48.0, conditions `QLoRA`, `single GPU`, `65B model` | MMLU benchmark evaluation, MMLU, Accuracy 63.4%, conditions `QLORA`, `65B` | row_selection + metric_priority | Memory/resource rows are not prioritized over downstream MMLU rows. |
| LoRA | wrong_row_selected | 5 EMPIRICAL EXPERIMENTS / Table 2 / page 6 | Experiments / Table 4 / page 8 | GPT-3 175B finetuning, GLUE, Parameter Reduction 10000x, conditions `LoRA`, `r=4`, `query and value matrices` | SAMSum, ROUGE-L 45.9, conditions `GPT-3`, `LoRA`, `4.7M` | row_selection + metric_priority | Extractor selected downstream quality rows, while golden targets parameter/resource rows. |
| LoRA | value_mismatch | 5 EMPIRICAL EXPERIMENTS / Table 4 / page 8 | Experiments / Table 4 / page 8 | WikiSQL, Accuracy 74.0%, conditions `LoRA`, `GPT-3`, `37.7M trainable params` | WikiSQL, Accuracy 73.4%, conditions `GPT-3`, `LoRA`, `4.7M` | row_selection + condition_filtering | Same table/task/metric, but wrong LoRA parameter setting. |
| Attention | right_value_metric_wrong_task_dataset | Results / Table 2 / page 8 | Results / Table 2 / page 10 | WMT14, BLEU 28.4, conditions `Transformer big`, `English-to-German` | English-to-German translation, newstest2014, BLEU 28.4, conditions `Transformer`, `big` | evaluator_normalization or golden_mapping | Value/metric/conditions align; task/dataset contract differs. This is a mapping issue, not row selection. |
| Attention | wrong_row_selected | Results / Table 2 / page 8 | Results / Table 2 / page 10 | WMT14, BLEU 41.0, conditions `Transformer big`, `English-to-French` | English-to-French translation, newstest2014, BLEU 41.8, conditions `Transformer`, `big` | golden_issue + row_selection | Same table and direction, but expected value differs from extracted value; verify golden label before changing evaluator. |

## Decision Matrix

| Priority | Area | Evidence | Suggested next action |
| ---: | --- | --- | --- |
| 1 | SELF-RAG method variant filtering | 8/8 failures are `wrong_method_variant` | Add row-selection rules requiring proposed method/model variant coverage before accepting rows. |
| 2 | Attention task/dataset mapping | Exact value/metric match for WMT14 De-En, but task/dataset differs | Decide canonical mapping for WMT14/newstest2014 and translation direction. Prefer evaluator normalization only after golden row review. |
| 3 | LoRA/QLoRA/FlashAttention row selection | Dominant failures are wrong table/metric target | Add metric/table priority rules or target hints from golden-like benchmark families. |
| 4 | Golden spot-check | Attention En-Fr expected 41.0 vs extracted 41.8; QLoRA/LoRA target choices vary | Review representative questionable labels before broad agent changes. |

## Conclusion

The v0.2 benchmark contract is technically working: PDF smoke workspaces contain
dataset-aware, source-aware v0.2 rows. The benchmark score remains 0/38 because
the extractor chooses rows that do not align with the current golden targets.

The next code change should target row selection, not parser/schema. The cleanest
first target is SELF-RAG method variant filtering because the failure is isolated
to one paper and one dominant class.
