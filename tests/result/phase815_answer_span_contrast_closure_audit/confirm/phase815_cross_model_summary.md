# Phase 815 Answer Span And Contrast Closure Audit (confirm)

- Source: Phase 814 saved rows; no new model forward pass.
- Boundary: answer span proxy closure requires answer-class closure, answer-unit closure, contrast-class cleared, and surface-valid answer unit.

## Model Summary

| model | rows | span_proxy | answer_class | answer_unit | contrast_cleared | raw_canon | strict_token | unit_closed_answer_not_closed | answer_closed_unit_fragmented | multi_token_target | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 642 | 97 | 486 | 97 | 642 | 0 | 0 | 0 | 389 | 214 | `{"answer_closed_unit_fragmented": 389, "span_proxy_closed_no_strict_token": 97, "global_competition_unclosed": 156}` |
| glm4 | 428 | 323 | 428 | 323 | 428 | 0 | 0 | 0 | 105 | 0 | `{"span_proxy_closed_no_strict_token": 323, "answer_closed_unit_fragmented": 105}` |
| deepseek7b | 424 | 1 | 1 | 423 | 419 | 1 | 0 | 422 | 0 | 212 | `{"unit_closed_answer_not_closed": 422, "span_proxy_closed_no_strict_token": 1, "contrast_interference": 1}` |

## Unit Closed But Answer Not Closed Blockers

| model | dominant blocker classes |
|---|---|
| qwen3 | `{}` |
| glm4 | `{}` |
| deepseek7b | `{"semantic_or_lexical_competitor": 319, "echo_token": 103}` |

## Best Rows

| model | case | unit | span_proxy | answer_class | answer_unit | contrast_clear | raw_canon | strict | first_non_answer | label |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 0 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
| deepseek7b | p765_0103_commonsense_question_abstract:justice:category | ` Abstract` | 0 | 0 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `unit_closed_answer_not_closed` |
