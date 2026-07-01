# Phase 815 Answer Span And Contrast Closure Audit (main)

- Source: Phase 814 saved rows; no new model forward pass.
- Boundary: answer span proxy closure requires answer-class closure, answer-unit closure, contrast-class cleared, and surface-valid answer unit.

## Model Summary

| model | rows | span_proxy | answer_class | answer_unit | contrast_cleared | raw_canon | strict_token | unit_closed_answer_not_closed | answer_closed_unit_fragmented | multi_token_target | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 280 | 70 | 199 | 70 | 280 | 0 | 0 | 0 | 129 | 140 | `{"answer_closed_unit_fragmented": 129, "span_proxy_closed_no_strict_token": 70, "global_competition_unclosed": 81}` |
| glm4 | 280 | 208 | 280 | 208 | 280 | 0 | 0 | 0 | 72 | 0 | `{"span_proxy_closed_no_strict_token": 208, "answer_closed_unit_fragmented": 72}` |
| deepseek7b | 276 | 3 | 3 | 276 | 274 | 0 | 0 | 273 | 0 | 138 | `{"unit_closed_answer_not_closed": 273, "span_proxy_closed_no_strict_token": 3}` |

## Unit Closed But Answer Not Closed Blockers

| model | dominant blocker classes |
|---|---|
| qwen3 | `{}` |
| glm4 | `{}` |
| deepseek7b | `{"semantic_or_lexical_competitor": 206, "echo_token": 63, "punctuation": 4}` |

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
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` `/whitespace_or_newline | `span_proxy_closed_no_strict_token` |
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
