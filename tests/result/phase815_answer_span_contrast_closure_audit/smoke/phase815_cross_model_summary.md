# Phase 815 Answer Span And Contrast Closure Audit (smoke)

- Source: Phase 814 saved rows; no new model forward pass.
- Boundary: answer span proxy closure requires answer-class closure, answer-unit closure, contrast-class cleared, and surface-valid answer unit.

## Model Summary

| model | rows | span_proxy | answer_class | answer_unit | contrast_cleared | raw_canon | strict_token | unit_closed_answer_not_closed | answer_closed_unit_fragmented | multi_token_target | labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3 | 29 | 12 | 29 | 12 | 29 | 0 | 0 | 0 | 17 | 0 | `{"answer_closed_unit_fragmented": 17, "span_proxy_closed_no_strict_token": 12}` |
| glm4 | 39 | 39 | 39 | 39 | 39 | 0 | 0 | 0 | 0 | 0 | `{"span_proxy_closed_no_strict_token": 39}` |
| deepseek7b | 29 | 27 | 27 | 29 | 29 | 0 | 0 | 2 | 0 | 0 | `{"span_proxy_closed_no_strict_token": 27, "unit_closed_answer_not_closed": 2}` |

## Unit Closed But Answer Not Closed Blockers

| model | dominant blocker classes |
|---|---|
| qwen3 | `{}` |
| glm4 | `{}` |
| deepseek7b | `{"semantic_or_lexical_competitor": 2}` |

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
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| qwen3 | p765_0041_commonsense_question_plant:oak:grows_on_tree | ` Yes` | 0 | 1 | 0 | 1 | 0 | 0 | ` No`/candidate_list_or_case_value | `answer_closed_unit_fragmented` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| glm4 | p765_0051_commonsense_question_plant:wheat:edible | ` Yes` | 1 | 1 | 1 | 1 | 0 | 0 | ` wheat`/candidate_list_or_case_value | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
| deepseek7b | p765_0075_commonsense_question_tool:hammer:edible | ` No` | 1 | 1 | 1 | 1 | 0 | 0 | ` \`/punctuation | `span_proxy_closed_no_strict_token` |
