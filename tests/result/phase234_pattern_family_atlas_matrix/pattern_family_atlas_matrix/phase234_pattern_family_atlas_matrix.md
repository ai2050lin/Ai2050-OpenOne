# Phase 234 Pattern Family Atlas Matrix

families: 9
modes: 72
seed_test_cases: 36

## Families

| priority | family | modes | mechanism |
| ---: | --- | ---: | --- |
| 1 | content_knowledge / 内容知识模式族 | 6 | PromptPattern -> ObjectRelationState -> MLPProduct -> ResidualWrite -> TargetPressure -> CompetitorPressure -> ReadoutWinner |
| 1 | output_protocol / 输出协议模式族 | 8 | InstructionFrame -> AnswerAnchor -> StepCondition -> GatedActivation -> ReadoutRegimeSelection |
| 2 | reasoning_constraint / 推理约束模式族 | 9 | ReasonTrigger -> ConstraintState -> StateMaintainPath -> ReadoutCompetition |
| 2 | syntax_structure / 语法结构模式族 | 9 | SyntaxTrigger -> BoundaryState -> ReadoutRegimeSelection -> ContinuationStopFormat |
| 2 | language_action / 语言动作模式族 | 8 | TaskFrame -> ActionMode -> OutputProtocol -> RolloutPattern |
| 3 | cross_lingual / 跨语言模式族 | 7 | LanguageSpecificInput -> SharedConstraintState -> OutputLanguageReadout |
| 1 | readout_competition / 竞争读出模式族 | 11 | TargetPressure -> CompetitorSource -> CompetitorPressureField -> WinnerRegime -> SecondCompetitorTakeover |
| 2 | state_drift / 状态维持与漂移模式族 | 7 | InitialState -> StateMaintainPath -> StepCondition -> RegimeSwitch -> DriftType |
| 1 | closure / 闭合模式族 | 7 | AnswerCorrect & PatternMatched & BoundaryStable & DoneStateStable & ModelStopExecuted & NoDrift |

## Test Levels

| level | id | metrics |
| ---: | --- | --- |
| 1 | behavior | answer_correct, pattern_match, drift_type, output_length, stop_type |
| 2 | prompt_trigger | prompt_variant, anchor_removed, instruction_replaced, trigger_token |
| 3 | gate_up_product | gate_delta, up_delta, product_delta, down_out_delta, product_recompute_error |
| 4 | residual_state | residual_direction, projection_shift, state_propagation |
| 5 | readout_competition | target_logit, target_rank, winner_regime, second_competitor, remaining_gap |
| 6 | competitor_source | source_delta, source_suppression, winner_changed, margin_help |
| 7 | rollout | token_sequence, regime_sequence, drift_step, stop_step |
| 8 | closure | model_close, done_state_stable, model_stop_executed, no_drift |

## Seed Cases

| case | family | mode | protocol | target | prompt |
| --- | --- | --- | --- | --- | --- |
| pf_case_0001 | content_knowledge | object_relation_value | short | red | What is the color of apple?\nAnswer with one word.\nAnswer: |
| pf_case_0002 | content_knowledge | object_relation_value | explain | red | What is the color of apple?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0003 | content_knowledge | object_relation_value | repeat | red | What is the color of apple?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0004 | content_knowledge | object_relation_value | list | red | What is the color of apple?\nAnswer as a short list.\nAnswer: |
| pf_case_0005 | content_knowledge | object_relation_value | short | yellow | What is the color of banana?\nAnswer with one word.\nAnswer: |
| pf_case_0006 | content_knowledge | object_relation_value | explain | yellow | What is the color of banana?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0007 | content_knowledge | object_relation_value | repeat | yellow | What is the color of banana?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0008 | content_knowledge | object_relation_value | list | yellow | What is the color of banana?\nAnswer as a short list.\nAnswer: |
| pf_case_0009 | content_knowledge | object_relation_value | short | green | What is the color of grass?\nAnswer with one word.\nAnswer: |
| pf_case_0010 | content_knowledge | object_relation_value | explain | green | What is the color of grass?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0011 | content_knowledge | object_relation_value | repeat | green | What is the color of grass?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0012 | content_knowledge | object_relation_value | list | green | What is the color of grass?\nAnswer as a short list.\nAnswer: |
| pf_case_0013 | content_knowledge | object_relation_value | short | white | What is the color of snow?\nAnswer with one word.\nAnswer: |
| pf_case_0014 | content_knowledge | object_relation_value | explain | white | What is the color of snow?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0015 | content_knowledge | object_relation_value | repeat | white | What is the color of snow?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0016 | content_knowledge | object_relation_value | list | white | What is the color of snow?\nAnswer as a short list.\nAnswer: |
| pf_case_0017 | content_knowledge | object_relation_value | short | black | What is the color of coal?\nAnswer with one word.\nAnswer: |
| pf_case_0018 | content_knowledge | object_relation_value | explain | black | What is the color of coal?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0019 | content_knowledge | object_relation_value | repeat | black | What is the color of coal?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0020 | content_knowledge | object_relation_value | list | black | What is the color of coal?\nAnswer as a short list.\nAnswer: |
| pf_case_0021 | content_knowledge | object_relation_value | short | sour | What is the taste of lemon?\nAnswer with one word.\nAnswer: |
| pf_case_0022 | content_knowledge | object_relation_value | explain | sour | What is the taste of lemon?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0023 | content_knowledge | object_relation_value | repeat | sour | What is the taste of lemon?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0024 | content_knowledge | object_relation_value | list | sour | What is the taste of lemon?\nAnswer as a short list.\nAnswer: |
| pf_case_0025 | content_knowledge | object_relation_value | short | hit | What is the function of hammer?\nAnswer with one word.\nAnswer: |
| pf_case_0026 | content_knowledge | object_relation_value | explain | hit | What is the function of hammer?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0027 | content_knowledge | object_relation_value | repeat | hit | What is the function of hammer?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0028 | content_knowledge | object_relation_value | list | hit | What is the function of hammer?\nAnswer as a short list.\nAnswer: |
| pf_case_0029 | content_knowledge | object_relation_value | short | car | What is the part_of of wheel?\nAnswer with one word.\nAnswer: |
| pf_case_0030 | content_knowledge | object_relation_value | explain | car | What is the part_of of wheel?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| pf_case_0031 | content_knowledge | object_relation_value | repeat | car | What is the part_of of wheel?\nAnswer with exactly the same answer word twice, separated by a comma.\nAnswer: |
| pf_case_0032 | content_knowledge | object_relation_value | list | car | What is the part_of of wheel?\nAnswer as a short list.\nAnswer: |
| reason_negation_0001 | reasoning_constraint | negation | special | no | If the apple is not green, is it green?\nAnswer with yes or no.\nAnswer: |
| syntax_boundary_0001 | syntax_structure | punctuation_boundary | special | red | Answer: red. Continue?\nAnswer with one word.\nAnswer: |
| readout_takeover_0001 | readout_competition | second_competitor_takeover | special | red | What is the color of apple?\nAnswer with the answer first, then one short reason using because.\nAnswer: |
| closure_stop_0001 | closure | model_close | special | white | What is the color of snow?\nAnswer with one word and stop.\nAnswer: |

## Program Phases

| order | phase | task | objective |
| ---: | --- | --- | --- |
| 1 | phase234 | pattern_family_matrix | 建立模式族、模式、测试层级、样例任务的机器可读矩阵。 |
| 2 | phase235 | behavior_family_benchmark | 先跑行为层模式分类，覆盖 qwen3、GLM4、DS7B。 |
| 3 | phase236 | prompt_trigger_family_atlas | 对高差异模式做 prompt trigger 和 anchor 消融。 |
| 4 | phase237 | gate_product_family_atlas | 采集 gate/up/product/down_out 跨模式差分。 |
| 5 | phase238 | readout_competition_family_atlas | 统一记录 winner、second competitor、remaining gap。 |
| 6 | phase239 | source_suppression_family_validation | 选择最稳定模式做 source-level suppression。 |
| 7 | phase240 | closure_candidate_family_validation | 选择少数模式尝试完整闭合。 |
