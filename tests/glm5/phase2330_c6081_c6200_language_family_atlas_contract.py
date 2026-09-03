#!/usr/bin/env python3
"""Audit Phase2315-2329 and freeze a broad typed language-family atlas contract."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2330_c6081_c6200_language_family_atlas_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/typed_language_family_atlas.jsonl"
CONFIG = OUT / "config/frozen_contract.json"
PHASE = 2330
CAMPAIGN = "C6081-C6200"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as old  # noqa: E402


OLD_FAMILIES = old.FAMILIES
NEW_FAMILIES = (
    "attribute_binding",
    "part_whole",
    "causal_relation",
    "negation_scope",
    "quantifier_scope",
    "preposition_direction",
    "coreference_anaphora",
    "punctuation_boundary",
    "quotation_speaker",
    "translation_equivalence",
    "style_register",
    "conjunction_scope",
)
FAMILIES = OLD_FAMILIES + NEW_FAMILIES
LANGUAGES = ("en", "zh")
SURFACES = ("narrative", "reported")
UNITS = 12
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
PARTITION_BY_UNIT = {unit: PARTITIONS[unit // 3] for unit in range(UNITS)}
MACROTYPE = {
    "agent_patient": "role_binding", "attitude_event": "context_scope",
    "comparison_order": "logical_relation", "location_binding": "relational_binding",
    "possession_query": "relational_binding", "relative_binding": "grammar_binding",
    "temporal_order": "logical_relation", "taxonomy_chain": "concept_graph",
    "attribute_binding": "concept_graph", "part_whole": "concept_graph",
    "causal_relation": "logical_relation", "negation_scope": "context_scope",
    "quantifier_scope": "logical_relation", "preposition_direction": "relational_binding",
    "coreference_anaphora": "context_scope", "punctuation_boundary": "form_boundary",
    "quotation_speaker": "context_scope", "translation_equivalence": "cross_language",
    "style_register": "output_control", "conjunction_scope": "grammar_binding",
}

ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\2e64eb62-18fb-4e66-9e6f-606e46c13e39\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\a82e5e9a-271f-4579-ba34-1a16087d841b\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\f6be2770-aa4d-4181-b895-ed1f20a8b531\pasted-text.txt"),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def other_attribute(language: str, unit: int) -> str:
    if language == "en":
        return old.EN_ATTRIBUTES[(unit + 7) % len(old.EN_ATTRIBUTES)]
    return old.ZH_ATTRIBUTES[(unit + 7) % len(old.ZH_ATTRIBUTES)]


def wrap(language: str, surface: str, text: str, unit: int) -> tuple[str, str]:
    if surface == "reported":
        text = (f'A curator reported, "{text}"' if language == "en"
                else f"一位记录员说：“{text}”")
    cue, cue_id = old.cue(language, unit)
    return text + cue, cue_id


def new_semantics(family: str, language: str, unit: int, state: int, surface: str) -> dict:
    v = old.lexicon(language, unit)
    a, b = v["a"], v["b"]
    obj, obj2 = v["object_a"], v["object_b"]
    loc, loc2 = v["location_a"], v["location_b"]
    target, wrong = ((a, b) if state == 0 else (b, a))
    graph: list[list[str]] = []
    changed = "state_assignment"
    invariant = "lexical_inventory_and_query"
    if language == "en":
        if family == "attribute_binding":
            item_target, item_wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            facts = f"The {item_target} was {v['attribute']}; the {item_wrong} was {other_attribute(language, unit)}."
            future, false = f"The {v['attribute']} item was the {item_target}.", f"The {v['attribute']} item was the {item_wrong}."
            target, wrong = item_target, item_wrong
            graph = [[target, "has_property", v["attribute"]], [wrong, "has_property", other_attribute(language, unit)]]
        elif family == "part_whole":
            whole_target, whole_wrong = ((loc, loc2) if state == 0 else (loc2, loc))
            facts = f"The {obj} was a component of the {whole_target}; the {obj2} was a component of the {whole_wrong}."
            future, false = f"The whole containing the {obj} was the {whole_target}.", f"The whole containing the {obj} was the {whole_wrong}."
            target, wrong = whole_target, whole_wrong
            graph = [[obj, "part_of", target], [obj2, "part_of", wrong]]
        elif family == "causal_relation":
            facts = f"{target} opened the vent, causing the {obj} to cool. {wrong} merely watched."
            future, false = f"The person whose action caused the {obj} to cool was {target}.", f"The person whose action caused the {obj} to cool was {wrong}."
            graph = [[target, "caused", f"{obj}_cool"], [wrong, "observed", f"{obj}_cool"]]
        elif family == "negation_scope":
            facts = f"{target} carried the {obj}; {wrong} did not carry the {obj}."
            future, false = f"The person who carried the {obj} was {target}.", f"The person who carried the {obj} was {wrong}."
            graph = [[target, "carried", obj], [wrong, "not_carried", obj]]
        elif family == "quantifier_scope":
            facts = f"Only {target}, and not {wrong}, inspected the {obj}."
            future, false = f"The sole person who inspected the {obj} was {target}.", f"The sole person who inspected the {obj} was {wrong}."
            graph = [["only", "scope", target], [target, "inspected", obj]]
        elif family == "preposition_direction":
            facts = f"{target} stood behind {wrong}; therefore {wrong} stood in front of {target}."
            future, false = f"The person behind the other was {target}.", f"The person behind the other was {wrong}."
            graph = [[target, "behind", wrong], [wrong, "in_front_of", target]]
        elif family == "coreference_anaphora":
            facts = f"{a} called {b}. In the next sentence, 'the former' means {target} and 'the latter' means {wrong}."
            future, false = f"The expression 'the former' referred to {target}.", f"The expression 'the former' referred to {wrong}."
            graph = [["former", "refers_to", target], ["latter", "refers_to", wrong]]
        elif family == "punctuation_boundary":
            target, wrong = (("question mark", "period") if state == 0 else ("period", "question mark"))
            kind = "direct question" if state == 0 else "declarative statement"
            facts = f"The final clause is a {kind}."
            future, false = f"Its closing punctuation should be a {target}.", f"Its closing punctuation should be a {wrong}."
            graph = [[kind, "closed_by", target]]
            changed, invariant = "sentence_function_and_boundary_mark", "metalanguage_and_output_interface"
        elif family == "quotation_speaker":
            facts = f'{target} said, "I stored the {obj}." {wrong} remained silent.'
            future, false = f"The speaker who used the word 'I' was {target}.", f"The speaker who used the word 'I' was {wrong}."
            graph = [["I", "quoted_speaker", target], [wrong, "silent", "true"]]
        elif family == "translation_equivalence":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            source = f"代号{unit}{'甲' if state == 0 else '乙'}"
            facts = f'In this bilingual glossary, the Chinese label "{source}" means "{target}", not "{wrong}".'
            future, false = f'The English translation of "{source}" is {target}.', f'The English translation of "{source}" is {wrong}.'
            graph = [[source, "translates_to", target], [source, "not_translates_to", wrong]]
            changed, invariant = "source_target_lexical_mapping", "referent_identity"
        elif family == "style_register":
            target, wrong = (("Thank you", "Thanks") if state == 0 else ("Thanks", "Thank you"))
            register = "formal" if state == 0 else "casual"
            facts = f"The requested register is {register}."
            future, false = f"The appropriate reply is '{target}'.", f"The appropriate reply is '{wrong}'."
            graph = [[register, "selects_form", target]]
            changed, invariant = "register_and_surface_form", "gratitude_intent"
        elif family == "conjunction_scope":
            facts = f"{target} repaired the {obj}, while {wrong} carried the {obj2}."
            future, false = f"The person linked to repairing the {obj} was {target}.", f"The person linked to repairing the {obj} was {wrong}."
            graph = [[target, "repaired", obj], [wrong, "carried", obj2]]
        else:
            raise KeyError(family)
    else:
        if family == "attribute_binding":
            item_target, item_wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            facts = f"{item_target}是{v['attribute']}的；{item_wrong}是{other_attribute(language, unit)}的。"
            future, false = f"具有{v['attribute']}属性的是{item_target}。", f"具有{v['attribute']}属性的是{item_wrong}。"
            target, wrong = item_target, item_wrong
            graph = [[target, "has_property", v["attribute"]], [wrong, "has_property", other_attribute(language, unit)]]
        elif family == "part_whole":
            whole_target, whole_wrong = ((loc, loc2) if state == 0 else (loc2, loc))
            facts = f"{obj}是{whole_target}的组成部分；{obj2}是{whole_wrong}的组成部分。"
            future, false = f"包含{obj}的整体是{whole_target}。", f"包含{obj}的整体是{whole_wrong}。"
            target, wrong = whole_target, whole_wrong
            graph = [[obj, "part_of", target], [obj2, "part_of", wrong]]
        elif family == "causal_relation":
            facts = f"{target}打开了通风口，导致{obj}冷却；{wrong}只是在旁边看。"
            future, false = f"其行动导致{obj}冷却的人是{target}。", f"其行动导致{obj}冷却的人是{wrong}。"
            graph = [[target, "caused", f"{obj}_cool"], [wrong, "observed", f"{obj}_cool"]]
        elif family == "negation_scope":
            facts = f"{target}携带了{obj}；{wrong}没有携带{obj}。"
            future, false = f"携带{obj}的人是{target}。", f"携带{obj}的人是{wrong}。"
            graph = [[target, "carried", obj], [wrong, "not_carried", obj]]
        elif family == "quantifier_scope":
            facts = f"只有{target}检查了{obj}，{wrong}没有检查。"
            future, false = f"唯一检查{obj}的人是{target}。", f"唯一检查{obj}的人是{wrong}。"
            graph = [["only", "scope", target], [target, "inspected", obj]]
        elif family == "preposition_direction":
            facts = f"{target}站在{wrong}后面，因此{wrong}站在{target}前面。"
            future, false = f"站在另一人后面的是{target}。", f"站在另一人后面的是{wrong}。"
            graph = [[target, "behind", wrong], [wrong, "in_front_of", target]]
        elif family == "coreference_anaphora":
            facts = f"{a}给{b}打了电话。下一句中，“前者”指{target}，“后者”指{wrong}。"
            future, false = f"“前者”指的是{target}。", f"“前者”指的是{wrong}。"
            graph = [["former", "refers_to", target], ["latter", "refers_to", wrong]]
        elif family == "punctuation_boundary":
            target, wrong = (("问号", "句号") if state == 0 else ("句号", "问号"))
            kind = "直接疑问句" if state == 0 else "陈述句"
            facts = f"最后一个分句是{kind}。"
            future, false = f"它结尾应使用{target}。", f"它结尾应使用{wrong}。"
            graph = [[kind, "closed_by", target]]
            changed, invariant = "sentence_function_and_boundary_mark", "metalanguage_and_output_interface"
        elif family == "quotation_speaker":
            facts = f"{target}说：“我存放了{obj}。”{wrong}一直沉默。"
            future, false = f"引语中使用“我”的说话者是{target}。", f"引语中使用“我”的说话者是{wrong}。"
            graph = [["I", "quoted_speaker", target], [wrong, "silent", "true"]]
        elif family == "translation_equivalence":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            source = f"code-{unit}-{'A' if state == 0 else 'B'}"
            facts = f"在这份双语词表中，英文标签“{source}”的中文意思是“{target}”，不是“{wrong}”。"
            future, false = f"“{source}”的中文翻译是{target}。", f"“{source}”的中文翻译是{wrong}。"
            graph = [[source, "translates_to", target], [source, "not_translates_to", wrong]]
            changed, invariant = "source_target_lexical_mapping", "referent_identity"
        elif family == "style_register":
            target, wrong = (("非常感谢", "谢了") if state == 0 else ("谢了", "非常感谢"))
            register = "正式" if state == 0 else "口语"
            facts = f"要求使用{register}语体。"
            future, false = f"合适的回答是“{target}”。", f"合适的回答是“{wrong}”。"
            graph = [[register, "selects_form", target]]
            changed, invariant = "register_and_surface_form", "gratitude_intent"
        elif family == "conjunction_scope":
            facts = f"{target}修理了{obj}，而{wrong}搬运了{obj2}。"
            future, false = f"与修理{obj}相联系的人是{target}。", f"与修理{obj}相联系的人是{wrong}。"
            graph = [[target, "repaired", obj], [wrong, "carried", obj2]]
        else:
            raise KeyError(family)
    prompt, cue_id = wrap(language, surface, facts, unit)
    return {
        "future_prompt": prompt, "future_target_text": future, "future_wrong_text": false,
        "identity_target": target, "identity_wrong": wrong,
        "role_values": {"primary": target, "secondary": wrong}, "semantic_graph": graph,
        "target_mention_order": "first" if unit % 2 == 0 else "last", "cue_id": cue_id,
        "external_operation": family, "preserved_invariant": invariant, "changed_variable": changed,
    }


def compile_material() -> tuple[list[dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    rows: list[dict] = []
    for family_index, family in enumerate(FAMILIES):
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    for state in (0, 1):
                        semantics = (old.compile_semantics(family, language, unit, state, surface)
                                     if family in OLD_FAMILIES else
                                     new_semantics(family, language, unit, state, surface))
                        prompt_ids = [int(x) for x in tokenizer.encode(semantics["future_prompt"], add_special_tokens=False)]
                        target_ids = old.answer_ids(tokenizer, semantics["future_target_text"], language)
                        wrong_ids = old.answer_ids(tokenizer, semantics["future_wrong_text"], language)
                        roles = {key: old.role_token_positions(tokenizer, prompt_ids, str(text), language)
                                 for key, text in semantics["role_values"].items()}
                        rows.append({
                            "case_id": f"c6081-{family}-{language}-{surface}-u{unit:02d}-s{state}",
                            "design_index": len(rows), "family": family, "family_index": family_index,
                            "macrotype": MACROTYPE[family], "language": language, "surface": surface,
                            "unit": unit, "state": state, "partition": PARTITION_BY_UNIT[unit],
                            **semantics, "future_prompt_ids": prompt_ids,
                            "future_target_ids": target_ids, "future_wrong_ids": wrong_ids,
                            "identity_target_ids": old.answer_ids(tokenizer, semantics["identity_target"], language),
                            "identity_wrong_ids": old.answer_ids(tokenizer, semantics["identity_wrong"], language),
                            "role_positions": roles, "boundary_position": len(prompt_ids) - 1,
                            "interface": "raw_declarative_natural_sentence_continuation",
                        })
    pair_groups: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        pair_groups[(row["family"], row["language"], row["surface"], row["unit"])].add(row["state"])
    cell_counts = Counter(
        (row["family"], row["language"], row["surface"], row["partition"], row["state"])
        for row in rows
    )
    audit = {
        "rows": len(rows), "families": len(FAMILIES), "macrotypes": len(set(MACROTYPE.values())),
        "unique_case_ids": len({row["case_id"] for row in rows}) == len(rows),
        "state_pairs_complete": all(value == {0, 1} for value in pair_groups.values()),
        "every_cell_has_three_units": all(value == 3 for value in cell_counts.values()),
        "target_wrong_text_distinct": all(row["future_target_text"] != row["future_wrong_text"] for row in rows),
        "nonempty_token_sequences": all(row["future_prompt_ids"] and row["future_target_ids"] and row["future_wrong_ids"] for row in rows),
        "first_token_collision_fraction": sum(row["future_target_ids"][0] == row["future_wrong_ids"][0] for row in rows) / len(rows),
        "replacement_character_count": sum("�" in canonical(row) for row in rows),
        "cell_count": len(cell_counts), "cell_min": min(cell_counts.values()), "cell_max": max(cell_counts.values()),
    }
    return rows, audit


def source_audit() -> dict:
    p2317 = (TESTS / "phase2317_c5161_c5240_directional_response_identification.py").read_text(encoding="utf-8")
    p2318 = (TESTS / "phase2318_c5241_c5320_crossmodel_directional_topology.py").read_text(encoding="utf-8")
    return {
        "base_rademacher_unit_normalized": "value /= np.linalg.norm(value.astype(np.float64))" in p2317,
        "pair_direction_is_unrenormalized_sum": "output.append(base[left] + base[right])" in p2317,
        "pair_superposition_scale_matched": "actual - predicted" in p2317 and "left" in p2317 and "right" in p2317,
        "crossmodel_base_unit_normalized": "value /= np.linalg.norm(value.astype(np.float64))" in p2318,
        "remaining_single_dose_limitation": "DOSE = contract.PERTURBATION_DOSE" in p2317,
        "derivative_written_float16": "dtype=np.float16" in p2317,
        "interpretation": (
            "The attachment's normalization concern is valid for the memo formula but not for the implementation: "
            "base directions were unit normalized. Pair sums were intentionally left unnormalized, so the measured "
            "direction is u+v and the additive comparison is scale-consistent. Multi-dose convergence and a richer null remain missing."
        ),
    }


def frozen_contract(audit: dict, source: dict) -> dict:
    attachment_hashes = [{"path": str(path), "sha256": file_hash(path)} for path in ATTACHMENTS]
    return {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model_load": True,
        "material": {"families": list(FAMILIES), "macrotype": MACROTYPE, "languages": list(LANGUAGES),
                     "surfaces": list(SURFACES), "units": UNITS, "partitions": list(PARTITIONS), "rows": audit["rows"]},
        "evidence_audit": {
            "retained": [
                "behavior before semantic interpretation", "all physical coordinates retained",
                "typed external language graph rather than flat token taxonomy",
                "natural semantic contrasts plus model-local random controls",
                "fresh lexical partitions and wrong-family/wrong-role/wrong-layer controls",
                "cross-model functional topology without coordinate-number alignment",
            ],
            "corrected": [
                "simple family/main-effect failure does not disprove all static or distributed encodings",
                "0.807-0.909 sign agreement is a candidate observation, not a topological invariant",
                "FP16/BF16 differences show numerical-format sensitivity, not pure observation noise",
                "runtime activation coordinates are not model parameters and are basis-dependent addresses",
                "Riemannian manifolds, geodesics, attractors, connections and tensor fields are hypotheses, not results",
                "four to eight probe directions identify only a projected response, never a full Jacobian",
            ],
            "source_code_audit": source,
        },
        "stage_plan": [
            "Qwen3-4B broad behavior and complete boundary/all-coordinate atlas over twenty families",
            "FP16/BF16 multi-dose normalized random and natural-direction calibration on a balanced panel",
            "full-coordinate family reuse/difference passports with discovery-confirmation freeze",
            "fresh_confirmation/fresh_lockbox prospective adjudication with explicit nulls",
            "sequential model-local functional replication only for surviving observations",
            "causal invocation/deletion/reversal/rescue only for prospectively surviving natural operations",
        ],
        "coordinate_policy": "all physical coordinates retained; no Top-K, PCA, projection, or pre-analysis compression",
        "visualization_policy": "publish important embedding/HiddenState parameter-level arrays; otherwise delete duplicate raw fields after verified summaries",
        "numerical_policy": {
            "base_directions_unit_normalized": True, "pair_sum_not_renormalized": True,
            "doses": [0.02, 0.01, 0.005, 0.0025], "formats": ["float16", "bfloat16"],
            "fp32_full_model_cuda_feasible": False,
            "fp32_reason": "local RTX 5080 has 16.3 GB; Qwen3-4B FP32 weights alone require about 16 GB before activations",
        },
        "claim_ladder": ["behavior", "observational_atlas", "prospective_prediction", "causal_net_effect", "mechanism_closure"],
        "attachment_hashes": attachment_hashes,
    }


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 附件证据重裁与二十语言模式族全坐标图谱总合同（{CAMPAIGN}） [{stamp}]

**测试原理、证据审查与测试用例。** 本期不加载模型，逐条核对三份附件、Phase2315–2329 记录与 Phase2317/2318 扰动源码。保留“行为先于机制解释、全坐标保留、未见词汇前瞻、模型本地功能比较、自然语义操作替代随机方向成为主线”；修正“简单均值失败即静态编码不存在”“符号一致即拓扑不变量”“FP16/BF16即纯保存噪声”“流形、测地线、吸引子、联络张量已经成立”等过度结论。源码确认基础 Rademacher 方向实际执行 `r/||r||`，成对方向是未再归一化的 `r_a+r_b`，所以叠加比较尺度匹配；旧 MEMO 公式未明确这一点。新材料一次冻结二十族、八宏类型、中英、叙事/转述、双状态、十二词汇单元与四个互斥分区，共 `{result['material_audit']['rows']}` 行。示例覆盖施受事、属性、part-whole、因果、否定、量词、介词方向、指代、标点、引语、翻译、风格和合取作用域；每族显式记录保持量、改变量和外部操作。

$$
u_k=rac{{r_k}}{{lVert r_kVert_2}},qquad
D_delta(u)=rac{{H(h+deltalVert hVert u)-H(h-deltalVert hVert u)}}{{2deltalVert hVert}}.
$$

$$
\mathcal G_{{language}}\longrightarrow
\mathcal G_{{HiddenState}}\longrightarrow
\text{{未见材料预测}}\longrightarrow
\text{{选择性因果验证}}.
$$

**结果汇总、相关文件与冻结方案。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；源码审计 `{json.dumps(result['source_audit'], ensure_ascii=False)}`；附件 SHA256 `{json.dumps(result['contract']['attachment_hashes'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2330_c6081_c6200_language_family_atlas_contract.py`；结果 `tests/glm5/result/phase2330_c6081_c6200_language_family_atlas_contract`。后续作为一个连续大阶段完成：Qwen3-4B 广谱行为与 boundary 全检查点全坐标场；FP16/BF16 多剂量校准；发现/确认全坐标复用—差异护照；fresh 前瞻裁决；仅对保留下来的规律顺序运行其他模型和因果验证。重要词嵌入与 HiddenState 坐标场发布到客户端，否则验证摘要后清理重复原始场。

**分析、理论进展、问题硬伤与结论。** 当前证据只支持“固定随机投影下存在可重复的局部响应成分，简单语言/表面/状态/族均值不能稳定预测幅值”；它既没有否定一切静态分布式结构，也没有证明流形几何。完整 Jacobian 需要至少 2560 个线性独立方向，现有 4–8 方向只得到 `J P_U`。单个 HiddenState 坐标是运行时激活地址，不是训练参数，也不具换基不变性，但用户要求的参数级显示可理解为“不压缩地显示每个原始激活坐标”。硬伤包括无独立人类盲评、模板生成器共享、16.3GB GPU 无法容纳 Qwen3-4B 完整 FP32 CUDA 前向、自由生成判分不完美。故本阶段不宣布新数学，只把微分几何、低秩、稀疏或张量模型作为常规全坐标观察之后的候选解释。

**下一大任务。** 直接运行二十族 Qwen3-4B FP16 行为与全边界场；无论部分因果门是否失败，都继续围绕行为通过或内部复现较强的特征积累图谱，而不回到已失败的简单均值路线。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    rows, audit = compile_material()
    source = source_audit()
    contract = frozen_contract(audit, source)
    write_rows(MATERIAL, rows)
    save(CONFIG, contract)
    checks = {
        "material_audit": all([
            audit["unique_case_ids"], audit["state_pairs_complete"], audit["every_cell_has_three_units"],
            audit["target_wrong_text_distinct"], audit["nonempty_token_sequences"], audit["replacement_character_count"] == 0,
        ]),
        "source_normalization_verified": source["base_rademacher_unit_normalized"],
        "pair_scale_verified": source["pair_direction_is_unrenormalized_sum"] and source["pair_superposition_scale_matched"],
        "twenty_families": len(FAMILIES) == 20,
        "four_partitions": set(PARTITION_BY_UNIT.values()) == set(PARTITIONS),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "material_audit": audit,
        "source_audit": source, "contract": contract,
        "hashes": {"material": file_hash(MATERIAL), "config": file_hash(CONFIG)},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    save(final_path, result)
    if not result["all_checks_passed"]:
        raise RuntimeError(("contract_failed", checks))
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
