#!/usr/bin/env python3
"""Independent-generator replication for surviving and behavior-strong language families."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2333 = RESULT / "phase2333_c6481_c6640_twenty_family_coordinate_atlas"
P2334 = RESULT / "phase2334_c6641_c6760_fresh_family_atlas_adjudication"
OUT = RESULT / "phase2335_c6761_c6920_independent_construction_replication"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
VIS = ROOT / "frontend/public/vis_data/research_kernel"
OLD_DELTA = VIS / "c6483_qwen4b_twenty_family_natural_state_delta.float32.npy"
BOUNDARY = OUT / "raw/qwen4b_independent_boundary.float16.npy"
BOUNDARY_PROGRESS = OUT / "raw/boundary_progress.json"
MATERIAL = OUT / "material/independent_constructions.jsonl"
PHASE = 2335
CAMPAIGN = "C6761-C6920"
FAMILIES = (
    "coreference_anaphora", "punctuation_boundary", "style_register", "translation_equivalence",
    "temporal_order", "attribute_binding", "quotation_speaker",
)
QPOINTS = {
    "coreference_anaphora": 23, "punctuation_boundary": 2, "style_register": 1,
    "translation_equivalence": 1, "temporal_order": 7, "attribute_binding": 2,
    "quotation_speaker": 4,
}
LANGUAGES = ("en", "zh")
SURFACES = ("direct", "paraphrase")
UNITS = 16
PARTITION = {unit: "independent_development" if unit < 8 else "independent_lockbox" for unit in range(UNITS)}
EPS = 1e-12

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2330_c6081_c6200_language_family_atlas_contract as broad  # noqa: E402
import phase2331_c6201_c6360_qwen4b_twenty_family_fullfield as broad_field  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return io.read_rows(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    io.write_rows(path, rows)


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def modules(model) -> list[Any]:
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def wrap(language: str, surface: str, facts: str) -> str:
    if language == "en":
        return (facts + " Complete the next factual sentence:" if surface == "direct" else
                f"A proofreader restates this record without changing its meaning: {facts} The restatement says that")
    return (facts + "请补全下一句事实：" if surface == "direct" else
            f"校对员在不改变含义的前提下转述这条记录：{facts}转述结论是：")


def style_pair(language: str, unit: int) -> tuple[str, str, str]:
    en = (
        ("gratitude", "I sincerely appreciate your help", "Thanks a lot"),
        ("apology", "Please accept my sincere apology", "Sorry about that"),
        ("request", "Would you kindly send the file", "Send me the file, please"),
        ("greeting", "It is a pleasure to meet you", "Nice to meet you"),
    )
    zh = (
        ("感谢", "谨向您表示诚挚感谢", "多谢啦"),
        ("道歉", "请接受我诚挚的歉意", "对不住啦"),
        ("请求", "烦请您发送该文件", "麻烦把文件发来"),
        ("问候", "很荣幸与您见面", "见到你真好"),
    )
    return (en if language == "en" else zh)[unit % 4]


def make_semantics(family: str, language: str, surface: str, unit: int, state: int) -> dict:
    v = io.lexicon(language, unit)
    a, b, obj, obj2 = v["a"], v["b"], v["object_a"], v["object_b"]
    target, wrong = ((a, b) if state == 0 else (b, a))
    variant = unit % 4
    if language == "en":
        if family == "coreference_anaphora":
            ordered = (target, wrong)
            facts = (
                f"The roster names {ordered[0]} first and {ordered[1]} second." if variant == 0 else
                f"Of the two participants, {ordered[0]} was introduced before {ordered[1]}." if variant == 1 else
                f"The pair appears in this order: {ordered[0]}, then {ordered[1]}." if variant == 2 else
                f"The earlier-mentioned person is {ordered[0]}; the later-mentioned person is {ordered[1]}."
            )
            future, false = f"The earlier-mentioned participant was {target}.", f"The earlier-mentioned participant was {wrong}."
        elif family == "punctuation_boundary":
            questions = (f"Where is the {obj}", f"Did {a} carry the {obj}", f"Who opened the {obj}", f"When did {a} arrive")
            statements = (f"The {obj} is nearby", f"{a} carried the {obj}", f"{a} opened the {obj}", f"{a} arrived today")
            target, wrong = (("question mark", "period") if state == 0 else ("period", "question mark"))
            sentence = questions[variant] if state == 0 else statements[variant]
            facts = f"The unpunctuated sentence is: '{sentence}'."
            future, false = f"Its required closing mark is a {target}.", f"Its required closing mark is a {wrong}."
        elif family == "style_register":
            intent, formal, casual = style_pair(language, unit)
            target, wrong = ((formal, casual) if state == 0 else (casual, formal))
            register = ("professional formal" if variant % 2 == 0 else "polite official") if state == 0 else ("relaxed casual" if variant % 2 == 0 else "informal conversational")
            facts = f"Express {intent} in a {register} register."
            future, false = f"The requested wording is '{target}'.", f"The requested wording is '{wrong}'."
        elif family == "translation_equivalence":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            code = f"nava-{unit}-x"
            facts = f"A separate lexicon defines '{code}' as '{target}' and explicitly excludes '{wrong}'."
            future, false = f"The translation of '{code}' is {target}.", f"The translation of '{code}' is {wrong}."
        elif family == "temporal_order":
            if variant % 2 == 0:
                facts = f"{target} departed after {wrong}."
            else:
                facts = f"{wrong} finished before {target}."
            future, false = f"The later person was {target}.", f"The later person was {wrong}."
        elif family == "attribute_binding":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            other = io.EN_ATTRIBUTES[(unit + 5) % len(io.EN_ATTRIBUTES)]
            facts = (f"Among two items, the {target} alone was {v['attribute']}; the {wrong} was {other}." if variant % 2 == 0 else
                     f"{v['attribute'].title()} described the {target}, whereas {other} described the {wrong}.")
            future, false = f"The {v['attribute']} object was the {target}.", f"The {v['attribute']} object was the {wrong}."
        elif family == "quotation_speaker":
            facts = (f'"I placed the {obj} here," announced {target}; {wrong} wrote nothing.' if variant % 2 == 0 else
                     f'{wrong} stayed silent while {target} declared, "I found the {obj}."')
            future, false = f"The first-person speaker was {target}.", f"The first-person speaker was {wrong}."
        else:
            raise KeyError(family)
    else:
        if family == "coreference_anaphora":
            facts = (
                f"名单先写{target}，再写{wrong}。" if variant == 0 else
                f"两位参与者中，{target}先被介绍，{wrong}后被介绍。" if variant == 1 else
                f"两人的出现顺序是：{target}，然后是{wrong}。" if variant == 2 else
                f"较早提及的人是{target}；较晚提及的人是{wrong}。"
            )
            future, false = f"较早提及的参与者是{target}。", f"较早提及的参与者是{wrong}。"
        elif family == "punctuation_boundary":
            questions = (f"{obj}在哪里", f"{a}带了{obj}吗", f"谁打开了{obj}", f"{a}何时到达")
            statements = (f"{obj}就在附近", f"{a}带了{obj}", f"{a}打开了{obj}", f"{a}今天到达")
            target, wrong = (("问号", "句号") if state == 0 else ("句号", "问号"))
            sentence = questions[variant] if state == 0 else statements[variant]
            facts = f"未加标点的句子是：“{sentence}”。"
            future, false = f"它结尾需要使用{target}。", f"它结尾需要使用{wrong}。"
        elif family == "style_register":
            intent, formal, casual = style_pair(language, unit)
            target, wrong = ((formal, casual) if state == 0 else (casual, formal))
            register = ("专业正式" if variant % 2 == 0 else "礼貌公文") if state == 0 else ("轻松口语" if variant % 2 == 0 else "日常随意")
            facts = f"请用{register}语体表达{intent}。"
            future, false = f"符合要求的措辞是“{target}”。", f"符合要求的措辞是“{wrong}”。"
        elif family == "translation_equivalence":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            code = f"nava-{unit}-x"
            facts = f"另一份词典把“{code}”定义为“{target}”，并明确排除“{wrong}”。"
            future, false = f"“{code}”的翻译是{target}。", f"“{code}”的翻译是{wrong}。"
        elif family == "temporal_order":
            facts = f"{target}在{wrong}之后离开。" if variant % 2 == 0 else f"{wrong}比{target}更早完成。"
            future, false = f"较晚的人是{target}。", f"较晚的人是{wrong}。"
        elif family == "attribute_binding":
            target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
            other = io.ZH_ATTRIBUTES[(unit + 5) % len(io.ZH_ATTRIBUTES)]
            facts = (f"两件物品中，只有{target}是{v['attribute']}的；{wrong}是{other}的。" if variant % 2 == 0 else
                     f"{v['attribute']}描述的是{target}，而{other}描述的是{wrong}。")
            future, false = f"具有{v['attribute']}属性的是{target}。", f"具有{v['attribute']}属性的是{wrong}。"
        elif family == "quotation_speaker":
            facts = (f"“我把{obj}放在这里了。”{target}宣布；{wrong}什么也没写。" if variant % 2 == 0 else
                     f"{wrong}保持沉默，而{target}说：“我找到了{obj}。”")
            future, false = f"使用第一人称的说话者是{target}。", f"使用第一人称的说话者是{wrong}。"
        else:
            raise KeyError(family)
    return {
        "future_prompt": wrap(language, surface, facts), "future_target_text": future,
        "future_wrong_text": false, "identity_target": target, "identity_wrong": wrong,
        "semantic_graph": [[family, "state", state]], "role_values": {"primary": target, "secondary": wrong},
        "external_operation": family, "preserved_invariant": "family_function_and_lexical_inventory",
        "changed_variable": "operation_state", "template_variant": variant,
    }


def compile_material() -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    for state in (0, 1):
                        sem = make_semantics(family, language, surface, unit, state)
                        prompt_ids = [int(x) for x in tokenizer.encode(sem["future_prompt"], add_special_tokens=False)]
                        rows.append({
                            "case_id": f"c6761-{family}-{language}-{surface}-u{unit:02d}-s{state}",
                            "design_index": len(rows), "family": family, "family_index": family_index,
                            "macrotype": broad.MACROTYPE[family], "language": language, "surface": surface,
                            "unit": unit, "state": state, "partition": PARTITION[unit], "cue_id": "independent",
                            "target_mention_order": "first" if unit % 2 == 0 else "last", **sem,
                            "future_prompt_ids": prompt_ids,
                            "future_target_ids": io.answer_ids(tokenizer, sem["future_target_text"], language),
                            "future_wrong_ids": io.answer_ids(tokenizer, sem["future_wrong_text"], language),
                            "identity_target_ids": io.answer_ids(tokenizer, sem["identity_target"], language),
                            "identity_wrong_ids": io.answer_ids(tokenizer, sem["identity_wrong"], language),
                            "boundary_position": len(prompt_ids) - 1,
                        })
    audit = {
        "rows": len(rows), "families": len(FAMILIES), "partitions": dict(defaultdict(int)),
        "unique": len({row["case_id"] for row in rows}) == len(rows),
        "no_replacement": all("�" not in json.dumps(row, ensure_ascii=False) for row in rows),
        "target_wrong_distinct": all(row["future_target_text"] != row["future_wrong_text"] for row in rows),
        "templates_per_family": 4,
    }
    audit["partitions"] = {p: sum(row["partition"] == p for row in rows) for p in set(PARTITION.values())}
    return rows, audit


def capture_boundary(model, device, rows: list[dict], batch_size: int = 12) -> dict:
    module_list = modules(model)
    shape = (len(rows), len(module_list), int(model.config.hidden_size))
    BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    if BOUNDARY.exists() and BOUNDARY_PROGRESS.exists():
        progress = json.loads(BOUNDARY_PROGRESS.read_text(encoding="utf-8"))
        completed = int(progress["completed"])
        field = np.lib.format.open_memmap(BOUNDARY, mode="r+")
    else:
        completed = 0
        field = np.lib.format.open_memmap(BOUNDARY, mode="w+", dtype=np.float16, shape=shape)
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for q, module in enumerate(module_list):
        def hook(_module, _inputs, value, q=q):
            captures[q] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                ids, mask, positions = baseline.pad_right([row["future_prompt_ids"] for row in batch], device, pad)
                captures.clear(); model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                ends = mask.sum(dim=1) - 1
                for q in range(len(module_list)):
                    selected = torch.stack([captures[q][local, ends[local]] for local in range(len(batch))])
                    field[start:start + len(batch), q] = selected.float().cpu().numpy().astype(np.float16)
                field.flush(); save(BOUNDARY_PROGRESS, {"completed": start + len(batch), "shape": list(shape)})
                print(f"[phase2335 boundary] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        field.flush(); close_memmap(field)
    return {"path": str(BOUNDARY.relative_to(ROOT)), "shape": list(shape)}


def behavior(scores: list[dict], free: list[dict]) -> dict:
    output = {"families": {}, "qualified": []}
    for family in FAMILIES:
        cells = {}
        passed = True
        for partition in set(PARTITION.values()):
            s = [row for row in scores if row["family"] == family and row["partition"] == partition]
            f = [row for row in free if row["family"] == family and row["partition"] == partition]
            cell = {
                "rows": len(s), "sum_accuracy": float(np.mean([row["correct_by_sum"] for row in s])),
                "mean_accuracy": float(np.mean([row["correct_by_mean"] for row in s])),
                "free_identity_accuracy": float(np.mean([row["first_identity_correct"] for row in f])),
            }
            cell["passed"] = min(cell["sum_accuracy"], cell["mean_accuracy"]) >= 0.70 and cell["free_identity_accuracy"] >= 0.40
            passed = passed and cell["passed"]; cells[partition] = cell
        output["families"][family] = {"qualified": passed, "partitions": cells}
        if passed: output["qualified"].append(family)
    return output


def pair_delta(field: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    lookup = {(row["family"], row["language"], row["surface"], row["unit"], row["state"]): row for row in rows}
    pairs, values = [], []
    for family in FAMILIES:
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    row0 = lookup[(family, language, surface, unit, 0)]
                    row1 = lookup[(family, language, surface, unit, 1)]
                    values.append(field[row1["design_index"]].astype(np.float32) - field[row0["design_index"]].astype(np.float32))
                    pairs.append({"pair_index": len(pairs), "family": family, "language": language, "surface": surface,
                                  "unit": unit, "partition": PARTITION[unit], "state0_case": row0["case_id"], "state1_case": row1["case_id"]})
    return np.stack(values).astype(np.float32), pairs


def sym(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(np.square(left - right, dtype=np.float64)) /
                 ((np.sum(np.square(left, dtype=np.float64)) + np.sum(np.square(right, dtype=np.float64))) / 2 + EPS))


def analyze(delta: np.ndarray, pairs: list[dict], old_delta: np.ndarray, old_pairs: list[dict], behavior_result: dict) -> dict:
    cells = {}
    for family in FAMILIES:
        q = QPOINTS[family]
        old_indices = [i for i, row in enumerate(old_pairs) if row["family"] == family and row["partition"] == "discovery"]
        dev_indices = [i for i, row in enumerate(pairs) if row["family"] == family and row["partition"] == "independent_development"]
        lock_indices = [i for i, row in enumerate(pairs) if row["family"] == family and row["partition"] == "independent_lockbox"]
        old_mean = old_delta[old_indices, q].astype(np.float64).mean(axis=0)
        dev_mean = delta[dev_indices, q].astype(np.float64).mean(axis=0)
        lock_mean = delta[lock_indices, q].astype(np.float64).mean(axis=0)
        wrong_family = FAMILIES[(FAMILIES.index(family) + 1) % len(FAMILIES)]
        wrong_indices = [i for i, row in enumerate(old_pairs) if row["family"] == wrong_family and row["partition"] == "discovery"]
        wrong_mean = old_delta[wrong_indices, q].astype(np.float64).mean(axis=0)
        comparisons = {}
        for name, left, right in (("old_to_development", old_mean, dev_mean), ("old_to_lockbox", old_mean, lock_mean),
                                  ("development_to_lockbox", dev_mean, lock_mean)):
            comparisons[name] = {"sign_agreement": float(np.mean(left * right > 0)), "symmetric_relative_mse": sym(left, right)}
        correct_errors = [np.sum(np.square(delta[i, q].astype(np.float64) - old_mean)) for i in lock_indices]
        wrong_errors = [np.sum(np.square(delta[i, q].astype(np.float64) - wrong_mean)) for i in lock_indices]
        control = {
            "old_correct_beats_wrong_family_fraction": float(np.mean(np.asarray(correct_errors) < np.asarray(wrong_errors))),
            "old_correct_beats_opposite_sign_fraction": float(np.mean([
                np.sum(np.square(delta[i, q].astype(np.float64) - old_mean)) < np.sum(np.square(delta[i, q].astype(np.float64) + old_mean))
                for i in lock_indices
            ])),
        }
        structural = all(row["sign_agreement"] >= 0.65 and row["symmetric_relative_mse"] <= 1.0 for row in comparisons.values())
        semantic = structural and behavior_result["families"][family]["qualified"]
        cells[family] = {"qpoint_frozen": q, "behavior_qualified": behavior_result["families"][family]["qualified"],
                         "comparisons": comparisons, "controls": control,
                         "cross_generator_structural_passed": structural, "semantic_candidate": semantic}
    return {"cells": cells,
            "structural_passed": [f for f, row in cells.items() if row["cross_generator_structural_passed"]],
            "semantic_candidates": [f for f, row in cells.items() if row["semantic_candidate"]],
            "thresholds": {"sign_agreement_min": 0.65, "symmetric_relative_mse_max": 1.0, "all_three_comparisons": True}}


def publish(field: np.ndarray, rows: list[dict], delta: np.ndarray, pairs: list[dict]) -> list[dict]:
    assets = []
    boundary_id = "c6761_qwen4b_independent_construction_boundary"
    boundary_binary = VIS / f"{boundary_id}.float16.npy"
    out = atlas.create_binary(boundary_binary.name, field.shape[0] * field.shape[1], field.shape[2], np.float16)
    out[:] = field.reshape(-1, field.shape[2]); out.flush(); close_memmap(out)
    metadata = [{"case_id": row["case_id"], "family": row["family"], "language": row["language"], "surface": row["surface"],
                 "unit": row["unit"], "state": row["state"], "partition": row["partition"], "qpoint": q,
                 "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == 37 else "block"}
                for row in rows for q in range(38)]
    assets.append(atlas.write_metadata(boundary_id, "Qwen3-4B independent-construction boundary field", boundary_binary, metadata,
                                       "Qwen3-4B-FP16", "embedding_hiddenstate_full_coordinate_v1", "independent replication field",
                                       "896 rows from an independent seven-family generator", "every boundary activation coordinate",
                                       {"coordinate_count": 2560, "all_checkpoints": True, "no_projection": True}))
    delta_id = "c6762_qwen4b_independent_construction_state_delta"
    delta_binary = VIS / f"{delta_id}.float32.npy"
    dout = atlas.create_binary(delta_binary.name, delta.shape[0] * delta.shape[1], delta.shape[2], np.float32)
    dout[:] = delta.reshape(-1, delta.shape[2]); dout.flush(); close_memmap(dout)
    dmeta = [{**pair, "qpoint": q, "checkpoint_kind": "embedding" if q == 0 else "final_norm" if q == 37 else "block"}
             for pair in pairs for q in range(38)]
    assets.append(atlas.write_metadata(delta_id, "Qwen3-4B independent-construction natural state deltas", delta_binary, dmeta,
                                       "Qwen3-4B-FP16", "full_coordinate_natural_state_delta_v1", "independent replication derived",
                                       "448 paired rows from an independent seven-family generator", "state1-state0 in every coordinate",
                                       {"coordinate_count": 2560, "all_checkpoints": True, "no_projection": True}))
    return assets


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    record = rf"""

## Phase {PHASE}: 七族独立构式生成器的反模板全坐标复验（{CAMPAIGN}） [{stamp}]

**测试原理、测试用例与公式。** 根据 Phase2334 路线裁决，另写不复用旧提示模板与旧cue的生成器；覆盖前者/后者等指代、真实问句/陈述句标点、感谢/道歉/请求/问候四类风格、独立伪词典翻译，以及原先行为较强的时间、属性、引语七族。每族16个unit、中英、direct/paraphrase、双状态，共 `{result['material_audit']['rows']}` 行；unit0–7为独立开发，unit8–15为独立lockbox。Qwen3-4B FP16单独加载，运行完整候选、自由生成和全38检查点×2560坐标场。所有qpoint沿用旧阶段冻结值，不在新结果上挑层。

$$
\Delta H^{{new}}_{{f,q}}=H^{{new}}_{{f,1,q}}-H^{{new}}_{{f,0,q}},
$$

$$
\operatorname{{Pass}}_f=\bigwedge_{{(a,b)\in\{{old\to dev,old\to lock,dev\to lock\}}}}
[A_{{sign}}(a,b)\ge0.65\land E_{{sym}}(a,b)\le1.0].
$$

**结果汇总、门槛与相关文件。** 材料审计 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；冻结层复验 `{json.dumps(result['analysis'], ensure_ascii=False)}`；发布 `{json.dumps(result['datasets'], ensure_ascii=False)}`；验证 `{json.dumps(result['verification'], ensure_ascii=False)}`；客户端构建 `{json.dumps(result['frontend_build'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2335_c6761_c6920_independent_construction_replication.py`；结果 `tests/glm5/result/phase2335_c6761_c6920_independent_construction_replication`。

**分析、理论进展、问题硬伤与结论。** 只有旧生成器→独立开发、旧生成器→独立lockbox、独立开发→独立lockbox三项同时满足门槛，才把早层规则从固定词串嫌疑提升为跨构式结构观察；仍需行为资格才能称语义候选。新材料依然是程序生成、没有人类盲评；自然差分同时改变多个token；全场在同一Qwen3-4B、同一FP16格式下，不能代表跨模型不变量。若标点/风格早层规律在此失败，最合理解释是旧固定词汇模板效应，而不是“语言没有编码”；若通过，也只说明分布式坐标场重复。

**下一阶段路线判断。** 若出现跨生成器且有行为资格的候选，目标仍相同，自动顺序运行模型本地相对深度复验，再决定因果调用；若只有结构候选，则优先扩大同族自然类型，不强行闭合；若全部失败，则回到七族中行为通过者的全坐标轨迹与输出边界关系，而不是重复差分搬运。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(record)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    rows, audit = compile_material(); write_rows(MATERIAL, rows)
    freeze = {"frozen_before_model_load": True, "families": list(FAMILIES), "qpoints": QPOINTS,
              "partitions": sorted(set(PARTITION.values())), "thresholds": {"sign": 0.65, "symmetric_mse": 1.0}}
    save(OUT / "config/frozen_contract.json", freeze)
    model = tokenizer = None
    try:
        model, tokenizer, device = model_utils.load_model("qwen3", dtype=torch.float16, use_8bit=False)
        scores = baseline.sequence_scores(model, device, rows, OUT / "behavior/sequence_scores.jsonl", batch_size=20)
        free = baseline.free_generation(model, tokenizer, device, rows, OUT / "behavior/free_generation.jsonl", batch_size=24)
        behavior_result = behavior(scores, free)
        boundary_record = capture_boundary(model, device, rows)
    finally:
        if model is not None: model_utils.release_model(model)
        del model, tokenizer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    field = np.load(BOUNDARY, mmap_mode="r")
    delta, pairs = pair_delta(field, rows); write_rows(OUT / "index/state_pairs.jsonl", pairs)
    old_pairs = read_rows(P2333 / "index/state_pairs.jsonl")
    old = np.load(OLD_DELTA, mmap_mode="r").reshape(len(old_pairs), 38, 2560)
    analysis = analyze(delta, pairs, old, old_pairs, behavior_result)
    datasets = publish(field, rows, delta, pairs)
    verification = [atlas.verify(row) for row in datasets]
    verified = all(all(value for key, value in row.items() if key != "id") for row in verification)
    if not verified: raise RuntimeError(("verification_failed", verification))
    catalog = atlas.update_catalog(datasets); build = atlas.frontend_build()
    if not build["passed"]: raise RuntimeError(("build_failed", build))
    checks = {"material": audit["unique"] and audit["no_replacement"] and audit["target_wrong_distinct"],
              "all_rows": len(rows) == 896, "boundary": boundary_record["shape"] == [896, 38, 2560],
              "frozen_qpoints": True, "assets_verified": verified, "frontend_build": build["passed"]}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material_audit": audit, "freeze": freeze,
              "behavior": behavior_result, "boundary": boundary_record, "analysis": analysis,
              "datasets": json.loads(json.dumps(datasets, ensure_ascii=False, default=str)),
              "verification": verification, "catalog": json.loads(json.dumps(catalog, ensure_ascii=False, default=str)),
              "frontend_build": build, "checks": checks, "all_checks_passed": all(checks.values())}
    save(final_path, result); close_memmap(field); close_memmap(old)
    if not result["all_checks_passed"]: raise RuntimeError(("phase2335_failed", checks))
    append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
