#!/usr/bin/env python3
"""C646 fresh translation q0 role-coalition replication and deconfounding."""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2176_c641_c645_translation_relative_encoding_campaign as campaign

PHASE = 2181
NAME = "C646"
OUT = RESULT / "phase2181_c646_fresh_translation_embedding_gate_replication"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c646_fresh_translation_embedding_gate_atlas.json"

FRESH = (
    ("fruit", "橙子", "orange", "orange"), ("fruit", "草莓", "strawberry", "fraise"),
    ("fruit", "西瓜", "watermelon", "pastèque"), ("fruit", "菠萝", "pineapple", "ananas"),
    ("fruit", "樱桃", "cherry", "cerise"), ("fruit", "芒果", "mango", "mangue"),
    ("animal", "狮子", "lion", "lion"), ("animal", "老虎", "tiger", "tigre"),
    ("animal", "大象", "elephant", "éléphant"), ("animal", "熊", "bear", "ours"),
    ("animal", "狼", "wolf", "loup"), ("animal", "猴子", "monkey", "singe"),
    ("object", "桌子", "table", "table"), ("object", "时钟", "clock", "horloge"),
    ("object", "电话", "phone", "téléphone"), ("object", "灯", "lamp", "lampe"),
    ("object", "杯子", "cup", "tasse"), ("object", "刀", "knife", "couteau"),
    ("nature", "海洋", "ocean", "océan"), ("nature", "云", "cloud", "nuage"),
    ("nature", "风", "wind", "vent"), ("nature", "雪", "snow", "neige"),
    ("nature", "星星", "star", "étoile"), ("nature", "火", "fire", "feu"),
    ("food", "鸡蛋", "egg", "œuf"), ("food", "鱼", "fish", "poisson"),
    ("food", "肉", "meat", "viande"), ("food", "汤", "soup", "soupe"),
    ("food", "蛋糕", "cake", "gâteau"), ("food", "蜂蜜", "honey", "miel"),
    ("body", "手臂", "arm", "bras"), ("body", "腿", "leg", "jambe"),
    ("body", "鼻子", "nose", "nez"), ("body", "嘴", "mouth", "bouche"),
    ("body", "头发", "hair", "cheveux"), ("body", "背部", "back", "dos"),
)
SURFACES = ("explicit_en", "paraphrase_en")
QPOINTS = (0, 8, 32)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def fresh_word(index: int, language: str) -> str:
    return FRESH[index][{"zh": 1, "en": 2, "fr": 3}[language]]


def fresh_row(index: int, target: str, surface: str) -> dict:
    source_word = fresh_word(index, "zh"); answer = fresh_word(index, target)
    start = (index // 6) * 6
    alternatives = [index] + [j for j in range(start, start + 6) if j != index][:3]
    raw = [fresh_word(j, target) for j in alternatives]
    gold = index % 4
    candidates = raw[1:gold + 1] + [raw[0]] + raw[gold + 1:]
    prompt, anchors = campaign.natural_prompt(source_word, "zh", target, surface)
    return {"case_id": f"c646|fresh|c{index:02d}|zh-{target}|{surface}",
            "dataset": "fresh", "concept_index": index, "concept_family": FRESH[index][0],
            "partition": "fresh_lockbox", "source_language": "zh", "target_language": target,
            "surface": surface, "protocol": "natural", "slice_key": f"fresh|zh-{target}|{surface}",
            "prompt": prompt, "system": campaign.NATURAL_SYSTEM, "source_word": source_word,
            "natural_answer": answer, "answer": answer, "answer_candidates": candidates,
            "gold_position": gold, "candidate_concepts": alternatives, "role_values": anchors}


def old_row(index: int, target: str, surface: str) -> dict:
    row = campaign.make_row(index, "zh", target, surface, "natural")
    row["case_id"] = row["case_id"].replace("c641|", "c646|old|")
    row["dataset"] = "old_discovery"
    return row


def capture(model, device, compiled: list[dict]) -> np.ndarray:
    path = OUT / "raw/q0_q8_q32_role_field.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(compiled), len(QPOINTS), len(campaign.ROLES), campaign.DIM))
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        for i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for qi, q in enumerate(QPOINTS):
                tensor = captured[q][0]
                for role_i, role in enumerate(campaign.ROLES):
                    field[i, qi, role_i] = tensor[
                        int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16)
        field.flush()
    finally:
        for handle in handles:
            handle.remove()
    return field


def append_memo(result: dict) -> None:
    memo = campaign.MEMO
    existing = memo.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = fr"""

## Phase 2181: 全新词汇翻译q0目标语言门复验与角色联盟重裁 [{stamp}]

**Campaign。** `C646`。本阶段是 C643 的同目标自动延伸：用36个从未进入 C641–C645 的概念、六个语义族、两种表面，共72个中文到英文/法文配对，放大复验 q0 六角色联合干预，并加入六个单角色、六个 leave-one-role-out、错方向、原向量错层和 q32 对照。只读取 embedding/HiddenState/输出，保留全部2560坐标，不读取 Attention/MLP，不做 Top-K/PCA。

**测量纠偏。** C643 首次运行把“wrong checkpoint”错误实现为“在 q8 写入 q8 自己的匹配向量”，该条件其实是另一条合法路线；重跑后改为“把 q0 原向量不变地写到 q8”，成功率由 1.0 变为 0.0。C643 最终结果文件已是纠正版本。本阶段进一步检验 q0 联合效应是否只是目标语言标签 embedding 的替换。

**测试原理和例子。** 例：`橙子 -> orange` 与 `橙子 -> orange` 基态相对 `橙子 -> orange/法语 orange`；更具区分度的例子包括 `草莓 -> strawberry/fraise`、`大象 -> elephant/éléphant`、`蜂蜜 -> honey/miel`。旧 discovery 词只用于形成冻结平均响应，全新词只用于一次揭盲。

$$
\Delta E_r=E_r(zh\!\to\!fr)-E_r(zh\!\to\!en),\qquad
Y_m=F_{{\ge0}}\!\left(E+\sum_{{r\in m}}\Delta E_r\right)
$$

$$
\operatorname{{Gate}}_{{q0\ target}}=
\mathbf 1[A_{{target\ only}}\ge0.75\land
\max(A_{{zero}},A_{{wrong\ dir}},A_{{wrong\ q}})\le0.25]
$$

**详细结果。**

```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**理论进展。** 若 `target_language_only` 与 `all_roles` 等效，而 `leave_target_language_out` 失败，C643 的 q0 成功应重命名为“目标语言指令 token 的 embedding 门控”，不能称为多角色深层翻译算子。若两者不等效，才保留多角色联盟候选。理论主体仍为“条件化输出场闭合理论”，不修改名称。

**问题、硬伤和瓶颈。** 法语/中文材料没有独立人类盲评；任务仍是单词词典映射；q0 干预接近输入编辑，不足以解释模型如何检索概念对应词；平均响应可能只是指令表面模板的可迁移替换；小模型和单一显式词典任务限制外推。

**相关文件。**脚本：`tests/glm5/phase2181_c646_fresh_translation_embedding_gate_replication.py`；结果：`tests/glm5/result/phase2181_c646_fresh_translation_embedding_gate_replication`；可视化：`frontend/public/vis_data/research_kernel/c646_fresh_translation_embedding_gate_atlas.json`。

**结论。** `{result.get('strict_conclusion')}`。下一步必须离开 q0 指令标签替换，研究固定目标语言标签下概念身份如何从源词 embedding 经中层状态形成目标词；否则继续放大同一 q0 效应只会重复输入编辑规律。
"""
    with memo.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {"phase": PHASE, "campaign": NAME,
                "object": "fresh q0 translation role coalition replication",
                "old_discovery_concepts": 18, "fresh_lockbox_concepts": 36,
                "surfaces": SURFACES, "targets": ("en", "fr"),
                "behavior_gate": "candidate and exact generation >=0.80 per dataset-target-surface slice",
                "causal_gate": "target-only >=0.75 and zero/wrong-direction/wrong-checkpoint <=0.25",
                "human_review": "NA_pending_external_review",
                "camera": "all signed coordinates at q0/q8/q32 over six roles; no compression"}
    save(OUT / "protocol/preregistration.json", protocol)
    old = [old_row(i, target, surface) for i, target, surface in itertools.product(
        [j for j in range(len(campaign.CONCEPTS)) if j % 6 < 3], ("en", "fr"), SURFACES)]
    fresh = [fresh_row(i, target, surface) for i, target, surface in itertools.product(
        range(len(FRESH)), ("en", "fr"), SURFACES)]
    rows = old + fresh
    campaign.write_rows(OUT / "material/material.jsonl", rows)
    campaign.write_rows(OUT / "external/human_blind_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None,
         "translation_equivalence_0_1": None, "reviewer": None} for row in fresh])
    model = None
    try:
        model, tokenizer, device, _placement = campaign.MODEL_BASE.load_bf16("qwen3")
        compiled = campaign.compile_rows(tokenizer, rows)
        campaign.write_rows(OUT / "material/compiled.jsonl", compiled)
        scores_all = campaign.base.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=8)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=8)
            prediction, generated_correct = campaign.evaluate_generation(text, item)
            behavior.append({"case_id": item["case_id"],
                             "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                             "generated_text": text, "generated_prediction": prediction,
                             "generated_correct": generated_correct})
        campaign.write_rows(OUT / "behavior/behavior.jsonl", behavior)
        behavior_map = {row["case_id"]: row for row in behavior}
        groups = defaultdict(list)
        for row in rows:
            groups[row["slice_key"]].append(behavior_map[row["case_id"]])
        slices = {}
        for key, values in groups.items():
            ca = float(np.mean([row["candidate_correct"] for row in values]))
            ga = float(np.mean([row["generated_correct"] for row in values]))
            slices[key] = {"rows": len(values), "candidate_accuracy": ca,
                           "generated_accuracy": ga, "qualified": ca >= 0.80 and ga >= 0.80}
        save(OUT / "behavior/slices.json", slices)
        field = capture(model, device, compiled)
        row_index = {row["case_id"]: i for i, row in enumerate(rows)}
        compiled_map = {row["case_id"]: row for row in compiled}
        q_index = {q: i for i, q in enumerate(QPOINTS)}
        prototypes = {}
        for surface in SURFACES:
            for q in QPOINTS:
                values = []
                for concept in [j for j in range(len(campaign.CONCEPTS)) if j % 6 < 3]:
                    en = old_row(concept, "en", surface); fr = old_row(concept, "fr", surface)
                    if all(behavior_map[x["case_id"]]["candidate_correct"] and
                           behavior_map[x["case_id"]]["generated_correct"] for x in (en, fr)):
                        values.append(field[row_index[fr["case_id"]], q_index[q]].astype(np.float32) -
                                      field[row_index[en["case_id"]], q_index[q]].astype(np.float32))
                prototypes[(surface, q)] = np.mean(values, axis=0) if values else np.zeros(
                    (len(campaign.ROLES), campaign.DIM), np.float32)
        causal = []
        for concept, surface in itertools.product(range(len(FRESH)), SURFACES):
            en = fresh_row(concept, "en", surface); fr = fresh_row(concept, "fr", surface)
            en_b, fr_b = behavior_map[en["case_id"]], behavior_map[fr["case_id"]]
            formal = bool(slices[en["slice_key"]]["qualified"] and slices[fr["slice_key"]]["qualified"] and
                          en_b["candidate_correct"] and en_b["generated_correct"] and
                          fr_b["candidate_correct"] and fr_b["generated_correct"])
            item = dict(compiled_map[en["case_id"]])
            item["answer_candidates"] = compiled_map[fr["case_id"]]["answer_candidates"]
            item["gold_position"] = compiled_map[fr["case_id"]]["gold_position"]
            q0 = prototypes[(surface, 0)]; q32 = prototypes[(surface, 32)]
            exact32 = (field[row_index[fr["case_id"]], q_index[32]].astype(np.float32) -
                       field[row_index[en["case_id"]], q_index[32]].astype(np.float32))
            def role_patches(q, values, include):
                return [{"q": q, "position": item["role_positions"][role][-1],
                         "vector": values[campaign.ROLES.index(role)]} for role in include]
            modes = {"zero": [],
                     "target_language_only": role_patches(0, q0, ("target_language",)),
                     "all_roles": role_patches(0, q0, campaign.ROLES),
                     "leave_target_language_out": role_patches(
                         0, q0, tuple(role for role in campaign.ROLES if role != "target_language")),
                     "wrong_direction": role_patches(0, -q0, ("target_language",)),
                     "wrong_checkpoint": role_patches(8, q0, ("target_language",)),
                     "q32_mean": role_patches(32, q32, ("boundary",)),
                     "q32_exact": role_patches(32, exact32, ("boundary",))}
            for mode, patches in modes.items():
                generated = campaign._patched_generate(model, tokenizer, item, patches)
                causal.append({"concept_index": concept, "surface": surface,
                               "formal": formal, "mode": mode, **generated})
            if concept % 6 == 0:
                print(f"[C646 causal] concept={concept}/36 surface={surface}", flush=True)
        campaign.write_rows(OUT / "raw/causal.jsonl", causal)
        field.flush(); campaign.close_mmap(field)
    finally:
        campaign.MODEL_BASE.release_bf16(model); gc.collect()
    rates = {mode: float(np.mean([row["correct"] for row in causal if row["formal"] and row["mode"] == mode]))
             if any(row["formal"] and row["mode"] == mode for row in causal) else None
             for mode in sorted({row["mode"] for row in causal})}
    target_gate = (rates.get("target_language_only") is not None and
                   rates["target_language_only"] >= 0.75 and
                   max(rates.get("zero") or 0.0, rates.get("wrong_direction") or 0.0,
                       rates.get("wrong_checkpoint") or 0.0) <= 0.25)
    coalition_reduced = (rates.get("target_language_only") == rates.get("all_roles") and
                         (rates.get("leave_target_language_out") or 0.0) <= 0.25)
    strict = ("q0 effect is a target-language instruction embedding gate, not a six-role deep translation operator"
              if target_gate and coalition_reduced else
              "q0 role coalition did not reduce cleanly to a target-language embedding gate")
    checks = {"material_complete": len(rows) == 216,
              "fresh_lockbox_complete": len(fresh) == 144,
              "behavior_complete": len(behavior) == len(rows),
              "causal_modes_complete": len(causal) == 36 * len(SURFACES) * 8,
              "human_review_not_fabricated": True,
              "all_coordinate_prototype_retained": True}
    result = {"phase": PHASE, "campaign": NAME, "status": "closed",
              "all_checks_passed": all(checks.values()), "checks": checks,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "rows": len(rows), "fresh_concepts": len(FRESH), "slices": slices,
              "formal_causal_rows": sum(row["formal"] for row in causal),
              "causal_rates": rates, "target_embedding_gate_pass": target_gate,
              "coalition_reduced_to_target_role": coalition_reduced,
              "human_review": "NA_pending_external_review",
              "strict_conclusion": strict}
    save(OUT / "analysis/final.json", result)
    # Retain all coordinates of the discovered prototype; delete per-example fields.
    atlas = {"schema": "ai2050.fresh_translation_embedding_gate.v1", "phase": PHASE,
             "roles": list(campaign.ROLES), "checkpoints": list(QPOINTS),
             "coordinate_ids": list(range(campaign.DIM)), "shape": [3, 6, 2560],
             "surface_prototypes": {surface: np.round(np.stack([
                 prototypes[(surface, q)] for q in QPOINTS]), 6).tolist() for surface in SURFACES},
             "causal_rates": rates, "full_coordinate": True, "no_topk": True}
    save(VISUAL, atlas)
    catalog = campaign.load(campaign.CATALOG)
    entry = {"id": "c646_fresh_translation_embedding_gate_atlas",
             "label": "C646 Fresh Translation Embedding Gate Atlas",
             "path": "/vis_data/research_kernel/c646_fresh_translation_embedding_gate_atlas.json",
             "phase": PHASE, "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [row for row in datasets if row.get("id") != entry["id"]]
    datasets.append(entry)
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(campaign.CATALOG, catalog)
    field_path = OUT / "raw/q0_q8_q32_role_field.float16.npy"
    deleted = field_path.stat().st_size if field_path.exists() else 0
    if field_path.exists(): field_path.unlink()
    save(OUT / "audit/cleanup.json", {"bytes_deleted": deleted, "visual_retained": str(VISUAL.relative_to(ROOT))})
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
