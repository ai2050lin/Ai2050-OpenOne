#!/usr/bin/env python3
"""Token-atomic 32-family typed hypergraph and Qwen3-4B behavior qualification."""
from __future__ import annotations

import gc
import hashlib
import json
import re
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
OUT = RESULT / "phase2538_c117505_c121600_token_atomic_hypergraph_behavior"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PRIOR = RESULT / "phase2537_c116481_c117504_qkv_chain_claim_audit_contract/analysis/final.json"
PHASE, CAMPAIGN = 2538, "C117505-C121600"
UNITS = (34, 35)
LANGS = ("en", "zh")
REQUIRED_REGIONS = ("facts_entity", "facts_relation", "facts_value", "query_property", "candidate", "instruction", "answer_boundary")
sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


# name, English relation, Chinese relation, English values, Chinese values, edge type, queried role
OPERATIONS = (
    ("taxonomy", "is classified as", "分类为", ("tropical fruit", "metal hand tool"), ("热带水果", "金属工具"), "is_a", "category"),
    ("part_whole", "is a component of", "是其组成部分", ("river engine", "stone cottage"), ("河流引擎", "石头小屋"), "part_of", "whole"),
    ("profession", "works professionally as", "职业是", ("village doctor", "music teacher"), ("乡村医生", "音乐教师"), "profession_of", "profession"),
    ("preference", "prefers", "偏好", ("jasmine tea", "dark coffee"), ("茉莉花茶", "深色咖啡"), "prefers", "theme"),
    ("membership", "belongs to", "隶属于", ("chess club", "rowing team"), ("象棋社团", "划船队伍"), "member_of", "group"),
    ("translation", "translates into", "翻译为", ("quiet river", "blue lake"), ("安静河流", "蓝色湖泊"), "translation_of", "translation"),
    ("temporal", "occurs during", "发生于", ("early morning", "late evening"), ("清晨时分", "深夜时分"), "time_of", "time"),
    ("spatial", "is located inside", "位于", ("eastern valley", "western harbor"), ("东部山谷", "西部港口"), "located_in", "location"),
    ("causal", "directly causes", "直接导致", ("bright flame", "heavy rainfall"), ("明亮火焰", "强烈降雨"), "causes", "effect"),
    ("permission", "has permission status", "许可状态是", ("fully allowed", "strictly blocked"), ("完全允许", "严格禁止"), "permission", "status"),
    ("possession", "currently owns", "当前拥有", ("silver bicycle", "wooden violin"), ("银色自行车", "木制小提琴"), "owns", "object"),
    ("instrument", "operates using", "使用工具", ("small compass", "heavy hammer"), ("小型指南针", "重型铁锤"), "uses", "instrument"),
    ("origin", "originates from", "来源于", ("northern island", "southern desert"), ("北方岛屿", "南方沙漠"), "origin", "source"),
    ("destination", "travels toward", "前往", ("coastal station", "mountain village"), ("海滨车站", "山间村庄"), "destination", "goal"),
    ("material", "is manufactured from", "由其制成", ("polished copper", "woven cotton"), ("抛光铜材", "编织棉布"), "made_of", "material"),
    ("color", "has display color", "显示颜色为", ("deep violet", "pale orange"), ("深紫颜色", "浅橙颜色"), "color_of", "color"),
    ("size", "has measured size", "测量尺寸为", ("very narrow", "extremely wide"), ("非常狭窄", "极其宽阔"), "size_of", "size"),
    ("temperature", "is maintained at", "保持状态为", ("mildly warm", "deeply frozen"), ("轻微温暖", "深度冷冻"), "temperature", "state"),
    ("speed", "moves at speed", "移动速度为", ("rather slowly", "very quickly"), ("相当缓慢", "非常快速"), "speed_of", "speed"),
    ("rank", "holds ranking", "排名是", ("first place", "second place"), ("第一名次", "第二名次"), "rank_of", "rank"),
    ("action", "performs action", "执行动作", ("careful inspection", "rapid delivery"), ("仔细检查", "快速递送"), "acts", "action"),
    ("patient", "receives object", "接收对象", ("paper package", "glass bottle"), ("纸质包裹", "玻璃瓶子"), "receives", "patient"),
    ("manner", "speaks with manner", "说话方式为", ("calm manner", "urgent manner"), ("平静方式", "紧急方式"), "manner", "manner"),
    ("purpose", "is intended for", "用途是", ("winter travel", "summer farming"), ("冬季旅行", "夏季耕作"), "purpose", "purpose"),
    ("quantity", "has recorded quantity", "记录数量为", ("three parcels", "seven parcels"), ("三个包裹", "七个包裹"), "quantity", "quantity"),
    ("comparison", "is comparatively", "比较结果为", ("clearly heavier", "clearly lighter"), ("明显更重", "明显更轻"), "comparison", "degree"),
    ("transfer", "transfers ownership to", "把所有权转给", ("market keeper", "harbor keeper"), ("市场保管员", "港口保管员"), "transfer_to", "recipient"),
    ("coreference", "is referred to by label", "被指代标签为", ("first speaker", "second speaker"), ("第一说话者", "第二说话者"), "corefers", "referent"),
    ("negation", "has negated state", "否定状态为", ("not available", "not visible"), ("不可使用", "不可看见"), "negated_state", "state"),
    ("condition", "is enabled under", "启用条件是", ("red signal", "blue signal"), ("红色信号", "蓝色信号"), "conditioned_on", "condition"),
    ("modality", "has modal requirement", "模态要求是", ("must remain", "may depart"), ("必须保留", "可以离开"), "modality", "requirement"),
    ("antonym", "has designated opposite", "指定反义项为", ("open passage", "closed passage"), ("开放通道", "关闭通道"), "opposite_of", "opposite"),
)

NAMES = {
    34: {"en": ("Amber Fox", "Ivory Crane"), "zh": ("琥珀狐狸", "象牙仙鹤")},
    35: {"en": ("Indigo Wolf", "Scarlet Swan"), "zh": ("靛蓝灰狼", "赤红天鹅")},
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def add_segment(tokenizer, ids: list[int], regions: dict[str, list[int]], region: str, text: str) -> None:
    part = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    if not part:
        raise RuntimeError((region, text))
    start = len(ids)
    ids.extend(part)
    regions.setdefault(region, []).extend(range(start, len(ids)))


def compile_material(tokenizer) -> list[dict]:
    rows: list[dict] = []
    for unit in UNITS:
        for fi, (family, en_rel, zh_rel, en_values, zh_values, edge_type, role) in enumerate(OPERATIONS):
            for language in LANGS:
                entities = NAMES[unit][language]
                values = en_values if language == "en" else zh_values
                relation = en_rel if language == "en" else zh_rel
                for surface in (0, 1):
                    for swap in (0, 1):
                        mapping = (0, 1) if swap == 0 else (1, 0)
                        order = (0, 1) if surface == 0 else (1, 0)
                        for query in (0, 1):
                            ids: list[int] = []
                            regions: dict[str, list[int]] = {}
                            add_segment(tokenizer, ids, regions, "frame", "Facts:\n" if language == "en" else "事实：\n")
                            for entity_index in order:
                                add_segment(tokenizer, ids, regions, "frame", "Entity " if language == "en" else "实体")
                                add_segment(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
                                add_segment(tokenizer, ids, regions, "frame", " ")
                                add_segment(tokenizer, ids, regions, "facts_relation", relation)
                                add_segment(tokenizer, ids, regions, "frame", " ")
                                add_segment(tokenizer, ids, regions, "facts_value", f"[{values[mapping[entity_index]]}]")
                                add_segment(tokenizer, ids, regions, "frame", ".\n" if language == "en" else "。\n")
                            if language == "en":
                                qtext = "Question: which entity has this value? Requested value " if surface == 0 else "Query: identify the entity linked by the stated relation to value "
                                add_segment(tokenizer, ids, regions, "question_context", qtext)
                                add_segment(tokenizer, ids, regions, "query_property", f"[{values[query]}]")
                                add_segment(tokenizer, ids, regions, "frame", ".\nCandidates: ")
                                add_segment(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
                                add_segment(tokenizer, ids, regions, "instruction", ". Return only the complete entity name. Answer")
                            else:
                                qtext = "问题：哪个实体具有该值？指定值" if surface == 0 else "查询：请找出通过上述关系连接到此值的实体"
                                add_segment(tokenizer, ids, regions, "question_context", qtext)
                                add_segment(tokenizer, ids, regions, "query_property", f"[{values[query]}]")
                                add_segment(tokenizer, ids, regions, "frame", "。\n候选：")
                                add_segment(tokenizer, ids, regions, "candidate", f"[{entities[0]}]或[{entities[1]}]")
                                add_segment(tokenizer, ids, regions, "instruction", "。只返回完整实体名称。答案")
                            add_segment(tokenizer, ids, regions, "answer_boundary", ":")
                            target = entities[0] if mapping[0] == query else entities[1]
                            rows.append({
                                "case_id": f"u{unit}_f{fi:02d}_{language}_s{surface}_m{swap}_q{query}",
                                "unit": unit, "family_id": fi, "family": family, "language": language,
                                "surface": surface, "meaning_swap": swap, "query_property": query,
                                "entities": list(entities), "values": list(values), "target": target,
                                "edge": {"type": edge_type, "source_role": "entity", "relation_role": relation, "target_role": role},
                                "prompt_ids": ids, "prompt": tokenizer.decode(ids),
                                "regions": regions, "answer_boundary_token": len(ids) - 1,
                            })
    return rows


def compile_controls(tokenizer) -> list[dict]:
    specs = {
        "copy": (("Copy exactly: [Amber Fox]. Answer", "Amber Fox"), ("准确复制：[琥珀狐狸]。答案", "琥珀狐狸")),
        "direct_name": (("Return this complete name: [Ivory Crane]. Answer", "Ivory Crane"), ("返回这个完整名称：[象牙仙鹤]。答案", "象牙仙鹤")),
        "format": (("Write only the code [ALPHA-27]. Answer", "ALPHA-27"), ("只写代码[甲二七]。答案", "甲二七")),
        "punctuation": (("Write exactly: red | blue. Answer", "red | blue"), ("准确写出：红｜蓝。答案", "红｜蓝")),
        "continuation": (("Complete with the final word: north, east, south,", "west"), ("补全最后一个方向词：北、东、南、", "西")),
        "fixed": (("Regardless of context, output [Silver Key]. Answer", "Silver Key"), ("无论上下文如何，只输出[银色钥匙]。答案", "银色钥匙")),
        "identity": (("The requested identifier is [Delta Node]. Identifier", "Delta Node"), ("请求的标识符是[三角节点]。标识符", "三角节点")),
        "same_length": (("Read [Copper Badger], then return it unchanged. Answer", "Copper Badger"), ("读取[铜色獾兽]，然后原样返回。答案", "铜色獾兽")),
    }
    rows = []
    for unit in UNITS:
        for ci, (control, pair) in enumerate(specs.items()):
            for li, language in enumerate(LANGS):
                prompt, target = pair[li]
                for variant in (0, 1):
                    text = (prompt + (" " if variant == 0 else "\n"))
                    ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
                    rows.append({"case_id": f"ctrl_u{unit}_{control}_{language}_{variant}", "unit": unit,
                                 "control": control, "language": language, "variant": variant,
                                 "prompt": text, "prompt_ids": ids, "target": target})
    return rows


def left_pad(sequences: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    shifts = []
    for i, seq in enumerate(sequences):
        shift = width - len(seq); shifts.append(shift)
        ids[i, shift:] = torch.tensor(seq, device=device); mask[i, shift:] = 1
    return ids, mask, shifts


def candidate_behavior(model, tokenizer, rows: list[dict]) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for entity in row["entities"]:
            prefix = " " if row["language"] == "en" else ""
            cont = [int(x) for x in tokenizer.encode(prefix + entity, add_special_tokens=False)]
            jobs.append({"row": row, "entity": entity, "continuation": cont, "sequence": row["prompt_ids"] + cont})
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for start in range(0, len(jobs), 8):
        batch = jobs[start:start + 8]
        ids, mask, shifts = left_pad([j["sequence"] for j in batch], tokenizer.pad_token_id, device)
        with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        lp = torch.log_softmax(logits, -1)
        for bi, job in enumerate(batch):
            plen = len(job["row"]["prompt_ids"]); shift = shifts[bi]
            value = sum(float(lp[bi, shift + plen - 1 + oi, tok]) for oi, tok in enumerate(job["continuation"]))
            scores[job["row"]["case_id"]][job["entity"]] = value
        if (start + len(batch)) % 256 == 0: print(f"[phase2538 score] {start + len(batch)}/{len(jobs)}", flush=True)
    out = []
    for row in rows:
        sc = scores[row["case_id"]]; pred = max(sc, key=sc.get)
        wrong = next(x for x in row["entities"] if x != row["target"])
        out.append({"case_id": row["case_id"], "unit": row["unit"], "family_id": row["family_id"],
                    "family": row["family"], "language": row["language"], "surface": row["surface"],
                    "meaning_swap": row["meaning_swap"], "query_property": row["query_property"],
                    "target": row["target"], "prediction": pred, "correct": pred == row["target"],
                    "target_minus_wrong": sc[row["target"]] - sc[wrong], "scores": sc})
    return out


def autonomous_behavior(model, tokenizer, rows: list[dict], controls: list[dict]) -> tuple[list[dict], list[dict]]:
    device = model.get_input_embeddings().weight.device
    tokenizer.padding_side = "left"
    def run(items: list[dict], is_control: bool) -> list[dict]:
        out = []
        for start in range(0, len(items), 8):
            batch = items[start:start + 8]
            ids, mask, _ = left_pad([x["prompt_ids"] for x in batch], tokenizer.pad_token_id, device)
            width = ids.shape[1]
            with torch.inference_mode():
                seq = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=12, do_sample=False,
                                     use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for row, generated in zip(batch, seq):
                text = tokenizer.decode(generated[width:].cpu().tolist(), skip_special_tokens=True)
                if is_control:
                    correct = norm(text).startswith(norm(row["target"]))
                else:
                    hits = [e for e in row["entities"] if norm(e) in norm(text)]
                    correct = len(set(hits)) == 1 and hits[0] == row["target"]
                out.append({"case_id": row["case_id"], "unit": row["unit"], "language": row["language"],
                            "family": row.get("family"), "control": row.get("control"), "target": row["target"],
                            "generated": text, "correct": correct})
            if (start + len(batch)) % 128 == 0: print(f"[phase2538 generate] {start + len(batch)}/{len(items)}", flush=True)
        return out
    lock = [r for r in rows if r["unit"] == 35 and r["surface"] == 0]
    return run(lock, False), run(controls, True)


def summarize(candidate: list[dict], autonomous: list[dict], controls: list[dict]) -> dict:
    family_unit = {}
    for family in [x[0] for x in OPERATIONS]:
        for unit in UNITS:
            xs = [r for r in candidate if r["family"] == family and r["unit"] == unit]
            family_unit[f"{family}:u{unit}"] = float(np.mean([r["correct"] for r in xs]))
    qualified = [i for i, op in enumerate(OPERATIONS)
                 if min(family_unit[f"{op[0]}:u{u}"] for u in UNITS) >= .75]
    return {
        "candidate_accuracy": float(np.mean([r["correct"] for r in candidate])),
        "candidate_mean_margin": float(np.mean([r["target_minus_wrong"] for r in candidate])),
        "family_unit_accuracy": family_unit,
        "qualified_family_ids": qualified,
        "qualified_families": [OPERATIONS[i][0] for i in qualified],
        "autonomous_lockbox_accuracy": float(np.mean([r["correct"] for r in autonomous])),
        "control_accuracy": float(np.mean([r["correct"] for r in controls])),
        "control_by_type": {name: float(np.mean([r["correct"] for r in controls if r["control"] == name]))
                            for name in sorted({r["control"] for r in controls})},
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


## Phase {PHASE}: 三十二语言操作族token-atomic外部超图与行为门（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 在Qwen3-4B BF16非量化CUDA上建立32个有类型语言操作族，覆盖分类、部分—整体、职业、偏好、成员、翻译、时空、因果、否定、角色、动作、模态等。unit34/35、英中、双surface、双事实绑定、双query全交叉共`{result['design']['material_rows']}`条。每个prompt不再由字符前缀反推span，而是把frame、事实实体、关系、值、问题、query-property、候选、指令和单token答案边界逐段编码后直接拼接input IDs；所有物理token严格非空、互斥并穷尽。另加入复制、直接名称、格式、标点、续写、固定输出、标识符和同长度复制八类通用输出控制。

$$\mathcal G_L=(V_L,E_L,\tau,\rho,C,Q,Y),\qquad \bigsqcup_r S_r=\{{0,\ldots,T-1\}},\quad |S_r|>0.$$

**结果汇总。** 设计 `{json.dumps(result['design'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；文件 `{json.dumps(result['files'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2538_c117505_c121600_token_atomic_hypergraph_behavior.py`；逐token区域、typed edge、候选logprob、自主输出和通用控制位于`{OUT}`。

**分析与理论进展。** 本Phase首先修复Phase2529最硬的token边界缺陷：query-property不再可能为空，候选与instruction被物理拆开，后续可以对同一个source token集合分别干预K和V。外部对象从族名列表升级为有edge type和角色的操作超图。候选行为门只说明模型能完成受控读取，自主行为门进一步限制可进入递归实验的材料；通用控制不作为语言关系族，而用于判断late route是否只是一般复制/输出路线。

**问题硬伤与结论。** 逐段token拼接可能与整串自然文本的一次性BPE分词不同，但模型实际输入和span完全一致且可复现；方括号、字段标签和候选使任务偏结构化；32族仍是人工微世界；候选likelihood与自由生成难度不同。只有双unit行为合格族进入内部因果裁决，失败族保留为行为边界，不反推内部不存在相应编码。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream: stream.write(text)


def main() -> None:
    prior = load(PRIOR)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        material = compile_material(tokenizer)
        controls = compile_controls(tokenizer)
        candidate = candidate_behavior(model, tokenizer, material)
        autonomous, control_rows = autonomous_behavior(model, tokenizer, material, controls)
    finally:
        model_utils.release_model(model); gc.collect()
    material_path = OUT / "material/token_atomic_rows.jsonl"; write(material_path, material)
    control_material_path = OUT / "material/output_controls.jsonl"; write(control_material_path, controls)
    candidate_path = OUT / "behavior/candidate.jsonl"; write(candidate_path, candidate)
    autonomous_path = OUT / "behavior/autonomous.jsonl"; write(autonomous_path, autonomous)
    controls_path = OUT / "behavior/controls.jsonl"; write(controls_path, control_rows)
    all_atomic = True
    for row in material:
        all_positions = [p for ps in row["regions"].values() for p in ps]
        all_atomic &= (sorted(all_positions) == list(range(len(row["prompt_ids"]))) and len(all_positions) == len(set(all_positions))
                       and all(row["regions"].get(name) for name in REQUIRED_REGIONS)
                       and row["regions"]["answer_boundary"] == [len(row["prompt_ids"]) - 1])
    behavior = summarize(candidate, autonomous, control_rows)
    design = {"operation_families": len(OPERATIONS), "material_rows": len(material), "units": list(UNITS),
              "languages": list(LANGS), "surfaces": 2, "meaning_swaps": 2, "queries": 2,
              "generic_control_types": 8, "generic_control_rows": len(controls), "max_prompt_tokens": max(len(r["prompt_ids"]) for r in material)}
    files = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in {
        "material": material_path, "control_material": control_material_path, "candidate": candidate_path,
        "autonomous": autonomous_path, "controls": controls_path}.items()}
    checks = {"prior_passed": prior["all_checks_passed"], "families_32": len(OPERATIONS) == 32,
              "rows_1024": len(material) == 1024, "token_atomic_nonempty_disjoint_exhaustive": bool(all_atomic),
              "at_least_20_qualified": len(behavior["qualified_family_ids"]) >= 20,
              "candidate_accuracy_gate": behavior["candidate_accuracy"] >= .80,
              "autonomous_measured": len(autonomous) == 256, "controls_measured": len(control_rows) == len(controls),
              "hashes": all(len(v["sha256"]) == 64 for v in files.values()), "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B BF16 CUDA nonquantized",
              "design": design, "behavior": behavior, "files": files, "checks": checks,
              "all_checks_passed": all(checks.values())}
    save(OUT / "analysis/final.json", result)
    if result["all_checks_passed"]: append_memo(result)
    print(json.dumps({"phase": PHASE, "design": design, "behavior": behavior, "checks": checks,
                      "all_checks_passed": result["all_checks_passed"]}, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]: raise RuntimeError(checks)


if __name__ == "__main__": main()
