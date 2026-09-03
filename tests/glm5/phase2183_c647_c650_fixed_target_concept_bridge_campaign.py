#!/usr/bin/env python3
"""C647-C650 fixed-target concept identity and translation-bridge campaign.

The campaign reads embeddings, post-block HiddenStates, final norm and output
behavior only.  It deliberately does not inspect attention, MLPs, weights or
gradients, and it never selects coordinates with Top-K/PCA/projection.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2176_c641_c645_translation_relative_encoding_campaign as translation
import phase2181_c646_fresh_translation_embedding_gate_replication as fresh_translation

PHASES = {
    "C647": (2183, "fixed_target_concept_material_behavior_field"),
    "C648": (2184, "concept_identity_readout_and_prospective_bridge"),
    "C649": (2185, "concept_transport_causal_ladder"),
    "C650": (2186, "concept_language_joint_composition_and_audit"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c650_fixed_target_concept_bridge_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c650_selected_concept_response.float16.npy"

FAMILIES = ("fruit", "animal", "object", "nature", "food", "body")
SOURCES = ("zh", "en")
SURFACES = ("explicit_en", "paraphrase_en", "instruction_fr")
PROTOCOLS = ("natural", "code")
ROLES = translation.ROLES
CHECKPOINTS = translation.CHECKPOINTS
DIM = translation.DIM
BEHAVIOR_GATE = 0.80
GAIN_GATE = 0.02
Q_CANDIDATES = (8, 16, 24)
BRIDGE_ROLES = ("source", "query", "boundary")
IDENTITY_ROLES = ("source", "query", "boundary")
QPOINTS = (0, 8, 16, 24, 32, 37)

# These 24 entries did not occur in C641-C646.  They are never used for model
# selection and are revealed only after the confirmation selection is saved.
LOCKBOX = (
    ("fruit", "猕猴桃", "kiwi", "kiwi"),
    ("fruit", "李子", "plum", "prune"),
    ("fruit", "椰子", "coconut", "noix de coco"),
    ("fruit", "杏子", "apricot", "abricot"),
    ("animal", "狐狸", "fox", "renard"),
    ("animal", "鹿", "deer", "cerf"),
    ("animal", "斑马", "zebra", "zèbre"),
    ("animal", "长颈鹿", "giraffe", "girafe"),
    ("object", "铅笔", "pencil", "crayon"),
    ("object", "镜子", "mirror", "miroir"),
    ("object", "雨伞", "umbrella", "parapluie"),
    ("object", "相机", "camera", "appareil photo"),
    ("nature", "湖泊", "lake", "lac"),
    ("nature", "沙漠", "desert", "désert"),
    ("nature", "岛屿", "island", "île"),
    ("nature", "雷声", "thunder", "tonnerre"),
    ("food", "黄油", "butter", "beurre"),
    ("food", "咖啡", "coffee", "café"),
    ("food", "茶", "tea", "thé"),
    ("food", "巧克力", "chocolate", "chocolat"),
    ("body", "手指", "finger", "doigt"),
    ("body", "牙齿", "tooth", "dent"),
    ("body", "脖子", "neck", "cou"),
    ("body", "肩膀", "shoulder", "épaule"),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close_mmap(array: Any) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def out(name: str) -> Path:
    return OUTS[name]


def final(name: str) -> dict:
    return load(out(name) / "analysis/final.json")


def field_path() -> Path:
    return out("C647") / "raw/fixed_target_role_field.float16.npy"


def index_path() -> Path:
    return out("C647") / "raw/hidden_index.jsonl"


def compiled_path() -> Path:
    return out("C647") / "material/compiled.jsonl"


def behavior_path() -> Path:
    return out("C647") / "behavior/behavior.jsonl"


def _records() -> list[dict]:
    sources = (
        ("discovery", tuple(translation.CONCEPTS)),
        ("confirmation", tuple(fresh_translation.FRESH)),
        ("lockbox", LOCKBOX),
    )
    records: list[dict] = []
    for partition, values in sources:
        partition_index = 0
        for family in FAMILIES:
            family_values = [value for value in values if value[0] == family]
            expected = 4 if partition == "lockbox" else 6
            if len(family_values) != expected:
                raise RuntimeError((partition, family, len(family_values), expected))
            for family_rank, value in enumerate(family_values):
                records.append({
                    "concept_uid": f"{partition[0]}_{family}_{family_rank}",
                    "partition": partition,
                    "partition_index": partition_index,
                    "family": family,
                    "family_rank": family_rank,
                    "words": {"zh": value[1], "en": value[2], "fr": value[3]},
                })
                partition_index += 1
    return records


RECORDS = _records()
RECORD_BY_UID = {row["concept_uid"]: row for row in RECORDS}


def _family_records(record: dict) -> list[dict]:
    return [row for row in RECORDS
            if row["partition"] == record["partition"] and row["family"] == record["family"]]


def _candidate_records(record: dict) -> tuple[list[dict], int]:
    family = _family_records(record)
    position = int(record["partition_index"]) % 4
    source_i = family.index(record)
    distractors = [family[(source_i + offset) % len(family)] for offset in range(1, 4)]
    raw = [record, *distractors]
    ordered = raw[1:position + 1] + [raw[0]] + raw[position + 1:]
    return ordered, position


def make_row(record: dict, source: str, surface: str, protocol: str,
             target: str = "fr", prefix: str = "c647") -> dict:
    candidates, gold_position = _candidate_records(record)
    source_word = record["words"][source]
    answer = record["words"][target]
    prompt, anchors = translation.natural_prompt(source_word, source, target, surface)
    system = translation.NATURAL_SYSTEM
    if protocol == "code":
        candidate_words = [row["words"][target] for row in candidates]
        codebook = "; ".join(f"{code} = {word}" for code, word in zip(translation.CODES, candidate_words))
        prompt += f" Codebook: {codebook}. Reply with exactly W, X, Y, or Z."
        answer_candidates = list(translation.CODES)
        answer_value = translation.CODES[gold_position]
        system = translation.CODE_SYSTEM
    else:
        answer_candidates = [row["words"][target] for row in candidates]
        answer_value = answer
    case_id = "|".join((prefix, record["concept_uid"], f"{source}-{target}", surface, protocol))
    return {
        "case_id": case_id,
        "concept_uid": record["concept_uid"],
        "concept_family": record["family"],
        "family_rank": record["family_rank"],
        "partition": record["partition"],
        "source_language": source,
        "target_language": target,
        "surface": surface,
        "protocol": protocol,
        "slice_key": "|".join((record["partition"], protocol, source, target, surface)),
        "prompt": prompt,
        "system": system,
        "source_word": source_word,
        "natural_answer": answer,
        "answer": answer_value,
        "answer_candidates": answer_candidates,
        "gold_position": gold_position,
        "candidate_concepts": [row["concept_uid"] for row in candidates],
        "role_values": anchors,
        "cross_model_subset": (record["partition"] == "lockbox" and
                               source == "zh" and surface == "explicit_en" and
                               protocol == "natural"),
    }


def make_material() -> list[dict]:
    return [make_row(record, source, surface, protocol)
            for record, source, surface, protocol in itertools.product(
                RECORDS, SOURCES, SURFACES, PROTOCOLS)]


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    return translation.compile_rows(tokenizer, rows)


def _protocols() -> dict[str, dict]:
    common = {
        "model": "Qwen3-4B BF16 CUDA; cross-model workers run sequentially only in C650",
        "camera": "embedding + 36 post-block HiddenStates + final norm + output; all 2560 signed coordinates",
        "forbidden": "Attention/MLP/weights/gradients/Top-K/PCA/projection and post-unblind threshold edits",
        "failure_policy": "a failed branch is retained as a boundary; it does not stop registered observations or positive controls",
        "human_review": "frozen template, status NA_pending_external_review unless independent humans actually submit ratings",
    }
    return {
        "C647": {**common,
            "object": "concept identity with target language fixed to French",
            "material": "96 concepts, six families, two source languages, three surfaces, natural and WXYZ output protocols",
            "partitions": "36 prior discovery concepts, 36 independent confirmation concepts, 24 wholly new lockbox concepts",
            "behavior_gate": "candidate and exact free generation each >=0.80 per frozen slice",
            "zero_models": "candidate position, source word, answer identity and surface lexical shortcuts audited before model load",
            "field": "all 1152 cases x 38 checkpoints x six semantic roles x all 2560 coordinates",
        },
        "C648": {**common,
            "object": "separate concept identity readability from prospective q32 response prediction",
            "readout": "centered full-coordinate nearest identity across unseen surfaces/sources; no fitted projection",
            "bridge": "coordinatewise affine and full-coordinate nearest-response models; fit discovery, select confirmation, reveal lockbox once",
            "bridge_sources": {"checkpoints": Q_CANDIDATES, "roles": BRIDGE_ROLES},
            "bridge_target": "same concept-pair contrast at q32 boundary",
            "gate": "confirmation and lockbox NRMSE improve over discovery mean by >=0.02",
        },
        "C649": {**common,
            "object": "generation-level concept substitution under a fixed French target instruction",
            "modes": ["zero", "exact_selected", "all_roles_selected", "cross_surface",
                      "cross_source", "predicted_q32", "exact_q32", "wrong_pair",
                      "wrong_direction", "wrong_role", "wrong_checkpoint"],
            "extra": "dose grid, 16 fixed interleaved coordinate bands and complements, deletion/rescue",
            "interpretation": "individual exact vectors are positive controls; prediction, transportability, sufficiency and necessity remain separate",
        },
        "C650": {**common,
            "object": "joint composition of q0 target-language gate and later concept substitution",
            "factorial": "A-English, A-French, B-English, B-French for every new lockbox pair",
            "causal_order": "q0 language edit is applied before the frozen later concept edit in one forward execution",
            "cross_models": "GLM4, DeepSeek-7B and Qwen3-14B behavior qualify independently before any hidden-state capture",
            "theory_gate": "no new foundational mathematics without behavior, prediction, causal specificity, composition, cross-model and human evidence",
        },
    }


def freeze_all(rows: list[dict]) -> None:
    protocols = _protocols()
    material_hash = digest(rows)
    for name, protocol in protocols.items():
        target = out(name)
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (target / part).mkdir(parents=True, exist_ok=True)
        prereg = target / "protocol/preregistration.json"
        if not prereg.exists():
            save(prereg, {
                "phase": PHASES[name][0], "campaign": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "material_sha256": material_hash,
                "protocol": protocol,
                "dependencies": {"C646": "q0 target-language instruction embedding gate; concept translation unresolved"},
            })
    write_rows(out("C647") / "material/fixed_target_factorial.jsonl", rows)
    write_rows(out("C647") / "external/human_blind_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None,
         "semantic_uniqueness_0_1": None, "translation_equivalence_0_1": None,
         "reviewer": None}
        for row in rows if row["partition"] == "lockbox" and row["protocol"] == "natural"
    ])
    save(out("C647") / "protocol/frozen_campaign_manifest.json", {
        "phase_order": {name: phase for name, (phase, _slug) in PHASES.items()},
        "material_sha256": material_hash,
        "rows": len(rows), "concepts": len(RECORDS),
        "coordinate_policy": "all physical activation coordinates",
        "unblind_order": ["discovery", "confirmation_selection", "lockbox_once"],
    })


def _append_memo(name: str, result: dict) -> None:
    phase, slug = PHASES[name]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    titles = {
        "C647": "固定目标语言的概念材料、行为双门与完整坐标场",
        "C648": "概念身份可读性与未见概念响应的前瞻桥",
        "C649": "概念替换的跨表面运输、剂量、坐标分区与删除救援",
        "C650": "概念身份与目标语言门的联合组合、跨模型与大阶段审计",
    }
    examples = {
        "C647": "目标语言始终固定为法语，例如 `苹果 -> pomme`、`banana -> banane`；新锁箱包括 `猕猴桃 -> kiwi`、`斑马 -> zèbre`、`肩膀 -> épaule`。同一概念跨中文/英文源词、三种自然表面和自然词/WXYZ输出协议正交变化。",
        "C648": "把同一家族内 A→B 的全场差分作为样本，例如 `猕猴桃→李子`。模型只在 discovery 学习从 q8/q16/q24 某角色差分到 q32 边界差分，confirmation 选择路线，随后一次揭示从未用于选择的 lockbox。",
        "C649": "在输入仍询问 A 的法语翻译时，写入 A→B 的个体状态差分，检查自由生成是否从 A 的法语词转为 B 的法语词；并与跨表面、跨源语言、错概念、反方向、错角色、错层和零干预比较。",
        "C650": "以 A 的英语输出为基态，先在 q0 把目标语言改为法语，再在后层写入 A→B 概念差分；目标是 B 的法语词。单独语言编辑应只得到 A-法语，单独概念编辑应只得到 B-英语，联合编辑才可能得到 B-法语。",
    }
    formulas = {
        "C647": r"""$$
\mathcal H(c,s,u,p)=\{H_{q,r,j}(c,s,u,p)\}_{q=0}^{37}{}_{r\in\mathcal R}{}_{j=1}^{2560},
\qquad t\equiv \mathrm{French}
$$
$$
\operatorname{Gate}_{slice}=\mathbf 1[A^{cand}\ge0.80\land A^{gen}\ge0.80]
$$""",
        "C648": r"""$$
D_{ab}^{s,u}(q,r)=H_{q,r}(b,s,u,t=fr)-H_{q,r}(a,s,u,t=fr)
$$
$$
\widehat D_{ab}(32,boundary)=\mu_y+\beta\odot(D_{ab}(q,r)-\mu_x),\quad
\beta_j=\frac{\sum_i(x_{ij}-\mu_{x,j})(y_{ij}-\mu_{y,j})}{\sum_i(x_{ij}-\mu_{x,j})^2+10^{-6}}
$$""",
        "C649": r"""$$
Y_m=F_{\ge q}\!\left(H_q+M_m\odot D_{ab}(q,r)\right),\qquad
M_k(j)=\mathbf 1[j\bmod16=k]
$$
$$
\operatorname{Specific}=A_{correct}-\max(A_{wrong\ pair},A_{wrong\ dir},A_{wrong\ role},A_{wrong\ q},A_{zero})
$$""",
        "C650": r"""$$
M_{c\times t}=H(B,fr)-H(B,en)-H(A,fr)+H(A,en)
$$
$$
Y_{joint}=F\!\left(H(A,en)+\Delta_{target}^{q0}+\Delta_{concept}^{q>0}\right)
$$
$$
\operatorname{Gate}_{new\ math}=\mathbf 1[G_{beh}G_{pred}G_{cause}G_{comp}G_{xmodel}G_{human}]
$$""",
    }
    protocol = load(out(name) / "protocol/preregistration.json")["protocol"]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    section = f"""

## Phase {phase}: {titles[name]} [{stamp}]

**Campaign与证据边界。** `{name}`（`{slug}`）。本期只读取词嵌入、每个block后的HiddenState、final norm和输出；保留全部2560个有符号激活坐标，不读取Attention、MLP、权重或梯度，不使用Top-K、PCA、投影筛选。`all_checks_passed`只表示冻结流程完整运行，不等于机制主张通过。

**运行前冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)}
```

**测试用例。** {examples[name]}

**测试原理与数学对象。**

{formulas[name]}

**详细结果与门槛。**

```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格结论。** {result.get('strict_interpretation', result.get('strict_conclusion', '见上方冻结裁决。'))} 理论主体继续使用“条件化输出场闭合理论”，本期仅更新固定目标语言下的概念身份响应、前瞻桥、可调用状态和联合组合证据，不更改理论名称。激活坐标不是模型权重；可读性、预测性、充分性、必要性、自然生成和跨模型同构互不等价。

**问题、硬伤与瓶颈。** 任务仍是受控单词翻译而非开放句子翻译；中文与法文自然度尚无独立人类盲评，故严格记为`NA_pending_external_review`；旧 discovery/confirmation 词曾在早期Campaign出现，只有24个本期lockbox词是全新材料；小模型可能依赖词典记忆和提示模板；完整坐标降低低值结构被压缩丢失的风险，但不能单独解决可识别性；坐标编号不可跨模型直接比较；端点二阶残差不是顺序计算的证明。

**相关文件。** 主脚本：`tests/glm5/phase2183_c647_c650_fixed_target_concept_bridge_campaign.py`；结果目录：`{out(name).relative_to(ROOT)}`；预注册：`{(out(name) / 'protocol/preregistration.json').relative_to(ROOT)}`；裁决：`{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**结论与下一步授权。** {result.get('next_authorization', '按冻结的大Campaign继续；失败仅淘汰对应解释。')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def _close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": bool(checks) and all(bool(value) for value in checks.values()),
        "checks": checks, **headline, "next_authorization": authorization,
    }
    save(out(name) / "analysis/final.json", result)
    _append_memo(name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _material_audit(rows: list[dict]) -> dict:
    duplicates = {}
    for language in ("zh", "en", "fr"):
        counts = defaultdict(list)
        for record in RECORDS:
            counts[record["words"][language].casefold()].append(record["concept_uid"])
        duplicates[language] = {word: ids for word, ids in counts.items() if len(ids) > 1}
    balance = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        balance[row["slice_key"]][row["gold_position"]] += 1
    return {
        "concepts": len(RECORDS), "rows": len(rows),
        "partition_counts": {part: sum(r["partition"] == part for r in RECORDS)
                             for part in ("discovery", "confirmation", "lockbox")},
        "family_counts": {family: sum(r["family"] == family for r in RECORDS)
                          for family in FAMILIES},
        "duplicate_lexemes_by_language": duplicates,
        "candidate_position_counts": dict(balance),
        "candidate_position_exact_balance": all(len(set(v)) == 1 for v in balance.values()),
        "semantic_uniqueness_machine_audit": all(not value for value in duplicates.values()),
        "human_naturalness": "NA_pending_external_review",
    }


def _capture_field(model, device, compiled: list[dict], behavior_map: dict,
                   slices: dict) -> tuple[list[dict], list[dict]]:
    states = np.lib.format.open_memmap(
        field_path(), mode="w+", dtype=np.float16,
        shape=(len(compiled), CHECKPOINTS, len(ROLES), DIM))
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured: list[torch.Tensor] = []
    handles = [module.register_forward_hook(
        lambda _module, _args, output: captured.append(
            output[0] if isinstance(output, tuple) else output)) for module in modules]
    panel_dir = out("C647") / "raw/full_token_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = {row["case_id"] for row in compiled
                 if row["partition"] == "lockbox" and row["protocol"] == "natural"
                 and row["surface"] == "explicit_en" and row["source_language"] == "zh"}
    index, panels = [], []
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((item["case_id"], len(captured), CHECKPOINTS))
            panel = None; panel_path = None
            if item["case_id"] in panel_ids:
                panel_path = panel_dir / f"row_{row_i:04d}.float16.npy"
                panel = np.lib.format.open_memmap(
                    panel_path, mode="w+", dtype=np.float16,
                    shape=(CHECKPOINTS, len(item["prompt_ids"]), DIM))
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                if panel is not None:
                    panel[q] = values
                for role_i, role in enumerate(ROLES):
                    states[row_i, q, role_i] = values[int(item["role_positions"][role][-1])]
            if panel is not None and panel_path is not None:
                panel.flush()
                panels.append({"case_id": item["case_id"],
                               "path": str(panel_path.relative_to(ROOT)),
                               "shape": list(panel.shape), "bytes": panel_path.stat().st_size})
                close_mmap(panel)
            behavior = behavior_map[item["case_id"]]
            index.append({
                "hidden_index": row_i, "case_id": item["case_id"],
                "concept_uid": item["concept_uid"], "concept_family": item["concept_family"],
                "partition": item["partition"], "source_language": item["source_language"],
                "target_language": item["target_language"], "surface": item["surface"],
                "protocol": item["protocol"], "slice_key": item["slice_key"],
                "role_positions": item["role_positions"],
                "candidate_correct": behavior["candidate_correct"],
                "generated_correct": behavior["generated_correct"],
                "slice_qualified": slices[item["slice_key"]]["qualified"],
            })
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C647 capture] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    states.flush(); close_mmap(states)
    write_rows(index_path(), index)
    save(out("C647") / "raw/full_token_panel_ledger.json", panels)
    return index, panels


def c647(rows: list[dict]) -> None:
    if (out("C647") / "analysis/final.json").exists():
        print("[C647] existing final, skip", flush=True)
        return
    audit = _material_audit(rows)
    save(out("C647") / "audit/material_audit.json", audit)
    if len(rows) != 1152 or not audit["candidate_position_exact_balance"] or not audit["semantic_uniqueness_machine_audit"]:
        raise RuntimeError(("material audit failed", audit))
    model = None
    try:
        model, tokenizer, device, placement = translation.MODEL_BASE.load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        scores_all = translation.base.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=8)
        behavior = []
        for row_i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = translation.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=10)
            prediction, generated_correct = translation.evaluate_generation(text, item)
            behavior.append({
                "case_id": item["case_id"],
                "candidate_prediction": int(np.argmax(scores)),
                "candidate_scores": scores,
                "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                "generated_text": text, "generated_prediction": prediction,
                "generated_correct": generated_correct,
            })
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C647 behavior] {row_i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
        behavior_map = {row["case_id"]: row for row in behavior}
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["slice_key"]].append(behavior_map[row["case_id"]])
        slices = {}
        for key, values in sorted(grouped.items()):
            ca = float(np.mean([value["candidate_correct"] for value in values]))
            ga = float(np.mean([value["generated_correct"] for value in values]))
            slices[key] = {"rows": len(values), "candidate_accuracy": ca,
                           "generated_accuracy": ga,
                           "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
        save(out("C647") / "behavior/slice_qualification.json", slices)
        index, panels = _capture_field(model, device, compiled, behavior_map, slices)
    finally:
        translation.MODEL_BASE.release_bf16(model); gc.collect()
    headline = {
        "material_audit": audit,
        "placement": placement,
        "rows": len(rows), "concepts": len(RECORDS), "slices": len(slices),
        "qualified_slices": sum(value["qualified"] for value in slices.values()),
        "partition_accuracy": {
            partition: {
                "candidate": float(np.mean([behavior[i]["candidate_correct"] for i, row in enumerate(rows)
                                             if row["partition"] == partition])),
                "generation": float(np.mean([behavior[i]["generated_correct"] for i, row in enumerate(rows)
                                              if row["partition"] == partition])),
            } for partition in ("discovery", "confirmation", "lockbox")
        },
        "slice_results": slices,
        "field_shape": [len(rows), CHECKPOINTS, len(ROLES), DIM],
        "full_token_panels": len(panels),
        "human_review": "NA_pending_external_review",
        "strict_interpretation": (
            "This phase establishes a fixed-French behavioral and all-coordinate observation base. "
            "Behavioral success is not evidence that concept identity has been isolated internally."),
    }
    _close("C647", headline, {
        "material_complete": len(rows) == 1152,
        "compiled_complete": len(compiled) == len(rows),
        "behavior_complete": len(behavior) == len(rows),
        "field_complete": len(index) == len(rows) and field_path().exists(),
        "full_coordinate": headline["field_shape"][-1] == DIM,
        "human_review_not_fabricated": True,
        "finite": finite(headline),
    }, "C648按确认集选择概念身份与q32响应桥，随后一次揭示全新词锁箱。")


def _lookup() -> tuple[dict[tuple, dict], dict[str, dict], dict[str, dict]]:
    index_rows = read_rows(index_path())
    lookup = {(row["concept_uid"], row["source_language"], row["surface"], row["protocol"]): row
              for row in index_rows}
    compiled = {row["case_id"]: row for row in read_rows(compiled_path())}
    behavior = {row["case_id"]: row for row in read_rows(behavior_path())}
    return lookup, compiled, behavior


def _formal(row: dict, behavior: dict[str, dict]) -> bool:
    item = behavior[row["case_id"]]
    return bool(row["slice_qualified"] and item["candidate_correct"] and item["generated_correct"])


def _pairs(partition: str) -> list[tuple[dict, dict]]:
    answer = []
    for family in FAMILIES:
        values = sorted([row for row in RECORDS if row["partition"] == partition and row["family"] == family],
                        key=lambda row: row["family_rank"])
        answer.extend((values[i], values[i + 1]) for i in range(0, len(values), 2))
    return answer


def _nrmse(predictions: list[np.ndarray], truths: list[np.ndarray]) -> float | None:
    if not truths:
        return None
    num = sum(float(np.square(pred - truth).sum()) for pred, truth in zip(predictions, truths))
    den = sum(float(np.square(truth).sum()) for truth in truths)
    return float(math.sqrt(num / max(den, 1e-12)))


def _mean_or_none(values: list[Any]) -> float | None:
    return float(np.mean(values)) if values else None


def _signed_agreement(predictions: list[np.ndarray], truths: list[np.ndarray]) -> float | None:
    if not truths:
        return None
    return float(np.mean([np.mean(np.sign(pred) == np.sign(truth))
                          for pred, truth in zip(predictions, truths)]))


def _fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm = x.mean(0); ym = y.mean(0)
    beta = ((x - xm) * (y - ym)).sum(0) / (np.square(x - xm).sum(0) + 1e-6)
    return xm.astype(np.float32), ym.astype(np.float32), beta.astype(np.float32)


def _pair_samples(partition: str, states: np.ndarray, lookup: dict, behavior: dict,
                  q: int, role: str) -> list[dict]:
    role_i = ROLES.index(role); boundary_i = ROLES.index("boundary")
    samples = []
    for pair_i, (a, b) in enumerate(_pairs(partition)):
        for source, surface in itertools.product(SOURCES, SURFACES):
            ar = lookup[(a["concept_uid"], source, surface, "natural")]
            br = lookup[(b["concept_uid"], source, surface, "natural")]
            formal = _formal(ar, behavior) and _formal(br, behavior)
            if not formal:
                continue
            x = (states[br["hidden_index"], q, role_i].astype(np.float32) -
                 states[ar["hidden_index"], q, role_i].astype(np.float32))
            y = (states[br["hidden_index"], 32, boundary_i].astype(np.float32) -
                 states[ar["hidden_index"], 32, boundary_i].astype(np.float32))
            samples.append({"pair_index": pair_i, "a": a["concept_uid"], "b": b["concept_uid"],
                            "source": source, "surface": surface, "x": x, "y": y})
    return samples


def _predict_nn(x: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    distance = np.square(x_train - x[None, :]).sum(1)
    return y_train[int(np.argmin(distance))]


def _identity_accuracy(states: np.ndarray, lookup: dict, behavior: dict, partition: str,
                       q: int, role: str, source_a: str, surface_a: str,
                       source_b: str, surface_b: str) -> dict:
    rows = [record for record in RECORDS if record["partition"] == partition]
    selected = []
    for record in rows:
        a = lookup[(record["concept_uid"], source_a, surface_a, "natural")]
        b = lookup[(record["concept_uid"], source_b, surface_b, "natural")]
        if _formal(a, behavior) and _formal(b, behavior):
            selected.append((record, a, b))
    if len(selected) < 4:
        return {"rows": len(selected), "accuracy": None, "chance": None}
    role_i = ROLES.index(role)
    av = np.stack([states[a["hidden_index"], q, role_i].astype(np.float32) for _r, a, _b in selected])
    bv = np.stack([states[b["hidden_index"], q, role_i].astype(np.float32) for _r, _a, b in selected])
    av -= av.mean(0); bv -= bv.mean(0)
    distance = np.square(av).sum(1)[:, None] + np.square(bv).sum(1)[None, :] - 2.0 * (av @ bv.T)
    prediction = np.argmin(distance, axis=1)
    return {"rows": len(selected), "accuracy": float(np.mean(prediction == np.arange(len(selected)))),
            "chance": 1.0 / len(selected)}


def c648() -> None:
    if (out("C648") / "analysis/final.json").exists():
        print("[C648] existing final, skip", flush=True)
        return
    states = np.load(field_path(), mmap_mode="r")
    lookup, _compiled, behavior = _lookup()

    identity_confirmation = []
    for q, role in itertools.product((8, 16, 24, 32, 37), IDENTITY_ROLES):
        cells = []
        for source in SOURCES:
            for surface in ("paraphrase_en", "instruction_fr"):
                metric = _identity_accuracy(states, lookup, behavior, "confirmation", q, role,
                                            source, "explicit_en", source, surface)
                cells.append({"source_a": source, "surface_a": "explicit_en",
                              "source_b": source, "surface_b": surface, **metric})
            metric = _identity_accuracy(states, lookup, behavior, "confirmation", q, role,
                                        "zh", "explicit_en", "en", "explicit_en")
            cells.append({"source_a": "zh", "surface_a": "explicit_en",
                          "source_b": "en", "surface_b": "explicit_en", **metric})
        valid = [cell["accuracy"] for cell in cells if cell["accuracy"] is not None]
        identity_confirmation.append({"checkpoint": q, "role": role, "mean_accuracy": float(np.mean(valid)) if valid else None,
                                      "cells": cells})
    identity_winner = max(identity_confirmation, key=lambda row: row["mean_accuracy"] or -1.0)
    identity_selection = {"checkpoint": identity_winner["checkpoint"], "role": identity_winner["role"],
                          "confirmation_mean_accuracy": identity_winner["mean_accuracy"]}

    bridge_confirmation = []
    fitted: dict[tuple, tuple] = {}
    for q, role in itertools.product(Q_CANDIDATES, BRIDGE_ROLES):
        train = _pair_samples("discovery", states, lookup, behavior, q, role)
        confirm = _pair_samples("confirmation", states, lookup, behavior, q, role)
        if not train or not confirm:
            continue
        x_train = np.stack([row["x"] for row in train]); y_train = np.stack([row["y"] for row in train])
        truths = [row["y"] for row in confirm]
        mean = y_train.mean(0)
        params = _fit_diagonal(x_train, y_train)
        fitted[(q, role)] = (params, x_train, y_train)
        xm, ym, beta = params
        predictions = {
            "diagonal": [ym + beta * (row["x"] - xm) for row in confirm],
            "nearest_response": [_predict_nn(row["x"], x_train, y_train) for row in confirm],
            "discovery_mean": [mean for _row in confirm],
            "zero": [np.zeros(DIM, np.float32) for _row in confirm],
            "wrong_direction": [-(ym + beta * (row["x"] - xm)) for row in confirm],
        }
        metrics = {name: {"nrmse": _nrmse(pred, truths),
                          "signed_coordinate_agreement": _signed_agreement(pred, truths)}
                   for name, pred in predictions.items()}
        bridge_confirmation.append({"checkpoint": q, "role": role,
                                    "train_rows": len(train), "confirmation_rows": len(confirm),
                                    "metrics": metrics})
    choices = [(row["metrics"][model]["nrmse"], row, model)
               for row in bridge_confirmation for model in ("diagonal", "nearest_response")]
    if not choices:
        raise RuntimeError("no behavior-qualified bridge candidates")
    _score, bridge_row, bridge_model = min(choices, key=lambda value: value[0])
    selection = {
        "identity": identity_selection,
        "bridge": {"checkpoint": bridge_row["checkpoint"], "role": bridge_row["role"],
                   "model": bridge_model,
                   "confirmation": bridge_row["metrics"],
                   "confirmation_gain_over_mean": (bridge_row["metrics"]["discovery_mean"]["nrmse"] -
                                                    bridge_row["metrics"][bridge_model]["nrmse"])},
        "frozen_before_lockbox": True,
    }
    save(out("C648") / "protocol/confirmation_selection_frozen.json", selection)

    # Lockbox is first read for reporting after the selection file exists.
    q = selection["bridge"]["checkpoint"]; role = selection["bridge"]["role"]
    lock = _pair_samples("lockbox", states, lookup, behavior, q, role)
    params, x_train, y_train = fitted[(q, role)]
    xm, ym, beta = params; mean = y_train.mean(0)
    truths = [row["y"] for row in lock]
    bridge_predictions = ([ym + beta * (row["x"] - xm) for row in lock]
                          if bridge_model == "diagonal" else
                          [_predict_nn(row["x"], x_train, y_train) for row in lock])
    lock_metrics = {
        bridge_model: {"nrmse": _nrmse(bridge_predictions, truths),
                       "signed_coordinate_agreement": _signed_agreement(bridge_predictions, truths)},
        "discovery_mean": {"nrmse": _nrmse([mean for _row in lock], truths),
                           "signed_coordinate_agreement": _signed_agreement([mean for _row in lock], truths)},
        "zero": {"nrmse": _nrmse([np.zeros(DIM, np.float32) for _row in lock], truths),
                 "signed_coordinate_agreement": _signed_agreement([np.zeros(DIM, np.float32) for _row in lock], truths)},
    }
    lock_gain = (lock_metrics["discovery_mean"]["nrmse"] - lock_metrics[bridge_model]["nrmse"]
                 if lock_metrics["discovery_mean"]["nrmse"] is not None and
                 lock_metrics[bridge_model]["nrmse"] is not None else None)

    identity_lockbox = []
    iq, ir = identity_selection["checkpoint"], identity_selection["role"]
    for source in SOURCES:
        for surface in ("paraphrase_en", "instruction_fr"):
            identity_lockbox.append({"source_a": source, "surface_a": "explicit_en",
                                     "source_b": source, "surface_b": surface,
                                     **_identity_accuracy(states, lookup, behavior, "lockbox", iq, ir,
                                                          source, "explicit_en", source, surface)})
    identity_lockbox.append({"source_a": "zh", "surface_a": "explicit_en",
                             "source_b": "en", "surface_b": "explicit_en",
                             **_identity_accuracy(states, lookup, behavior, "lockbox", iq, ir,
                                                  "zh", "explicit_en", "en", "explicit_en")})

    if bridge_model == "diagonal":
        np.savez(out("C648") / "raw/selected_bridge_model.npz",
                 model=np.asarray([bridge_model]), checkpoint=np.asarray([q]),
                 role=np.asarray([ROLES.index(role)]), x_mean=xm, y_mean=ym, beta=beta)
    else:
        np.savez(out("C648") / "raw/selected_bridge_model.npz",
                 model=np.asarray([bridge_model]), checkpoint=np.asarray([q]),
                 role=np.asarray([ROLES.index(role)]), x_train=x_train, y_train=y_train)

    passport = np.zeros((CHECKPOINTS, len(ROLES), DIM), np.float32)
    count = np.zeros((CHECKPOINTS, len(ROLES)), np.int32)
    for partition in ("discovery", "confirmation", "lockbox"):
        for a, b in _pairs(partition):
            for source, surface in itertools.product(SOURCES, SURFACES):
                ar = lookup[(a["concept_uid"], source, surface, "natural")]
                br = lookup[(b["concept_uid"], source, surface, "natural")]
                if not (_formal(ar, behavior) and _formal(br, behavior)):
                    continue
                delta = (states[br["hidden_index"]].astype(np.float32) -
                         states[ar["hidden_index"]].astype(np.float32))
                passport += np.square(delta); count += 1
    passport = np.sqrt(passport / np.maximum(count[:, :, None], 1))
    np.save(out("C648") / "raw/full_coordinate_pair_response_rms.float32.npy", passport)
    close_mmap(states)

    confirmation_gain = selection["bridge"]["confirmation_gain_over_mean"]
    bridge_pass = (lock_gain is not None and confirmation_gain >= GAIN_GATE and lock_gain >= GAIN_GATE)
    identity_values = [row["accuracy"] for row in identity_lockbox if row["accuracy"] is not None]
    identity_pass = bool(identity_values) and float(np.mean(identity_values)) >= 0.50
    headline = {
        "identity_confirmation_selection": identity_selection,
        "identity_lockbox_cells": identity_lockbox,
        "identity_lockbox_mean_accuracy": float(np.mean(identity_values)) if identity_values else None,
        "identity_readout_pass": identity_pass,
        "bridge_selection": selection["bridge"],
        "bridge_lockbox_rows": len(lock), "bridge_lockbox_metrics": lock_metrics,
        "bridge_lockbox_gain_over_mean": lock_gain,
        "prospective_bridge_pass": bridge_pass,
        "natural_lockbox_route_status": ("formal" if lock else "NA_behavior_unqualified"),
        "human_review": "NA_pending_external_review",
        "strict_interpretation": (
            "Centered full-coordinate identity retrieval and unseen concept-pair response prediction are distinct. "
            "A readable identity does not prove that the selected state is used; a passing diagonal/nearest bridge "
            "would predict q32 response, not identify a unique coordinate circuit."),
    }
    save(out("C648") / "analysis/identity_confirmation_tournament.json", identity_confirmation)
    save(out("C648") / "analysis/bridge_confirmation_tournament.json", bridge_confirmation)
    _close("C648", headline, {
        "selection_frozen_before_lockbox": (out("C648") / "protocol/confirmation_selection_frozen.json").exists(),
        "confirmation_tournament_complete": len(bridge_confirmation) == len(Q_CANDIDATES) * len(BRIDGE_ROLES),
        "lockbox_empty_route_recorded_without_nan": len(lock) == 0 or all(
            value["nrmse"] is not None for value in lock_metrics.values()),
        "model_saved": (out("C648") / "raw/selected_bridge_model.npz").exists(),
        "all_coordinate_passport": passport.shape == (CHECKPOINTS, len(ROLES), DIM),
        "finite": finite(headline),
    }, "无论预测桥是否通过，C649继续运行个体精确正控、跨表面/跨源语言运输和完整负控；失败只限制解释。")


def _patches(item: dict, q: int, values: list[tuple[str, np.ndarray]]) -> list[dict]:
    return [{"q": q, "position": int(item["role_positions"][role][-1]), "vector": vector}
            for role, vector in values]


def _eval_item(base_item: dict, target_item: dict) -> dict:
    item = dict(base_item)
    item["answer_candidates"] = target_item["answer_candidates"]
    item["gold_position"] = target_item["gold_position"]
    return item


def _selected_prediction(model_data: Any, x: np.ndarray) -> np.ndarray:
    model = str(model_data["model"][0])
    if model == "diagonal":
        return (model_data["y_mean"].astype(np.float32) +
                model_data["beta"].astype(np.float32) *
                (x - model_data["x_mean"].astype(np.float32)))
    return _predict_nn(x, model_data["x_train"].astype(np.float32),
                       model_data["y_train"].astype(np.float32))


def c649() -> None:
    if (out("C649") / "analysis/final.json").exists():
        print("[C649] existing final, skip", flush=True)
        return
    states = np.load(field_path(), mmap_mode="r")
    lookup, compiled, behavior = _lookup()
    selection = load(out("C648") / "protocol/confirmation_selection_frozen.json")["bridge"]
    selected_q = int(selection["checkpoint"]); selected_role = selection["role"]
    selected_role_i = ROLES.index(selected_role); boundary_i = ROLES.index("boundary")
    model_data = np.load(out("C648") / "raw/selected_bridge_model.npz", allow_pickle=False)
    pairs = _pairs("lockbox")
    formal_cases = []
    for pair_i, (a, b) in enumerate(pairs):
        for surface in ("explicit_en", "paraphrase_en"):
            ar = lookup[(a["concept_uid"], "zh", surface, "natural")]
            br = lookup[(b["concept_uid"], "zh", surface, "natural")]
            if _formal(ar, behavior) and _formal(br, behavior):
                formal_cases.append((pair_i, a, b, surface, ar, br))

    model = None; causal = []
    try:
        model, tokenizer, _device, _placement = translation.MODEL_BASE.load_bf16("qwen3")
        wrong_q = {8: 16, 16: 24, 24: 16}[selected_q]
        for case_i, (pair_i, a, b, surface, ar, br) in enumerate(formal_cases):
            base_item = compiled[ar["case_id"]]; target_item = compiled[br["case_id"]]
            item = _eval_item(base_item, target_item)
            exact = (states[br["hidden_index"], selected_q, selected_role_i].astype(np.float32) -
                     states[ar["hidden_index"], selected_q, selected_role_i].astype(np.float32))
            exact_q32 = (states[br["hidden_index"], 32, boundary_i].astype(np.float32) -
                         states[ar["hidden_index"], 32, boundary_i].astype(np.float32))
            all_roles = [(role, states[br["hidden_index"], selected_q, role_i].astype(np.float32) -
                                states[ar["hidden_index"], selected_q, role_i].astype(np.float32))
                         for role_i, role in enumerate(ROLES)]
            donor_surface = "paraphrase_en" if surface == "explicit_en" else "explicit_en"
            dar = lookup[(a["concept_uid"], "zh", donor_surface, "natural")]
            dbr = lookup[(b["concept_uid"], "zh", donor_surface, "natural")]
            cross_surface = (states[dbr["hidden_index"], selected_q, selected_role_i].astype(np.float32) -
                             states[dar["hidden_index"], selected_q, selected_role_i].astype(np.float32))
            ear = lookup[(a["concept_uid"], "en", surface, "natural")]
            ebr = lookup[(b["concept_uid"], "en", surface, "natural")]
            cross_source = (states[ebr["hidden_index"], selected_q, selected_role_i].astype(np.float32) -
                            states[ear["hidden_index"], selected_q, selected_role_i].astype(np.float32))
            wrong_a, wrong_b = pairs[(pair_i + 1) % len(pairs)]
            war = lookup[(wrong_a["concept_uid"], "zh", surface, "natural")]
            wbr = lookup[(wrong_b["concept_uid"], "zh", surface, "natural")]
            wrong_pair = (states[wbr["hidden_index"], selected_q, selected_role_i].astype(np.float32) -
                          states[war["hidden_index"], selected_q, selected_role_i].astype(np.float32))
            predicted_q32 = _selected_prediction(model_data, exact)
            wrong_role = "query" if selected_role != "query" else "instruction"
            modes = {
                "zero": [],
                "exact_selected": _patches(item, selected_q, [(selected_role, exact)]),
                "all_roles_selected": _patches(item, selected_q, all_roles),
                "cross_surface": _patches(item, selected_q, [(selected_role, cross_surface)]),
                "cross_source": _patches(item, selected_q, [(selected_role, cross_source)]),
                "predicted_q32": _patches(item, 32, [("boundary", predicted_q32)]),
                "exact_q32": _patches(item, 32, [("boundary", exact_q32)]),
                "wrong_pair": _patches(item, selected_q, [(selected_role, wrong_pair)]),
                "wrong_direction": _patches(item, selected_q, [(selected_role, -exact)]),
                "wrong_role": _patches(item, selected_q, [(wrong_role, exact)]),
                "wrong_checkpoint": _patches(item, wrong_q, [(selected_role, exact)]),
            }
            for mode, patches in modes.items():
                generated = translation._patched_generate(model, tokenizer, item, patches, max_new_tokens=10)
                causal.append({"pair_index": pair_i, "a": a["concept_uid"], "b": b["concept_uid"],
                               "surface": surface, "mode": mode, **generated})
            if case_i % 4 == 0 or case_i + 1 == len(formal_cases):
                print(f"[C649 main causal] {case_i + 1}/{len(formal_cases)}", flush=True)

        dose = []
        fixed_cases = [row for row in formal_cases if row[3] == "explicit_en"]
        for pair_i, a, b, surface, ar, br in fixed_cases:
            item = _eval_item(compiled[ar["case_id"]], compiled[br["case_id"]])
            vectors = {
                "selected": (states[br["hidden_index"], selected_q, selected_role_i].astype(np.float32) -
                             states[ar["hidden_index"], selected_q, selected_role_i].astype(np.float32)),
                "q32": (states[br["hidden_index"], 32, boundary_i].astype(np.float32) -
                        states[ar["hidden_index"], 32, boundary_i].astype(np.float32)),
            }
            for kind, scale in itertools.product(("selected", "q32"), (0.25, 0.5, 0.75, 1.0, 1.25)):
                qv, role = (selected_q, selected_role) if kind == "selected" else (32, "boundary")
                generated = translation._patched_generate(
                    model, tokenizer, item, _patches(item, qv, [(role, vectors[kind] * scale)]),
                    max_new_tokens=10)
                dose.append({"pair_index": pair_i, "kind": kind, "dose": scale, **generated})
        write_rows(out("C649") / "raw/dose_generation.jsonl", dose)

        bands = []
        for pair_i, a, b, surface, ar, br in fixed_cases:
            item = _eval_item(compiled[ar["case_id"]], compiled[br["case_id"]])
            vector = (states[br["hidden_index"], 32, boundary_i].astype(np.float32) -
                      states[ar["hidden_index"], 32, boundary_i].astype(np.float32))
            for band in range(16):
                mask = (np.arange(DIM) % 16 == band).astype(np.float32)
                for kind, selected_vector in (("band", vector * mask), ("complement", vector * (1.0 - mask))):
                    generated = translation._patched_generate(
                        model, tokenizer, item, _patches(item, 32, [("boundary", selected_vector)]),
                        max_new_tokens=10)
                    bands.append({"pair_index": pair_i, "band": band, "kind": kind, **generated})
        write_rows(out("C649") / "raw/interleaved_band_generation.jsonl", bands)

        deletion = []
        for list_i, (pair_i, a, b, surface, ar, br) in enumerate(fixed_cases):
            target_b = compiled[br["case_id"]]
            eval_a = _eval_item(target_b, compiled[ar["case_id"]])
            delta = (states[br["hidden_index"], 32, boundary_i].astype(np.float32) -
                     states[ar["hidden_index"], 32, boundary_i].astype(np.float32))
            wrong_pair_i, wrong_a, wrong_b, _s, wrong_ar, wrong_br = fixed_cases[(list_i + 1) % len(fixed_cases)]
            wrong_delta = (states[wrong_br["hidden_index"], 32, boundary_i].astype(np.float32) -
                           states[wrong_ar["hidden_index"], 32, boundary_i].astype(np.float32))
            modes = {
                "target_b_baseline_evaluated_as_a": [],
                "delete_exact_to_a": _patches(eval_a, 32, [("boundary", -delta)]),
                "delete_wrong_pair_to_a": _patches(eval_a, 32, [("boundary", -wrong_delta)]),
            }
            for mode, patches in modes.items():
                generated = translation._patched_generate(model, tokenizer, eval_a, patches, max_new_tokens=10)
                deletion.append({"pair_index": pair_i, "wrong_pair_index": wrong_pair_i,
                                 "mode": mode, **generated})
        write_rows(out("C649") / "raw/deletion_generation.jsonl", deletion)
    finally:
        translation.MODEL_BASE.release_bf16(model); gc.collect()
        model_data.close(); close_mmap(states)

    write_rows(out("C649") / "raw/main_causal_generation.jsonl", causal)
    rates = {mode: _mean_or_none([row["correct"] for row in causal if row["mode"] == mode])
             for mode in sorted({row["mode"] for row in causal})}
    dose_rates = {f"{kind}@{scale}": _mean_or_none([row["correct"] for row in dose
                                                     if row["kind"] == kind and row["dose"] == scale])
                  for kind, scale in itertools.product(("selected", "q32"), (0.25, 0.5, 0.75, 1.0, 1.25))}
    band_rates = {kind: [_mean_or_none([row["correct"] for row in bands
                                       if row["kind"] == kind and row["band"] == band])
                         for band in range(16)] for kind in ("band", "complement")}
    deletion_rates = {mode: _mean_or_none([row["correct"] for row in deletion if row["mode"] == mode])
                      for mode in sorted({row["mode"] for row in deletion})}
    best_control = max((rates.get(name) or 0.0) for name in
                       ("zero", "wrong_pair", "wrong_direction", "wrong_role", "wrong_checkpoint"))
    best_exact = max(rates.get("exact_selected") or 0.0, rates.get("exact_q32") or 0.0)
    transport_pass = best_exact >= 0.50 and best_exact - best_control >= 0.25
    cross_context_pass = ((rates.get("cross_surface") or 0.0) >= 0.50 and
                          (rates.get("cross_source") or 0.0) >= 0.50)
    predicted_pass = ((rates.get("predicted_q32") or 0.0) >= 0.50 and
                      (rates.get("predicted_q32") or 0.0) - best_control >= 0.25)
    headline = {
        "selected_checkpoint": selected_q, "selected_role": selected_role,
        "formal_pair_contexts": len(formal_cases), "main_causal_rates": rates,
        "dose_rates": dose_rates, "interleaved_band_rates": band_rates,
        "deletion_rates": deletion_rates,
        "individual_exact_transport_pass": transport_pass,
        "cross_context_transport_pass": cross_context_pass,
        "predicted_state_causal_pass": predicted_pass,
        "strict_interpretation": (
            "Exact individual differences, cross-context donors and a discovery-fitted predicted response are adjudicated separately. "
            "A successful exact q32 edit is sufficiency for this task boundary, not proof of a unique concept code; fixed bands "
            "bracket distributed sufficiency and are not coordinate discovery."),
    }
    _close("C649", headline, {
        "formal_cases_exist": len(formal_cases) > 0,
        "main_modes_complete": len(causal) == len(formal_cases) * 11,
        "dose_complete": len(dose) == len([row for row in formal_cases if row[3] == "explicit_en"]) * 10,
        "bands_complete": len(bands) == len([row for row in formal_cases if row[3] == "explicit_en"]) * 32,
        "deletion_complete": len(deletion) == len([row for row in formal_cases if row[3] == "explicit_en"]) * 3,
        "finite": finite(headline),
    }, "C650继续真实层序联合组合；任何精确正控不得代替未见预测或必要性。")


def _capture_selected(model, device, compiled: list[dict], checkpoints: list[int], path: Path) -> np.memmap:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                     shape=(len(compiled), len(checkpoints), len(ROLES), DIM))
    captured = []
    handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(
        output[0] if isinstance(output, tuple) else output)) for module in modules]
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for qi, q in enumerate(checkpoints):
                values = captured[q][0]
                for role_i, role in enumerate(ROLES):
                    field[row_i, qi, role_i] = values[int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    return field


def _composition_rows() -> list[dict]:
    lockbox = [record for record in RECORDS if record["partition"] == "lockbox"]
    rows = [make_row(record, "zh", "explicit_en", "natural", target=target, prefix="c650")
            for record, target in itertools.product(lockbox, ("en", "fr"))]
    for row in rows:
        row["cross_model_subset"] = True
    return rows


def _composition_key(row: dict) -> tuple:
    return row["concept_uid"], row["target_language"]


def c650() -> None:
    if (out("C650") / "analysis/final.json").exists():
        print("[C650] existing final, skip", flush=True)
        return
    selection = load(out("C648") / "protocol/confirmation_selection_frozen.json")["bridge"]
    selected_q = int(selection["checkpoint"]); selected_role = selection["role"]
    checkpoints = sorted(set((0, selected_q, 32, 37)))
    qmap = {q: i for i, q in enumerate(checkpoints)}
    rows = _composition_rows()
    write_rows(out("C650") / "material/joint_factorial.jsonl", rows)
    model = None; causal = []
    try:
        model, tokenizer, device, placement = translation.MODEL_BASE.load_bf16("qwen3")
        compiled_rows = compile_rows(tokenizer, rows)
        write_rows(out("C650") / "material/compiled.jsonl", compiled_rows)
        scores_all = translation.base.old.previous.c607.batch_candidate_scores(
            model, device, compiled_rows, batch_size=8)
        behavior = []
        for item, scores in zip(compiled_rows, scores_all):
            text = translation.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=10)
            prediction, generated_correct = translation.evaluate_generation(text, item)
            behavior.append({"case_id": item["case_id"], "target_language": item["target_language"],
                             "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                             "generated_text": text, "generated_prediction": prediction,
                             "generated_correct": generated_correct})
        write_rows(out("C650") / "behavior/qwen_behavior.jsonl", behavior)
        behavior_map = {row["case_id"]: row for row in behavior}
        route_metrics = {}
        for target in ("en", "fr"):
            selected = [row for row in behavior if row["target_language"] == target]
            ca = float(np.mean([row["candidate_correct"] for row in selected]))
            ga = float(np.mean([row["generated_correct"] for row in selected]))
            route_metrics[target] = {"rows": len(selected), "candidate_accuracy": ca,
                                     "generated_accuracy": ga,
                                     "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
        field = _capture_selected(model, device, compiled_rows, checkpoints,
                                  out("C650") / "raw/joint_role_field.float16.npy")
        compiled = {row["case_id"]: row for row in compiled_rows}
        row_map = {_composition_key(row): i for i, row in enumerate(compiled_rows)}
        pairs = _pairs("lockbox")
        role_i = ROLES.index(selected_role); target_i = ROLES.index("target_language")
        boundary_i = ROLES.index("boundary")
        interaction_sums = np.zeros((len(checkpoints), len(ROLES), DIM), np.float64)
        interaction_count = 0
        formal_pairs = []
        for pair_i, (a, b) in enumerate(pairs):
            ids = {name: row_map[key] for name, key in {
                "ae": (a["concept_uid"], "en"), "af": (a["concept_uid"], "fr"),
                "be": (b["concept_uid"], "en"), "bf": (b["concept_uid"], "fr")}.items()}
            items = {name: compiled_rows[i] for name, i in ids.items()}
            formal = (all(value["qualified"] for value in route_metrics.values()) and
                      all(behavior_map[item["case_id"]]["candidate_correct"] and
                          behavior_map[item["case_id"]]["generated_correct"] for item in items.values()))
            if not formal:
                continue
            formal_pairs.append((pair_i, a, b, ids, items))
            interaction = (field[ids["bf"]].astype(np.float32) - field[ids["be"]].astype(np.float32) -
                           field[ids["af"]].astype(np.float32) + field[ids["ae"]].astype(np.float32))
            interaction_sums += np.square(interaction); interaction_count += 1
        interaction_rms = np.sqrt(interaction_sums / max(interaction_count, 1)).astype(np.float32)
        np.save(out("C650") / "raw/joint_interaction_rms.float32.npy", interaction_rms)
        for list_i, (pair_i, a, b, ids, items) in enumerate(formal_pairs):
            item = _eval_item(items["ae"], items["bf"])
            language = (field[ids["af"], qmap[0], target_i].astype(np.float32) -
                        field[ids["ae"], qmap[0], target_i].astype(np.float32))
            concept_fr = (field[ids["bf"], qmap[selected_q], role_i].astype(np.float32) -
                          field[ids["af"], qmap[selected_q], role_i].astype(np.float32))
            concept_en = (field[ids["be"], qmap[selected_q], role_i].astype(np.float32) -
                          field[ids["ae"], qmap[selected_q], role_i].astype(np.float32))
            concept_q32 = (field[ids["bf"], qmap[32], boundary_i].astype(np.float32) -
                           field[ids["af"], qmap[32], boundary_i].astype(np.float32))
            exact_joint_q32 = (field[ids["bf"], qmap[32], boundary_i].astype(np.float32) -
                               field[ids["ae"], qmap[32], boundary_i].astype(np.float32))
            wrong_ids = formal_pairs[(list_i + 1) % len(formal_pairs)][3]
            wrong_concept = (field[wrong_ids["bf"], qmap[selected_q], role_i].astype(np.float32) -
                             field[wrong_ids["af"], qmap[selected_q], role_i].astype(np.float32))
            modes = {
                "zero": [],
                "language_only": _patches(item, 0, [("target_language", language)]),
                "concept_only_en": _patches(item, selected_q, [(selected_role, concept_en)]),
                "language_then_concept_fr": (_patches(item, 0, [("target_language", language)]) +
                                             _patches(item, selected_q, [(selected_role, concept_fr)])),
                "language_then_incompatible_en_concept": (_patches(item, 0, [("target_language", language)]) +
                                                           _patches(item, selected_q, [(selected_role, concept_en)])),
                "language_then_concept_q32": (_patches(item, 0, [("target_language", language)]) +
                                              _patches(item, 32, [("boundary", concept_q32)])),
                "exact_joint_q32": _patches(item, 32, [("boundary", exact_joint_q32)]),
                "wrong_language": (_patches(item, 0, [("target_language", -language)]) +
                                   _patches(item, selected_q, [(selected_role, concept_fr)])),
                "wrong_concept": (_patches(item, 0, [("target_language", language)]) +
                                  _patches(item, selected_q, [(selected_role, wrong_concept)])),
            }
            for mode, patches in modes.items():
                generated = translation._patched_generate(model, tokenizer, item, patches, max_new_tokens=10)
                causal.append({"pair_index": pair_i, "a": a["concept_uid"], "b": b["concept_uid"],
                               "mode": mode, **generated})
        write_rows(out("C650") / "raw/joint_causal_generation.jsonl", causal)
        field.flush(); close_mmap(field)
    finally:
        translation.MODEL_BASE.release_bf16(model); gc.collect()

    # Cross-model workers are process-isolated and strictly sequential.
    worker = TESTS / "phase2186_c650_fixed_target_cross_model_worker.py"
    cross_results = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        target = out("C650") / f"external/{model_name}/final.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(worker), "--model", model_name,
                   "--material", str(out("C650") / "material/joint_factorial.jsonl"),
                   "--output", str(target)]
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0 and not target.exists():
            save(target, {"status": "worker_process_error", "returncode": completed.returncode,
                          "hiddenstate_ran": False})
        cross_results[model_name] = load(target)

    rates = {mode: float(np.mean([row["correct"] for row in causal if row["mode"] == mode]))
             for mode in sorted({row["mode"] for row in causal})} if causal else {}
    controls = max((rates.get(name, 0.0) for name in ("zero", "wrong_language", "wrong_concept")), default=0.0)
    composition_pass = (rates.get("language_then_concept_fr", 0.0) >= 0.50 and
                        rates.get("language_then_concept_fr", 0.0) - controls >= 0.25)
    cross_qualified = [name for name, value in cross_results.items() if value.get("status") == "closed"]
    cross_model_pass = len(cross_qualified) >= 2

    # Exact all-coordinate visual payload: one lockbox pair, every checkpoint,
    # semantic role and physical coordinate.  The binary retains float16 values.
    c647_states = np.load(field_path(), mmap_mode="r")
    lookup, _compiled, _behavior = _lookup()
    a, b = _pairs("lockbox")[0]
    ar = lookup[(a["concept_uid"], "zh", "explicit_en", "natural")]
    br = lookup[(b["concept_uid"], "zh", "explicit_en", "natural")]
    selected_response = (c647_states[br["hidden_index"]].astype(np.float32) -
                         c647_states[ar["hidden_index"]].astype(np.float32)).astype(np.float16)
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, selected_response)
    atlas = {
        "schema": "ai2050.fixed_target_concept_bridge.v1", "phase": PHASES["C650"][0],
        "campaigns": ["C647", "C648", "C649", "C650"],
        "selected_pair": {"a": a, "b": b, "source": "zh", "target": "fr", "surface": "explicit_en"},
        "checkpoints": list(range(CHECKPOINTS)), "roles": list(ROLES),
        "coordinate_ids": list(range(DIM)),
        "selected_pair_response_shape": list(selected_response.shape),
        "selected_pair_response": np.round(selected_response.astype(np.float32), 6).tolist(),
        "binary_float16": "/vis_data/research_kernel/c650_selected_concept_response.float16.npy",
        "embedding_checkpoint": 0, "post_block_checkpoints": list(range(1, 37)), "final_norm_checkpoint": 37,
        "bridge_selection": selection, "causal_rates": final("C649")["main_causal_rates"],
        "composition_rates": rates, "full_coordinate": True, "no_topk": True,
        "warning": "activation coordinates are not model parameters or cross-model coordinate identities",
    }
    save(VISUAL, atlas)
    close_mmap(c647_states)
    catalog = load(CATALOG)
    entry = {"id": "c650_fixed_target_concept_bridge_atlas",
             "label": "C650 Fixed-Target Concept Bridge Atlas",
             "path": "/vis_data/research_kernel/c650_fixed_target_concept_bridge_atlas.json",
             "binary_path": "/vis_data/research_kernel/c650_selected_concept_response.float16.npy",
             "phase": PHASES["C650"][0], "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [row for row in datasets if row.get("id") != entry["id"]]
    datasets.append(entry); catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)

    # Keep the selected visual tensor and two exact full-token panels only.
    cleanup = {"deleted": [], "retained": [str(VISUAL.relative_to(ROOT)), str(VISUAL_BINARY.relative_to(ROOT))],
               "bytes_deleted": 0}
    panel_ledger = load(out("C647") / "raw/full_token_panel_ledger.json")
    retain_panels = {row["path"] for row in panel_ledger
                     if row["case_id"] in (ar["case_id"], br["case_id"])}
    cleanup["retained"].extend(sorted(retain_panels))
    for row in panel_ledger:
        path = ROOT / row["path"]
        if row["path"] not in retain_panels and path.exists():
            cleanup["bytes_deleted"] += path.stat().st_size
            cleanup["deleted"].append(str(path.relative_to(ROOT))); path.unlink()
    for path in (field_path(), out("C650") / "raw/joint_role_field.float16.npy"):
        if path.exists():
            cleanup["bytes_deleted"] += path.stat().st_size
            cleanup["deleted"].append(str(path.relative_to(ROOT))); path.unlink()
    save(out("C650") / "audit/cleanup.json", cleanup)

    math_gate = bool(final("C647")["qualified_slices"] > 0 and
                     final("C648")["prospective_bridge_pass"] and
                     final("C649")["predicted_state_causal_pass"] and
                     composition_pass and cross_model_pass and False)  # human review is NA
    headline = {
        "qwen_behavior_routes": route_metrics, "qwen_formal_pairs": len(formal_pairs),
        "qwen_joint_causal_rates": rates, "joint_composition_pass": composition_pass,
        "cross_model_results": cross_results, "cross_model_qualified": cross_qualified,
        "cross_model_topology_pass": cross_model_pass,
        "human_review": "NA_pending_external_review",
        "new_foundational_mathematics_gate": math_gate,
        "visual": str(VISUAL.relative_to(ROOT)), "visual_binary": str(VISUAL_BINARY.relative_to(ROOT)),
        "cleanup": cleanup,
        "strict_interpretation": (
            "The campaign distinguishes an input target-language gate, concept identity readability, prospective q32 response "
            "prediction, callable concept substitution and joint composition.  Only the gates reported as passed are retained. "
            "No result licenses a fixed concept vector, unique coordinate gears, universal translation mechanism or new mathematics; "
            "independent human naturalness remains unavailable."),
    }
    _close("C650", headline, {
        "joint_material_complete": len(rows) == 48,
        "qwen_behavior_complete": len(behavior) == 48,
        "joint_causal_complete": len(causal) == len(formal_pairs) * 9,
        "cross_models_attempted_sequentially": len(cross_results) == 3,
        "visual_full_coordinate": selected_response.shape == (CHECKPOINTS, len(ROLES), DIM),
        "cleanup_recorded": (out("C650") / "audit/cleanup.json").exists(),
        "human_review_not_fabricated": True,
        "finite": finite(headline),
    }, "本大阶段已闭合。若目标继续相同，下一阶段应扩展到多词句子翻译与概念关系网络，但只能复用本期真正通过的窄门。")


def main() -> None:
    rows = make_material()
    freeze_all(rows)
    c647(rows)
    c648()
    c649()
    c650()


if __name__ == "__main__":
    main()
