#!/usr/bin/env python3
"""C641-C645 multilingual translation relative-encoding campaign."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2170_c635_c637_fresh_code_lockbox_campaign as previous

base = previous.base
MODEL_BASE = previous.MODEL_BASE
PHASES = {
    "C641": (2176, "multilingual_translation_material_behavior_field"),
    "C642": (2177, "translation_factorial_response_laws"),
    "C643": (2178, "upstream_q32_translation_bridge"),
    "C644": (2179, "translation_triangle_crossmodel_topology"),
    "C645": (2180, "translation_visual_theory_audit"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
        for name, (phase, slug) in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c645_translation_relative_encoding_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"

LANGUAGES = ("zh", "en", "fr")
SURFACES = ("explicit_en", "paraphrase_en", "instruction_zh", "instruction_fr")
PROTOCOLS = ("natural", "code")
CODES = ("W", "X", "Y", "Z")
ROLES = ("source", "source_language", "target_language", "instruction", "query", "boundary")
CHECKPOINTS = 38
DIM = 2560
BEHAVIOR_GATE = 0.80
CONTROL_MARGIN = 0.02
QPOINTS = (0, 8, 16, 24, 32, 37)
NATURAL_SYSTEM = (
    "You are a precise dictionary translator. Follow the requested source and target "
    "languages. Reply with only the requested word, without explanation or thinking aloud.")
CODE_SYSTEM = (
    "Solve the requested translation or same-language copy using the supplied codebook. "
    "Reply with exactly one code letter: W, X, Y, or Z.")

LANGUAGE_LABELS = {
    "en": {"zh": "Chinese", "en": "English", "fr": "French"},
    "zh": {"zh": "中文", "en": "英语", "fr": "法语"},
    "fr": {"zh": "chinois", "en": "anglais", "fr": "français"},
}

# Six independent semantic families; each family uses three discovery, one
# confirmation and two lockbox concepts.
CONCEPTS = (
    ("fruit", "苹果", "apple", "pomme"),
    ("fruit", "香蕉", "banana", "banane"),
    ("fruit", "葡萄", "grape", "raisin"),
    ("fruit", "柠檬", "lemon", "citron"),
    ("fruit", "梨", "pear", "poire"),
    ("fruit", "桃", "peach", "pêche"),
    ("animal", "猫", "cat", "chat"),
    ("animal", "狗", "dog", "chien"),
    ("animal", "马", "horse", "cheval"),
    ("animal", "兔子", "rabbit", "lapin"),
    ("animal", "牛", "cow", "vache"),
    ("animal", "羊", "sheep", "mouton"),
    ("object", "椅子", "chair", "chaise"),
    ("object", "门", "door", "porte"),
    ("object", "窗户", "window", "fenêtre"),
    ("object", "书", "book", "livre"),
    ("object", "钥匙", "key", "clé"),
    ("object", "瓶子", "bottle", "bouteille"),
    ("nature", "太阳", "sun", "soleil"),
    ("nature", "月亮", "moon", "lune"),
    ("nature", "河流", "river", "rivière"),
    ("nature", "山", "mountain", "montagne"),
    ("nature", "森林", "forest", "forêt"),
    ("nature", "雨", "rain", "pluie"),
    ("food", "面包", "bread", "pain"),
    ("food", "牛奶", "milk", "lait"),
    ("food", "奶酪", "cheese", "fromage"),
    ("food", "米饭", "rice", "riz"),
    ("food", "盐", "salt", "sel"),
    ("food", "糖", "sugar", "sucre"),
    ("body", "手", "hand", "main"),
    ("body", "脚", "foot", "pied"),
    ("body", "眼睛", "eye", "œil"),
    ("body", "耳朵", "ear", "oreille"),
    ("body", "心脏", "heart", "cœur"),
    ("body", "头", "head", "tête"),
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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def begin(name: str, protocol: dict, dependencies: dict) -> Path:
    out = OUTS[name]
    for part in ("protocol", "material", "behavior", "analysis", "raw", "audit", "external"):
        (out / part).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "dependencies": dependencies,
        "camera": "embedding + all post-block HiddenStates + final norm + output; all signed coordinates; no Top-K/PCA/attention/MLP/weights/gradients",
        "branch_policy": "failure closes only the registered slice or route",
        "human_review_policy": "NA unless independent humans actually fill the frozen template",
    })
    print(f"=== {name} phase={PHASES[name][0]} ===", flush=True)
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": bool(checks) and all(bool(value) for value in checks.values()),
              "headline": headline, "checks": checks,
              "next_authorization": authorization}
    save(OUTS[name] / "analysis/final.json", result)
    append_phase_memo(name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def append_phase_memo(name: str, result: dict) -> None:
    phase, slug = PHASES[name]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    protocol = load(OUTS[name] / "protocol/preregistration.json")["protocol"]
    formulas = {
        "C641": r"""$$
\mathcal H(x)=\{H_{q,r,j}(x)\}_{q=0}^{37}{}_{r\in\mathcal R}{}_{j=1}^{2560},\qquad
\operatorname{Gate}_{s}=\mathbf 1[A^{cand}_{s}\ge0.80\land A^{gen}_{s}\ge0.80]
$$""",
        "C642": r"""$$
R^{lang}(c,s)=\mathcal H(c,zh\!\to\!fr,s)-\mathcal H(c,zh\!\to\!en,s)
$$
$$
M_{c\times t}=H(c_b,fr)-H(c_b,en)-H(c_a,fr)+H(c_a,en)
$$
$$
E_{\triangle}=\big[H(zh\!\to\!fr)-H(zh\!\to\!zh)\big]
-\big[H(zh\!\to\!en)-H(zh\!\to\!zh)+H(en\!\to\!fr)-H(en\!\to\!en)\big]
$$""",
        "C643": r"""$$
\widehat{R}_{32,b}=\mu_y+\beta\odot(H_{q,r}-\mu_x),\qquad
\beta_j=\frac{\sum_i(x_{ij}-\mu_{x,j})(y_{ij}-\mu_{y,j})}
{\sum_i(x_{ij}-\mu_{x,j})^2+10^{-6}}
$$
$$
Y_{do}=F_{\ge q}\!\left(H_q+\Delta H_{q,r}\right)
$$""",
        "C644": r"""$$
T_m(d,r)=\frac{\sqrt{\mathbb E_c[(\Delta H^{(m)}_{d,r,:})^2]}}
{\sqrt{\sum_{r'}\mathbb E_c[(\Delta H^{(m)}_{d,r',:})^2]}}
$$""",
        "C645": r"""$$
\mathcal Z=[\mathcal H]_{\mathcal R},\qquad
\operatorname{Gate}_{new\ foundational\ math}
=\mathbf 1[G_{beh}\land G_{pred}\land G_{cause}\land G_{comp}\land G_{xmodel}\land G_{human}]
$$""",
    }
    titles = {
        "C641": "多语言翻译材料、行为双门与全坐标场",
        "C642": "翻译析因响应、二阶交互与路径三角",
        "C643": "上游状态到q32翻译响应的前瞻桥接与干预",
        "C644": "翻译路径组合与跨模型功能拓扑",
        "C645": "翻译相对编码图谱、清理与理论总审计",
    }
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    section = f"""

## Phase {phase}: {titles[name]} [{timestamp}]

**Campaign。** `{name}`（`{slug}`）。本阶段只读取词嵌入、各 block 后 HiddenState、final norm 与输出；保留全部有符号物理坐标，不读取 Attention/MLP/权重/梯度，不做 Top-K、PCA 或投影压缩。

**测试原理与冻结合同。**

{canonical(protocol)}

**测试用例。** 六个语义族共36个概念，中文、英文、法文构成源语言×目标语言的九格（含同语言复制），四种合法表面和自然词/冻结 WXYZ 两种输出协议。例：`苹果 -> apple`、`苹果 -> pomme`、`apple -> pomme`，并与 `apple -> apple`、错方向、换概念和换表面条件分账。人工自然度模板已冻结；无人实际填写时严格记为 `NA_pending_external_review`。

**数学对象。**

{formulas[name]}

**详细结果与门槛。**

```json
{json.dumps(result["headline"], ensure_ascii=False, indent=2, allow_nan=False)}
```

合同完整性检查：`{canonical(result["checks"])}`。`all_checks_passed={str(result["all_checks_passed"]).lower()}` 只表示该阶段按冻结流程完整运行，不自动等于机制主张通过。

**结果分析与理论进展。**最严格解释以 `strict_interpretation` 字段为准。离散输出正确、全场可预测、路径可调用、组合可交换和跨模型同构是互不等价的证据层；本阶段不把高行为准确率或低预测误差外推为“固定翻译向量”“唯一神经电路”或新基础数学。理论主体仍为“条件化输出场闭合理论”，本轮只增加翻译条件下的状态响应、交互残差、路径与功能拓扑证据，不修改理论名称。

**问题、硬伤与瓶颈。**材料是受控字典翻译而非开放句子翻译；中文/法文自然度尚无独立人类盲评；小模型可能以词典记忆、复制或输出准备完成任务；物理坐标不可跨模型直接对齐；观察性关系即使前瞻复现也不等价于因果机制；全坐标保存降低了遗漏低值参数的风险，但不能单独解决可识别性。

**相关文件。**脚本：`tests/glm5/phase2176_c641_c645_translation_relative_encoding_campaign.py`；结果目录：`{OUTS[name].relative_to(ROOT)}`；预注册：`{(OUTS[name] / 'protocol/preregistration.json').relative_to(ROOT)}`；裁决：`{(OUTS[name] / 'analysis/final.json').relative_to(ROOT)}`。

**结论及下一步授权。** `{result["next_authorization"]}`。失败只淘汰对应路线，不覆盖其他观察分支；只有已冻结的分区、门槛和控制允许进入正式解释。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def close_mmap(array: Any) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def partition(concept_index: int) -> str:
    local = concept_index % 6
    return "discovery" if local < 3 else "confirmation" if local == 3 else "lockbox"


def concept_word(concept_index: int, language: str) -> str:
    offset = {"zh": 1, "en": 2, "fr": 3}[language]
    return CONCEPTS[concept_index][offset]


def family_indices(concept_index: int) -> list[int]:
    start = (concept_index // 6) * 6
    return list(range(start, start + 6))


def natural_prompt(source_word: str, source: str, target: str,
                   surface: str) -> tuple[str, dict[str, str]]:
    if surface == "explicit_en":
        source_label, target_label = LANGUAGE_LABELS["en"][source], LANGUAGE_LABELS["en"][target]
        prompt = (f'Translate the {source_label} word "{source_word}" into {target_label}. '
                  "Reply with only the translated word.")
        anchors = {"source": source_word, "source_language": source_label,
                   "target_language": target_label, "instruction": "Translate",
                   "query": "translated word"}
    elif surface == "paraphrase_en":
        source_label, target_label = LANGUAGE_LABELS["en"][source], LANGUAGE_LABELS["en"][target]
        prompt = (f'How do you say "{source_word}" from {source_label} in {target_label}? '
                  "Answer with the corresponding word only.")
        anchors = {"source": source_word, "source_language": source_label,
                   "target_language": target_label, "instruction": "say",
                   "query": "corresponding word"}
    elif surface == "instruction_zh":
        source_label, target_label = LANGUAGE_LABELS["zh"][source], LANGUAGE_LABELS["zh"][target]
        prompt = f'“{source_word}”是一个{source_label}词。它用{target_label}怎么说？只回答对应词。'
        anchors = {"source": source_word, "source_language": source_label,
                   "target_language": target_label, "instruction": "怎么说",
                   "query": "对应词"}
    else:
        source_label, target_label = LANGUAGE_LABELS["fr"][source], LANGUAGE_LABELS["fr"][target]
        prompt = (f'Comment dit-on le mot {source_label} « {source_word} » en {target_label} ? '
                  "Répondez uniquement par le mot correspondant.")
        anchors = {"source": source_word, "source_language": source_label,
                   "target_language": target_label, "instruction": "Comment dit-on",
                   "query": "mot correspondant"}
    return prompt, anchors


def make_row(concept_index: int, source: str, target: str,
             surface: str, protocol: str) -> dict:
    family = CONCEPTS[concept_index][0]
    source_word = concept_word(concept_index, source)
    answer = concept_word(concept_index, target)
    alternatives = [concept_index]
    for candidate in family_indices(concept_index):
        if candidate != concept_index and len(alternatives) < 4:
            alternatives.append(candidate)
    raw_candidates = [concept_word(index, target) for index in alternatives]
    gold_position = concept_index % 4
    candidates = raw_candidates[1:gold_position + 1] + [raw_candidates[0]] + raw_candidates[gold_position + 1:]
    prompt, anchors = natural_prompt(source_word, source, target, surface)
    system = NATURAL_SYSTEM
    if protocol == "code":
        codebook = "; ".join(f"{code} = {word}" for code, word in zip(CODES, candidates))
        prompt += f" Codebook: {codebook}. Reply with exactly W, X, Y, or Z."
        answer_candidates = list(CODES)
        answer_value = CODES[gold_position]
        system = CODE_SYSTEM
    else:
        answer_candidates = candidates
        answer_value = answer
    return {
        "case_id": (f"c641|c{concept_index:02d}|{source}-{target}|{surface}|{protocol}"),
        "concept_index": concept_index, "concept_family": family,
        "partition": partition(concept_index), "source_language": source,
        "target_language": target, "surface": surface, "protocol": protocol,
        "slice_key": "|".join((protocol, source, target, surface)),
        "prompt": prompt, "system": system, "source_word": source_word,
        "natural_answer": answer, "answer": answer_value,
        "answer_candidates": answer_candidates, "gold_position": gold_position,
        "candidate_concepts": alternatives,
        "role_values": anchors,
        "cross_model_subset": (protocol == "natural" and surface == "explicit_en" and
                               partition(concept_index) == "lockbox" and
                               (source, target) in (("zh", "en"), ("zh", "fr"),
                                                    ("en", "en"), ("en", "fr"))),
    }


def make_material() -> list[dict]:
    return [make_row(concept, source, target, surface, protocol)
            for concept, source, target, surface, protocol in itertools.product(
                range(len(CONCEPTS)), LANGUAGES, LANGUAGES, SURFACES, PROTOCOLS)]


def material_path() -> Path:
    return OUTS["C641"] / "material/translation_factorial.jsonl"


def compiled_path() -> Path:
    return OUTS["C641"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C641"] / "behavior/qwen_behavior.jsonl"


def states_path() -> Path:
    return OUTS["C641"] / "raw/role_field.float16.npy"


def index_path() -> Path:
    return OUTS["C641"] / "raw/hidden_index.jsonl"


def normalize_answer(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    value = unicodedata.normalize("NFC", value).strip().lower()
    value = " ".join(value.split())
    return value.strip(" \t\r\n.,;:!?\"'`()[]{}«»，。；：！？")


def generated_position(text: str, candidates: list[str]) -> int:
    value = normalize_answer(text)
    matches = [i for i, candidate in enumerate(candidates)
               if value == normalize_answer(candidate)]
    return matches[0] if len(matches) == 1 else -1


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = base.old.previous.c607.text_core.chat_ids(tokenizer, row["system"], row["prompt"])
        candidate_ids = [tokenizer.encode(" " + answer, add_special_tokens=False)
                         for answer in row["answer_candidates"]]
        positions = {}
        for role, value in row["role_values"].items():
            spans = base.aligned_spans(tokenizer, row["prompt"], ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, "uncompiled role"))
            # In same-language copy controls the source and target labels are
            # identical.  Their semantic roles are distinguished by order.
            positions[role] = spans[-1] if role == "target_language" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids,
                         "role_positions": positions})
    return compiled


def evaluate_generation(text: str, item: dict) -> tuple[int, bool]:
    prediction = generated_position(text, item["answer_candidates"])
    return prediction, prediction == item["gold_position"]


def capture_field(model, device, compiled: list[dict], behavior: dict[str, dict],
                  slices: dict) -> tuple[list[dict], list[dict]]:
    states = np.lib.format.open_memmap(states_path(), mode="w+", dtype=np.float16,
                                       shape=(len(compiled), CHECKPOINTS, len(ROLES), DIM))
    token_dir = OUTS["C641"] / "raw/full_token_panel"
    token_dir.mkdir(parents=True, exist_ok=True)
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    captured = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handles = [module.register_forward_hook(hook) for module in modules]
    registered = {row["case_id"] for row in compiled
                  if row["partition"] == "lockbox" and row["protocol"] == "natural"
                  and row["surface"] == "explicit_en" and row["source_language"] == "zh"
                  and row["target_language"] in ("en", "fr")}
    index, panels = [], []
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != CHECKPOINTS:
                raise RuntimeError((item["case_id"], len(captured), CHECKPOINTS))
            panel = None
            if item["case_id"] in registered:
                path = token_dir / f"row_{row_i:04d}.float16.npy"
                panel = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float16,
                    shape=(CHECKPOINTS, len(item["prompt_ids"]), DIM))
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                if panel is not None:
                    panel[q] = values
                for role_i, role in enumerate(ROLES):
                    position = int(item["role_positions"][role][-1])
                    states[row_i, q, role_i] = values[position]
            if panel is not None:
                panel.flush()
                panels.append({"case_id": item["case_id"],
                               "path": str(path.relative_to(ROOT)),
                               "shape": list(panel.shape), "bytes": path.stat().st_size})
                close_mmap(panel); del panel
            b = behavior[item["case_id"]]
            index.append({"hidden_index": row_i, "case_id": item["case_id"],
                          "concept_index": item["concept_index"],
                          "concept_family": item["concept_family"],
                          "partition": item["partition"],
                          "source_language": item["source_language"],
                          "target_language": item["target_language"],
                          "surface": item["surface"], "protocol": item["protocol"],
                          "slice_key": item["slice_key"],
                          "role_positions": item["role_positions"],
                          "candidate_correct": b["candidate_correct"],
                          "generated_correct": b["generated_correct"],
                          "slice_qualified": slices[item["slice_key"]]["qualified"]})
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C641 capture] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    states.flush(); close_mmap(states)
    write_rows(index_path(), index)
    save(OUTS["C641"] / "raw/full_token_panel_ledger.json", panels)
    return index, panels


def c641() -> None:
    out = begin("C641", {
        "object": "three-language translation and same-language copy factorial",
        "coverage": {"concepts": 36, "semantic_families": 6,
                     "source_languages": LANGUAGES, "target_languages": LANGUAGES,
                     "surfaces": SURFACES, "protocols": PROTOCOLS},
        "rows": 2592,
        "partition": "per semantic family: concepts 0-2 discovery, 3 confirmation, 4-5 lockbox",
        "behavior_gate": "candidate and exact free generation each >=0.80 per protocol-source-target-surface slice",
        "zero_model": "gold candidate position is exactly balanced over 36 concepts in every slice",
        "field": "all rows x embedding/36 blocks/final x six roles x all 2560 signed coordinates",
        "audit_correction": "C638 is discrete-output equivalence only; C639 partitions are a coarse sufficiency bracket; C636 partition tests were preregistered with a six-sample compute cap.",
    }, {"C640": load(RESULT / "phase2175_c640_identity_equivalence_visual_theory_audit/analysis/final.json")["all_checks_passed"]})
    rows = make_material()
    if len(rows) != 2592 or len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("translation material cardinality failure")
    write_rows(material_path(), rows)
    human = [{"case_id": row["case_id"], "naturalness_1_5": None,
              "semantic_uniqueness_0_1": None, "translation_equivalence_0_1": None,
              "reviewer": None} for row in rows if row["partition"] == "lockbox"]
    write_rows(out / "external/human_blind_template.jsonl", human)
    model = None
    try:
        model, tokenizer, device, placement = MODEL_BASE.load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        scores_all = base.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=8)
        behavior = []
        for row_i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=8)
            candidate = int(np.argmax(scores))
            generated, generated_correct = evaluate_generation(text, item)
            behavior.append({"case_id": item["case_id"],
                             "candidate_prediction": candidate,
                             "candidate_scores": scores,
                             "candidate_correct": candidate == item["gold_position"],
                             "generated_text": text,
                             "generated_prediction": generated,
                             "generated_correct": generated_correct})
            if row_i % 64 == 0 or row_i + 1 == len(compiled):
                print(f"[C641 behavior] {row_i + 1}/{len(compiled)}", flush=True)
        write_rows(behavior_path(), behavior)
        by_behavior = {row["case_id"]: row for row in behavior}
        groups = defaultdict(list)
        for row in rows:
            groups[row["slice_key"]].append(by_behavior[row["case_id"]])
        slices = {}
        for key, values in sorted(groups.items()):
            candidate = float(np.mean([value["candidate_correct"] for value in values]))
            generated = float(np.mean([value["generated_correct"] for value in values]))
            slices[key] = {"rows": len(values), "candidate_accuracy": candidate,
                           "generated_accuracy": generated,
                           "qualified": candidate >= BEHAVIOR_GATE and generated >= BEHAVIOR_GATE}
        save(out / "behavior/slice_qualification.json", slices)
        index, panels = capture_field(model, device, compiled, by_behavior, slices)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    zero_counts = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        zero_counts[row["slice_key"]][row["gold_position"]] += 1
    zero_rates = {key: [count / sum(counts) for count in counts]
                  for key, counts in zero_counts.items()}
    save(out / "audit/zero_model_balance.json", zero_rates)
    by_protocol = {protocol: {"candidate_accuracy": float(np.mean([
                                behavior[i]["candidate_correct"] for i, row in enumerate(rows)
                                if row["protocol"] == protocol])),
                              "generated_accuracy": float(np.mean([
                                behavior[i]["generated_correct"] for i, row in enumerate(rows)
                                if row["protocol"] == protocol]))}
                   for protocol in PROTOCOLS}
    headline = {"status": "translation_material_behavior_field_closed",
                "rows": len(rows), "slices": len(slices),
                "qualified_slices": sum(value["qualified"] for value in slices.values()),
                "protocol_accuracy": by_protocol,
                "slice_results": slices,
                "partition_counts": {part: sum(row["partition"] == part for row in rows)
                                     for part in ("discovery", "confirmation", "lockbox")},
                "capture_shape": [len(rows), CHECKPOINTS, len(ROLES), DIM],
                "full_token_panels": len(panels),
                "zero_model_max": max(max(values) for values in zero_rates.values()),
                "human_review": "NA_pending_external_review",
                "material_sha256": digest(rows),
                "strict_interpretation": "Programmatic dictionary uniqueness and exact balance do not substitute for independent human translation review."}
    close("C641", headline, {
        "material_complete": len(rows) == 2592,
        "behavior_complete": len(behavior) == len(rows),
        "field_complete": headline["capture_shape"] == [2592, 38, 6, 2560],
        "zero_balanced": headline["zero_model_max"] == 0.25,
        "panels_complete": len(panels) == 24,
        "human_not_fabricated": all(row["reviewer"] is None for row in human),
        "finite": finite(headline),
    }, "C642_translation_factorial_response_laws")


def _case_map(rows: list[dict]) -> dict[tuple, dict]:
    return {(row["concept_index"], row["source_language"], row["target_language"],
             row["surface"], row["protocol"]): row for row in rows}


def _nrmse_sums(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(np.square(p - t).sum()), float(np.square(t).sum())


def _finish_nrmse(sums: tuple[float, float]) -> float:
    return float(math.sqrt(sums[0] / max(sums[1], 1e-12)))


def c642() -> None:
    out = begin("C642", {
        "object": "full-coordinate factorial translation response laws",
        "primary_response": "H(zh->fr)-H(zh->en) at every checkpoint, role and coordinate",
        "candidate_predictors": ["global_mean", "surface_mean", "family_surface_mean",
                                 "base_conditioned_coordinate_diagonal"],
        "selection": "choose each checkpoint-role predictor on confirmation concepts only; report lockbox once",
        "interaction": "concept x target-language Mobius residual",
        "path_test": "direct zh->fr versus zh->en plus en->fr translation triangle",
        "no_compression": True,
    }, {"C641": final("C641")["all_checks_passed"]})
    rows = read_rows(index_path())
    lookup = _case_map(rows)
    states = np.load(states_path(), mmap_mode="r")
    response_path = out / "raw/target_language_response.float16.npy"
    response = np.lib.format.open_memmap(
        response_path, mode="w+", dtype=np.float16,
        shape=(len(CONCEPTS) * len(SURFACES), CHECKPOINTS, len(ROLES), DIM))
    response_index = []
    base_indices = []
    for concept in range(len(CONCEPTS)):
        for surface in SURFACES:
            en = lookup[(concept, "zh", "en", surface, "natural")]
            fr = lookup[(concept, "zh", "fr", surface, "natural")]
            ri = len(response_index)
            response[ri] = (states[fr["hidden_index"]].astype(np.float32) -
                            states[en["hidden_index"]].astype(np.float32)).astype(np.float16)
            formal = bool(en["slice_qualified"] and fr["slice_qualified"] and
                          en["candidate_correct"] and en["generated_correct"] and
                          fr["candidate_correct"] and fr["generated_correct"])
            response_index.append({"response_index": ri, "concept_index": concept,
                                   "concept_family": CONCEPTS[concept][0],
                                   "partition": partition(concept), "surface": surface,
                                   "formal": formal, "en_case_id": en["case_id"],
                                   "fr_case_id": fr["case_id"]})
            base_indices.append(en["hidden_index"])
    response.flush()
    write_rows(out / "raw/target_language_response_index.jsonl", response_index)

    formal_counts = {part: sum(row["formal"] and row["partition"] == part
                               for row in response_index)
                     for part in ("discovery", "confirmation", "lockbox")}
    metrics = []
    selected = []
    aggregate = defaultdict(lambda: [0.0, 0.0])
    selected_sums = defaultdict(lambda: [0.0, 0.0])
    rolled_sums = defaultdict(lambda: [0.0, 0.0])
    model_names = ("global_mean", "surface_mean", "family_surface_mean",
                   "base_conditioned_coordinate_diagonal")
    for q in range(CHECKPOINTS):
        for role_i, role in enumerate(ROLES):
            train = [row for row in response_index
                     if row["partition"] == "discovery" and row["formal"]]
            global_mean = np.mean([response[row["response_index"], q, role_i].astype(np.float32)
                                   for row in train], axis=0) if train else np.zeros(DIM, np.float32)
            surface_mean, family_surface_mean, diagonal = {}, {}, {}
            for surface in SURFACES:
                group = [row for row in train if row["surface"] == surface]
                surface_mean[surface] = np.mean([
                    response[row["response_index"], q, role_i].astype(np.float32)
                    for row in group], axis=0) if group else global_mean
                for family in sorted({item[0] for item in CONCEPTS}):
                    fg = [row for row in group if row["concept_family"] == family]
                    family_surface_mean[(family, surface)] = np.mean([
                        response[row["response_index"], q, role_i].astype(np.float32)
                        for row in fg], axis=0) if fg else surface_mean[surface]
                if group:
                    x = np.stack([states[base_indices[row["response_index"]], q, role_i].astype(np.float32)
                                  for row in group])
                    y = np.stack([response[row["response_index"], q, role_i].astype(np.float32)
                                  for row in group])
                    xm, ym = x.mean(0), y.mean(0)
                    beta = ((x - xm) * (y - ym)).sum(0) / (
                        np.square(x - xm).sum(0) + 1e-6)
                    diagonal[surface] = (xm, ym, beta)
            cell = {name: {} for name in model_names}
            for split in ("confirmation", "lockbox"):
                targets = [row for row in response_index
                           if row["partition"] == split and row["formal"]]
                sums = {name: [0.0, 0.0] for name in model_names}
                for row in targets:
                    truth = response[row["response_index"], q, role_i].astype(np.float32)
                    x = states[base_indices[row["response_index"]], q, role_i].astype(np.float32)
                    xm, ym, beta = diagonal.get(row["surface"],
                                                (np.zeros(DIM), global_mean, np.zeros(DIM)))
                    predictions = {
                        "global_mean": global_mean,
                        "surface_mean": surface_mean[row["surface"]],
                        "family_surface_mean": family_surface_mean[
                            (row["concept_family"], row["surface"])],
                        "base_conditioned_coordinate_diagonal": ym + beta * (x - xm),
                    }
                    for name, prediction in predictions.items():
                        err, den = _nrmse_sums(prediction, truth)
                        sums[name][0] += err; sums[name][1] += den
                        aggregate[(split, name)][0] += err
                        aggregate[(split, name)][1] += den
                cell_name = {name: _finish_nrmse(tuple(values))
                             for name, values in sums.items()}
                for name, value in cell_name.items():
                    cell[name][split] = value
            winner = min(model_names, key=lambda name: cell[name].get("confirmation", math.inf))
            selected.append({"checkpoint": q, "role": role, "model": winner,
                             "confirmation_nrmse": cell[winner].get("confirmation"),
                             "lockbox_nrmse": cell[winner].get("lockbox")})
            metrics.append({"checkpoint": q, "role": role, "models": cell})
            lockbox = [row for row in response_index
                       if row["partition"] == "lockbox" and row["formal"]]
            actual_pool = [response[row["response_index"], q, role_i].astype(np.float32)
                           for row in lockbox]
            for j, row in enumerate(lockbox):
                truth = actual_pool[j]
                x = states[base_indices[row["response_index"]], q, role_i].astype(np.float32)
                xm, ym, beta = diagonal.get(row["surface"],
                                            (np.zeros(DIM), global_mean, np.zeros(DIM)))
                predictions = {
                    "global_mean": global_mean,
                    "surface_mean": surface_mean[row["surface"]],
                    "family_surface_mean": family_surface_mean[
                        (row["concept_family"], row["surface"])],
                    "base_conditioned_coordinate_diagonal": ym + beta * (x - xm),
                }
                err, den = _nrmse_sums(predictions[winner], truth)
                selected_sums["lockbox"][0] += err; selected_sums["lockbox"][1] += den
                rolled = actual_pool[(j + 1) % len(actual_pool)]
                err, den = _nrmse_sums(rolled, truth)
                rolled_sums["lockbox"][0] += err; rolled_sums["lockbox"][1] += den
    save(out / "analysis/predictor_cells.json", metrics)
    save(out / "analysis/selected_predictor_map.json", selected)

    # Full-coordinate concept x target-language interaction residuals.
    pairs = [(start + a, start + b) for start in range(0, len(CONCEPTS), 6)
             for a, b in ((0, 1), (2, 3), (4, 5))]
    mobius_path = out / "raw/concept_target_mobius.float16.npy"
    mobius = np.lib.format.open_memmap(
        mobius_path, mode="w+", dtype=np.float16,
        shape=(len(pairs) * len(SURFACES) * len(PROTOCOLS), CHECKPOINTS, len(ROLES), DIM))
    mobius_index = []
    for a, b in pairs:
        for surface in SURFACES:
            for protocol in PROTOCOLS:
                ae = lookup[(a, "zh", "en", surface, protocol)]
                af = lookup[(a, "zh", "fr", surface, protocol)]
                be = lookup[(b, "zh", "en", surface, protocol)]
                bf = lookup[(b, "zh", "fr", surface, protocol)]
                mi = len(mobius_index)
                value = (states[bf["hidden_index"]].astype(np.float32) -
                         states[be["hidden_index"]].astype(np.float32) -
                         states[af["hidden_index"]].astype(np.float32) +
                         states[ae["hidden_index"]].astype(np.float32))
                mobius[mi] = value.astype(np.float16)
                cells = (ae, af, be, bf)
                formal = all(cell["slice_qualified"] and cell["candidate_correct"] and
                             cell["generated_correct"] for cell in cells)
                mobius_index.append({"mobius_index": mi, "concept_a": a, "concept_b": b,
                                     "partition": partition(a), "surface": surface,
                                     "protocol": protocol, "formal": formal})
    mobius.flush()
    write_rows(out / "raw/concept_target_mobius_index.jsonl", mobius_index)

    triangle_path = out / "raw/translation_triangle_residual.float16.npy"
    triangle = np.lib.format.open_memmap(
        triangle_path, mode="w+", dtype=np.float16,
        shape=(len(CONCEPTS) * len(SURFACES), CHECKPOINTS, len(ROLES), DIM))
    triangle_index = []
    triangle_sums = defaultdict(lambda: [0.0, 0.0])
    for concept in range(len(CONCEPTS)):
        for surface in SURFACES:
            zz = lookup[(concept, "zh", "zh", surface, "natural")]
            ze = lookup[(concept, "zh", "en", surface, "natural")]
            ee = lookup[(concept, "en", "en", surface, "natural")]
            ef = lookup[(concept, "en", "fr", surface, "natural")]
            zf = lookup[(concept, "zh", "fr", surface, "natural")]
            truth = states[zf["hidden_index"]].astype(np.float32) - states[zz["hidden_index"]].astype(np.float32)
            composed = ((states[ze["hidden_index"]].astype(np.float32) - states[zz["hidden_index"]].astype(np.float32)) +
                        (states[ef["hidden_index"]].astype(np.float32) - states[ee["hidden_index"]].astype(np.float32)))
            residual = truth - composed
            ti = len(triangle_index); triangle[ti] = residual.astype(np.float16)
            cells = (zz, ze, ee, ef, zf)
            formal = all(cell["slice_qualified"] and cell["candidate_correct"] and
                         cell["generated_correct"] for cell in cells)
            if formal:
                err, den = _nrmse_sums(composed, truth)
                triangle_sums[partition(concept)][0] += err
                triangle_sums[partition(concept)][1] += den
            triangle_index.append({"triangle_index": ti, "concept_index": concept,
                                   "partition": partition(concept), "surface": surface,
                                   "formal": formal})
    triangle.flush()
    write_rows(out / "raw/translation_triangle_index.jsonl", triangle_index)

    response_rms = {}
    for part in formal_counts:
        numerator = 0.0; count = 0
        for row in response_index:
            if row["partition"] == part and row["formal"]:
                value = response[row["response_index"]].astype(np.float32)
                numerator += float(np.square(value).sum()); count += value.size
        response_rms[part] = float(np.sqrt(numerator / count)) if count else None
    mobius_rms = {}
    for protocol in PROTOCOLS:
        numerator = 0.0; count = 0
        for row in mobius_index:
            if row["protocol"] == protocol and row["formal"]:
                value = mobius[row["mobius_index"]].astype(np.float32)
                numerator += float(np.square(value).sum()); count += value.size
        mobius_rms[protocol] = float(np.sqrt(numerator / count)) if count else None
    aggregate_report = {f"{split}|{name}": _finish_nrmse(tuple(values))
                        for (split, name), values in aggregate.items()}
    selected_lockbox = _finish_nrmse(tuple(selected_sums["lockbox"]))
    rolled_lockbox = _finish_nrmse(tuple(rolled_sums["lockbox"]))
    headline = {
        "status": "translation_factorial_response_laws_closed",
        "formal_counts": formal_counts, "response_rms": response_rms,
        "predictor_nrmse": aggregate_report,
        "selected_lockbox_nrmse": selected_lockbox,
        "rolled_actual_lockbox_nrmse": rolled_lockbox,
        "mobius_rows": len(mobius_index), "mobius_rms": mobius_rms,
        "triangle_formal": sum(row["formal"] for row in triangle_index),
        "triangle_nrmse_by_partition": {part: _finish_nrmse(tuple(values))
                                         for part, values in triangle_sums.items()},
        "strict_interpretation": (
            "A target-language response or interaction residual is an observational full-field object. "
            "Prediction below controls would support a reusable conditional law, not a unique circuit or new mathematics."),
    }
    save(out / "analysis/summary.json", headline)
    close_mmap(triangle); close_mmap(mobius); close_mmap(response); close_mmap(states)
    close("C642", headline, {
        "response_complete": len(response_index) == 144,
        "mobius_complete": len(mobius_index) == 144,
        "triangle_complete": len(triangle_index) == 144,
        "selection_frozen_before_lockbox_reporting": len(selected) == CHECKPOINTS * len(ROLES),
        "finite": finite(headline),
    }, "C643_upstream_q32_translation_bridge")


def _fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm = x.mean(0); ym = y.mean(0)
    beta = ((x - xm) * (y - ym)).sum(0) / (np.square(x - xm).sum(0) + 1e-6)
    return xm.astype(np.float32), ym.astype(np.float32), beta.astype(np.float32)


def _patched_generate(model, tokenizer, item: dict, patches: list[dict],
                      max_new_tokens: int = 6) -> dict:
    core = model.model
    handles = []
    for patch in patches:
        q = int(patch["q"]); position = int(patch["position"])
        vector = np.asarray(patch["vector"], dtype=np.float32)
        def make_hook(pos_value: int, vector_value: np.ndarray):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                if pos_value < changed.shape[1]:
                    changed[0, pos_value] += torch.as_tensor(
                        vector_value, dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return hook
        module = core.embed_tokens if q == 0 else core.norm if q == 37 else core.layers[q - 1]
        handles.append(module.register_forward_hook(make_hook(position, vector)))
    ids = torch.tensor([item["prompt_ids"]], dtype=torch.long,
                       device=next(model.parameters()).device)
    mask = torch.ones_like(ids); generated = []
    try:
        for _ in range(max_new_tokens):
            pos = mask.long().cumsum(-1) - 1
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                               use_cache=False, return_dict=True).logits
            token = int(torch.argmax(logits[0, -1]).item())
            generated.append(token)
            new = torch.tensor([[token]], dtype=torch.long, device=ids.device)
            ids = torch.cat((ids, new), dim=1)
            mask = torch.cat((mask, torch.ones_like(new)), dim=1)
            if token == tokenizer.eos_token_id:
                break
    finally:
        for handle in handles:
            handle.remove()
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    prediction, correct = evaluate_generation(text, item)
    return {"text": text, "prediction": prediction, "correct": correct,
            "token_ids": generated}


def c643() -> None:
    out = begin("C643", {
        "object": "prospective upstream-to-q32 target-language response bridge",
        "source_candidates": {"checkpoints": [0, 8, 16, 24], "roles": ROLES},
        "target": "q32 boundary full 2560-coordinate zh->fr minus zh->en response",
        "model": "coordinatewise affine bridge fitted on discovery concepts",
        "selection": "lowest confirmation NRMSE, then one lockbox reveal",
        "controls": ["mean", "wrong_direction", "row_roll", "wrong_role"],
        "causal_modes": ["zero", "selected_role", "all_roles", "wrong_direction",
                         "wrong_checkpoint", "q32_discovery_mean", "q32_exact"],
        "interpretation": "causal generation is reported separately from observational prediction",
    }, {"C642": final("C642")["all_checks_passed"]})
    c641_rows = read_rows(index_path()); lookup = _case_map(c641_rows)
    compiled = {row["case_id"]: row for row in read_rows(compiled_path())}
    states = np.load(states_path(), mmap_mode="r")
    response = np.load(OUTS["C642"] / "raw/target_language_response.float16.npy", mmap_mode="r")
    rindex = read_rows(OUTS["C642"] / "raw/target_language_response_index.jsonl")
    base_indices = [lookup[(row["concept_index"], "zh", "en", row["surface"], "natural")]["hidden_index"]
                    for row in rindex]
    train = [row for row in rindex if row["partition"] == "discovery" and row["formal"]]
    confirmation = [row for row in rindex if row["partition"] == "confirmation" and row["formal"]]
    lockbox = [row for row in rindex if row["partition"] == "lockbox" and row["formal"]]
    target_q, target_role = 32, ROLES.index("boundary")
    y_train = np.stack([response[row["response_index"], target_q, target_role].astype(np.float32)
                        for row in train]) if train else np.empty((0, DIM), np.float32)
    y_mean = y_train.mean(0) if len(y_train) else np.zeros(DIM, np.float32)
    tournament = []
    fitted = {}
    for q in (0, 8, 16, 24):
        for role_i, role in enumerate(ROLES):
            x_train = np.stack([states[base_indices[row["response_index"]], q, role_i].astype(np.float32)
                                for row in train]) if train else np.empty((0, DIM), np.float32)
            if len(x_train):
                params = _fit_diagonal(x_train, y_train)
            else:
                params = (np.zeros(DIM, np.float32), y_mean, np.zeros(DIM, np.float32))
            fitted[(q, role)] = params
            split_metrics = {}
            for split_name, split_rows in (("confirmation", confirmation), ("lockbox", lockbox)):
                sums = {name: [0.0, 0.0] for name in ("bridge", "mean", "wrong_direction", "row_roll")}
                truths = [response[row["response_index"], target_q, target_role].astype(np.float32)
                          for row in split_rows]
                for i, row in enumerate(split_rows):
                    truth = truths[i]
                    x = states[base_indices[row["response_index"]], q, role_i].astype(np.float32)
                    xm, ym, beta = params
                    pred = ym + beta * (x - xm)
                    predictions = {"bridge": pred, "mean": y_mean,
                                   "wrong_direction": -pred,
                                   "row_roll": truths[(i + 1) % len(truths)] if truths else y_mean}
                    for name, value in predictions.items():
                        err, den = _nrmse_sums(value, truth)
                        sums[name][0] += err; sums[name][1] += den
                split_metrics[split_name] = {name: _finish_nrmse(tuple(values))
                                             for name, values in sums.items()}
            tournament.append({"checkpoint": q, "role": role, **split_metrics})
    save(out / "analysis/bridge_tournament.json", tournament)
    winner = min(tournament, key=lambda row: row["confirmation"]["bridge"])
    selected_q, selected_role = winner["checkpoint"], winner["role"]
    xm, ym, beta = fitted[(selected_q, selected_role)]
    np.savez(out / "raw/selected_bridge_model.npz", x_mean=xm, y_mean=ym, beta=beta,
             checkpoint=np.asarray([selected_q]), role=np.asarray([ROLES.index(selected_role)]))
    confirmation_gain = winner["confirmation"]["mean"] - winner["confirmation"]["bridge"]
    lockbox_gain = winner["lockbox"]["mean"] - winner["lockbox"]["bridge"]

    # Discovery-only response prototypes at every role support a preregistered
    # causal ladder.  The exact lockbox q32 vector is a positive control only.
    discovery_role_means = {}
    for q in (0, 8, 16, 24, 32):
        for role_i, role in enumerate(ROLES):
            discovery_role_means[(q, role)] = np.mean([
                response[row["response_index"], q, role_i].astype(np.float32)
                for row in train], axis=0) if train else np.zeros(DIM, np.float32)
    model = None; causal = []
    try:
        model, tokenizer, device, _placement = MODEL_BASE.load_bf16("qwen3")
        selected_role_i = ROLES.index(selected_role)
        wrong_q = {0: 8, 8: 16, 16: 8, 24: 16}[selected_q]
        for test_i, row in enumerate(lockbox):
            concept, surface = row["concept_index"], row["surface"]
            base_row = lookup[(concept, "zh", "en", surface, "natural")]
            target_row = lookup[(concept, "zh", "fr", surface, "natural")]
            item = dict(compiled[base_row["case_id"]])
            item["answer_candidates"] = compiled[target_row["case_id"]]["answer_candidates"]
            item["gold_position"] = compiled[target_row["case_id"]]["gold_position"]
            exact = response[row["response_index"], 32, target_role].astype(np.float32)
            def patches_for(q: int, role_vectors: list[tuple[str, np.ndarray]]) -> list[dict]:
                return [{"q": q, "position": item["role_positions"][role][-1], "vector": vector}
                        for role, vector in role_vectors]
            modes = {
                "zero": [],
                "selected_role": patches_for(selected_q, [(selected_role, discovery_role_means[(selected_q, selected_role)])]),
                "all_roles": patches_for(selected_q, [(role_name, discovery_role_means[(selected_q, role_name)])
                                                        for role_name in ROLES]),
                "wrong_direction": patches_for(selected_q, [(role_name, -discovery_role_means[(selected_q, role_name)])
                                                              for role_name in ROLES]),
                "wrong_checkpoint": patches_for(wrong_q, [(role_name, discovery_role_means[(selected_q, role_name)])
                                                            for role_name in ROLES]),
                "q32_discovery_mean": patches_for(32, [("boundary", discovery_role_means[(32, "boundary")])]),
                "q32_exact": patches_for(32, [("boundary", exact)]),
            }
            for mode, patches in modes.items():
                result = _patched_generate(model, tokenizer, item, patches)
                causal.append({"concept_index": concept, "surface": surface,
                               "partition": row["partition"], "formal": row["formal"],
                               "mode": mode, **result})
            if test_i % 8 == 0 or test_i + 1 == len(lockbox):
                print(f"[C643 causal] {test_i + 1}/{len(lockbox)}", flush=True)
    finally:
        MODEL_BASE.release_bf16(model); gc.collect()
    write_rows(out / "raw/causal_generation.jsonl", causal)
    causal_rates = {mode: float(np.mean([row["correct"] for row in causal if row["mode"] == mode]))
                    for mode in sorted({row["mode"] for row in causal})}
    bridge_pass = confirmation_gain >= CONTROL_MARGIN and lockbox_gain >= CONTROL_MARGIN
    causal_specific = (
        causal_rates.get("all_roles", 0.0) > max(causal_rates.get("zero", 0.0),
                                                 causal_rates.get("wrong_direction", 0.0),
                                                 causal_rates.get("wrong_checkpoint", 0.0)) and
        causal_rates.get("q32_exact", 0.0) > causal_rates.get("zero", 0.0))
    headline = {
        "status": "upstream_q32_translation_bridge_closed",
        "formal_counts": {"discovery": len(train), "confirmation": len(confirmation),
                          "lockbox": len(lockbox)},
        "selected_checkpoint": selected_q, "selected_role": selected_role,
        "winner_metrics": winner, "confirmation_gain_over_mean": confirmation_gain,
        "lockbox_gain_over_mean": lockbox_gain,
        "prospective_bridge_pass": bridge_pass,
        "causal_generation_rates": causal_rates,
        "causal_specificity_pass": causal_specific,
        "strict_interpretation": (
            "An observational diagonal bridge and a causal state addition are distinct claims. "
            "Only q32 exact is a positive control; upstream success would still identify a callable state route, not a unique circuit."),
    }
    close_mmap(response); close_mmap(states)
    close("C643", headline, {
        "tournament_complete": len(tournament) == 24,
        "lockbox_causal_complete": len(causal) == len(lockbox) * 7,
        "model_saved": (out / "raw/selected_bridge_model.npz").exists(),
        "finite": finite(headline),
    }, "C644_translation_triangle_crossmodel_topology")


def c644() -> None:
    out = begin("C644", {
        "object": "translation path composition and cross-model functional topology",
        "qwen_topology": "RMS target-language response by relative depth and semantic role",
        "cross_models": ["GLM4", "DeepSeek-7B", "Qwen3-14B"],
        "execution": "strictly sequential model load/release",
        "comparison": "relative-depth role topology and triangle NRMSE only; never physical coordinate IDs",
        "behavior_gate": "all four cross-model routes independently require candidate and exact generation >=0.80",
    }, {"C643": final("C643")["all_checks_passed"]})
    response = np.load(OUTS["C642"] / "raw/target_language_response.float16.npy", mmap_mode="r")
    rindex = read_rows(OUTS["C642"] / "raw/target_language_response_index.jsonl")
    selected_rows = [row for row in rindex if row["partition"] == "lockbox" and row["formal"]]
    qpoints = (0, 8, 16, 24, 32, 37)
    qwen_topology = []
    for q in qpoints:
        role_rms = []
        for role_i in range(len(ROLES)):
            numerator = 0.0; count = 0
            for row in selected_rows:
                value = response[row["response_index"], q, role_i].astype(np.float32)
                numerator += float(np.square(value).sum()); count += value.size
            role_rms.append(math.sqrt(numerator / max(count, 1)))
        scale = math.sqrt(sum(value * value for value in role_rms)) or 1.0
        qwen_topology.append({"relative_depth": q / 37.0, "checkpoint": q,
                              "role_rms_normalized": [value / scale for value in role_rms]})
    save(out / "analysis/qwen3_4b_translation_response_topology.json", qwen_topology)
    close_mmap(response)

    worker = TESTS / "phase2179_c644_translation_model_worker.py"
    worker_results = {}
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        model_dir = out / "raw" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / "summary.json"
        command = [sys.executable, str(worker), "--model", model_name,
                   "--material", str(material_path()), "--output", str(target)]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if target.exists():
            worker_results[model_name] = load(target)
        else:
            worker_results[model_name] = {"status": "worker_missing_output",
                                          "returncode": completed.returncode,
                                          "hiddenstate_ran": False}
        torch.cuda.empty_cache(); gc.collect()
    save(out / "analysis/cross_model_results.json", worker_results)
    topology_comparison = {}
    qwen_values = np.asarray([row["role_rms_normalized"] for row in qwen_topology], dtype=np.float64)
    for model_name, result in worker_results.items():
        if result.get("hiddenstate_ran") and result.get("topology"):
            topology = load(ROOT / result["topology"])
            values = np.asarray([row["role_rms_normalized"] for row in topology], dtype=np.float64)
            topology_comparison[model_name] = {
                "mean_absolute_role_topology_difference": float(np.mean(np.abs(values - qwen_values))),
                "translation_triangle_nrmse": result.get("translation_triangle_nrmse"),
                "coordinates": result.get("coordinates"),
                "status": result.get("status"),
            }
        else:
            topology_comparison[model_name] = {"status": result.get("status"),
                                               "mean_absolute_role_topology_difference": None}
    save(out / "analysis/topology_comparison.json", topology_comparison)
    qwen_triangle = final("C642")["headline"].get("triangle_nrmse_by_partition", {})
    headline = {
        "status": "translation_triangle_crossmodel_topology_closed",
        "qwen_topology": qwen_topology,
        "qwen_triangle_nrmse": qwen_triangle,
        "cross_model": topology_comparison,
        "models_attempted_sequentially": list(worker_results),
        "models_behavior_qualified": [name for name, value in worker_results.items()
                                      if value.get("status") == "closed"],
        "strict_interpretation": (
            "A similar normalized role topology is only a candidate functional invariant. "
            "Tokenizer, width and checkpoint differences forbid coordinate-level cross-model identity claims."),
    }
    close("C644", headline, {
        "qwen_topology_complete": len(qwen_topology) == 6,
        "all_models_attempted": len(worker_results) == 3,
        "all_worker_outputs_accounted": all(value.get("status") for value in worker_results.values()),
        "finite": finite(headline),
    }, "C645_translation_visual_theory_audit")


def _catalog_visual() -> None:
    catalog = load(CATALOG)
    entry = {"id": "c645_translation_relative_encoding_atlas",
             "label": "C645 Translation Relative Encoding Atlas",
             "path": "/vis_data/research_kernel/c645_translation_relative_encoding_atlas.json",
             "phase": 2180, "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    datasets = catalog.setdefault("field_datasets", [])
    datasets[:] = [row for row in datasets if row.get("id") != entry["id"]]
    datasets.append(entry)
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)


def c645() -> None:
    out = begin("C645", {
        "object": "exact-coordinate translation atlas, storage cleanup and theory audit",
        "display": "one untouched lockbox concept: zh->en base, zh->fr target and signed response across 38x6x2560",
        "new_math_gate": ["broad_behavior", "prospective_prediction", "specific_causality",
                          "registered_composition", "cross_model_replication", "independent_human_review"],
        "cleanup": "delete non-displayed large HiddenState tensors only after visual export and result audit",
    }, {name: final(name)["all_checks_passed"] for name in ("C641", "C642", "C643", "C644")})
    rows = read_rows(index_path()); lookup = _case_map(rows)
    states = np.load(states_path(), mmap_mode="r")
    response = np.load(OUTS["C642"] / "raw/target_language_response.float16.npy", mmap_mode="r")
    rindex = read_rows(OUTS["C642"] / "raw/target_language_response_index.jsonl")
    selected = next(row for row in rindex if row["concept_index"] == 4 and
                    row["surface"] == "explicit_en")
    base_row = lookup[(4, "zh", "en", "explicit_en", "natural")]
    target_row = lookup[(4, "zh", "fr", "explicit_en", "natural")]
    base_field = states[base_row["hidden_index"]].astype(np.float32)
    target_field = states[target_row["hidden_index"]].astype(np.float32)
    response_field = response[selected["response_index"]].astype(np.float32)
    c643_result = final("C643")["headline"]
    c644_result = final("C644")["headline"]
    panel_ledger = load(OUTS["C641"] / "raw/full_token_panel_ledger.json")
    selected_panel = next((row for row in panel_ledger if row["case_id"] == base_row["case_id"]), None)
    displayed_panel_url = None
    if selected_panel:
        source = ROOT / selected_panel["path"]
        destination = VISUAL.parent / "c645_translation_selected_full_token.float16.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        displayed_panel_url = "/vis_data/research_kernel/c645_translation_selected_full_token.float16.npy"
    visual = {
        "schema": "ai2050.translation_relative_encoding_atlas.v1",
        "phase": 2180, "campaign": "C641-C645",
        "heatmap_type": "embedding_hiddenstate_full_coordinate",
        "title": "Translation Relative Encoding: Embedding-to-HiddenState Exact Coordinate Atlas",
        "roles": list(ROLES), "checkpoints": list(range(CHECKPOINTS)),
        "coordinate_ids": list(range(DIM)), "shape": [CHECKPOINTS, len(ROLES), DIM],
        "selected_case": {"concept_index": 4, "concept": {lang: concept_word(4, lang) for lang in LANGUAGES},
                          "surface": "explicit_en", "base_case_id": base_row["case_id"],
                          "target_case_id": target_row["case_id"]},
        "fields": {
            "zh_to_en_base": np.round(base_field, 6).tolist(),
            "zh_to_fr_target": np.round(target_field, 6).tolist(),
            "target_language_signed_response": np.round(response_field, 6).tolist(),
        },
        "embedding": {"checkpoint": 0,
                      "base": np.round(base_field[0], 6).tolist(),
                      "target": np.round(target_field[0], 6).tolist(),
                      "response": np.round(response_field[0], 6).tolist()},
        "selected_full_token_panel": {"url": displayed_panel_url,
                                      "metadata": selected_panel},
        "bridge": c643_result,
        "cross_model_topology": c644_result,
        "reading_rule": "Every value is a signed physical activation coordinate. No Top-K, PCA or magnitude threshold was applied.",
    }
    save(VISUAL, visual); _catalog_visual()
    close_mmap(response); close_mmap(states)

    cleanup_targets = [states_path(),
                       OUTS["C642"] / "raw/target_language_response.float16.npy",
                       OUTS["C642"] / "raw/concept_target_mobius.float16.npy",
                       OUTS["C642"] / "raw/translation_triangle_residual.float16.npy"]
    cleanup_targets.extend(Path(ROOT / value["path"]) for value in panel_ledger)
    cross = load(OUTS["C644"] / "analysis/cross_model_results.json")
    for value in cross.values():
        if value.get("field"):
            cleanup_targets.append(ROOT / value["field"])
    cleanup = []
    for path in cleanup_targets:
        if path.exists():
            size = path.stat().st_size; path.unlink()
            cleanup.append({"path": str(path.relative_to(ROOT)), "bytes_deleted": size})
    save(out / "audit/storage_cleanup.json", {
        "files_deleted": len(cleanup), "bytes_deleted": sum(row["bytes_deleted"] for row in cleanup),
        "files": cleanup, "displayed_visual_retained": str(VISUAL.relative_to(ROOT)),
        "displayed_full_token_retained": displayed_panel_url})

    c641_result = final("C641")["headline"]
    natural_slices = [value for key, value in c641_result["slice_results"].items()
                      if key.startswith("natural|")]
    gates = {
        "broad_behavior": bool(natural_slices) and all(row["qualified"] for row in natural_slices),
        "prospective_prediction": bool(c643_result["prospective_bridge_pass"]),
        "specific_causality": bool(c643_result["causal_specificity_pass"]),
        "registered_composition": False,
        "cross_model_replication": len(c644_result["models_behavior_qualified"]) == 3,
        "independent_human_review": False,
    }
    headline = {
        "status": "translation_relative_encoding_campaign_closed",
        "audit_corrections": [
            "C638 supports discrete output identity equivalence only; continuous logit signatures differ.",
            "C639 brackets 160-coordinate insufficiency and 2400-coordinate sufficiency but proves no minimum.",
            "C631 finite response matrices are not a full Jacobian and do not prove mathematical nonlinearity.",
            "C636 used a preregistered six-sample partition compute cap.",
            "No independent human review was available; naturalness remains NA, never imputed."],
        "evidence_gates": gates,
        "new_foundational_math_gate": all(gates.values()),
        "visual": str(VISUAL.relative_to(ROOT)),
        "visual_bytes": VISUAL.stat().st_size,
        "cleanup": load(out / "audit/storage_cleanup.json"),
        "theory_update": (
            "The stable object, if supported, is a context-indexed translation response field linking concept, "
            "source language, target language, surface, role and checkpoint. This refines the existing conditionalized "
            "output-field theory; it does not license a fixed translation vector or a new foundational mathematics claim."),
        "strict_interpretation": (
            "Registered composition and independent human review were not passed in this campaign. "
            "Therefore the mathematical-upgrade gate remains closed even if narrower behavior, prediction or causal gates pass."),
    }
    close("C645", headline, {
        "visual_written": VISUAL.exists(),
        "exact_coordinate_shape": list(base_field.shape) == [38, 6, 2560],
        "catalog_registered": any(row.get("id") == "c645_translation_relative_encoding_atlas"
                                  for row in load(CATALOG).get("field_datasets", [])),
        "large_fields_cleaned": all(not path.exists() for path in cleanup_targets),
        "math_gate_conservative": headline["new_foundational_math_gate"] == all(gates.values()),
        "finite": finite(headline),
    }, "campaign_complete_next_route_requires_new_preregistration")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=("all", *PHASES), default="all")
    args = parser.parse_args()
    functions = {"C641": c641, "C642": c642, "C643": c643, "C644": c644, "C645": c645}
    if args.campaign == "all":
        for name in PHASES:
            functions[name]()
    else:
        functions[args.campaign]()


if __name__ == "__main__":
    main()
