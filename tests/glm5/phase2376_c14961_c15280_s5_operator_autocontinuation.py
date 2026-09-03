#!/usr/bin/env python3
"""Automatic continuation: five-sentence S5 full-coordinate operator lockbox."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase2376_c14961_c15280_s5_operator_autocontinuation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/s5_long_sentence_index.jsonl"
STATES = OUT / "raw/qwen4b_s5_boundary_all_layers.float16.npy"
DECISIONS = OUT / "raw/qwen4b_s5_decisions.float32.npy"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2376
CAMPAIGN = "C14961-C15280"
FAMILIES = ("temporal_narrative", "causal_process", "taxonomy_explanation", "spatial_route",
            "procedure", "comparison", "dialogue_coreference", "scientific_description")
TERMS = {"temporal_narrative": ("expedition", "chronology"), "causal_process": ("irrigation system", "causal chain"),
         "taxonomy_explanation": ("botanical archive", "category ladder"), "spatial_route": ("mountain route", "spatial path"),
         "procedure": ("laboratory protocol", "ordered procedure"), "comparison": ("design committee", "evidence ranking"),
         "dialogue_coreference": ("review meeting", "speaker reference"), "scientific_description": ("field observatory", "physical sequence")}
MARKERS = ("Alder", "Birch", "Cedar", "Dahlia", "Elm", "Fir", "Ginkgo", "Hazel", "Iris", "Juniper",
           "Kestrel", "Linden", "Maple", "Nectar", "Olive", "Pine", "Quartz", "Rowan", "Saffron", "Tulip",
           "Umber", "Violet", "Willow", "Xenia", "Yarrow", "Zephyr", "Amber", "Beryl", "Copper", "Delta")
PERMS = tuple(itertools.permutations(range(5)))
SOURCE_BY_UNIT = ((0, 1, 2, 3, 4), (2, 0, 4, 1, 3), (3, 1, 4, 0, 2))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2369_c12721_c13040_qwen_longrange_full_field as collect  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def render5(family: str, unit: int, language: str) -> tuple[list[str], list[str]]:
    topic, order = TERMS[family]; base = FAMILIES.index(family) * 3 + unit * 7; sentences, markers = [], []
    for sid in range(5):
        marker = MARKERS[(base + sid) % len(MARKERS)]; code = f"S5{FAMILIES.index(family)}{unit}{sid}"; markers.append(marker)
        if language == "en":
            sentences.append(f"{marker} preserves item {sid + 1}: the {topic} records unique code {code}, while three reviewers keep this complete sentence tied only to its place in the {order}.")
        else:
            sentences.append(f"{marker}保留第{sid + 1}项：{topic}记录唯一编号{code}；三名复核者保留完整句子，并确认它只对应{order}中的这个位置。")
    return sentences, markers


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    for family in FAMILIES:
        for unit in range(3):
            source = SOURCE_BY_UNIT[unit]
            for language in ("en", "zh"):
                sentences, markers = render5(family, unit, language)
                source_lines = "\n".join(f"Sentence {slot + 1}: {sentences[sid]}" for slot, sid in enumerate(source))
                for pi, target_perm in enumerate(PERMS):
                    marker_order = ", ".join(markers[i] for i in target_perm)
                    instruction = (f"Return only the comma-separated source sentence numbers whose leading markers appear in this requested order: {marker_order}. Answer:"
                                   if language == "en" else f"只返回首标记按以下顺序出现时对应的来源句编号，用逗号分隔：{marker_order}。答案：")
                    prompt = source_lines + "\n\n" + instruction; foil_perm = target_perm[1:] + target_perm[:1]
                    target = ",".join(str(source.index(sid) + 1) for sid in target_perm)
                    foil = ",".join(str(source.index(sid) + 1) for sid in foil_perm)
                    ti, fi = tokenizer.encode(target, add_special_tokens=False), tokenizer.encode(foil, add_special_tokens=False)
                    rows.append({"case_id": f"c14961-s5-{family}-u{unit}-{language}-p{pi:03d}", "design_index": len(rows),
                                 "family": family, "unit": unit, "partition": ("train", "confirmation", "lockbox")[unit],
                                 "language": language, "source_perm": list(source), "target_perm": list(target_perm), "permutation_index": pi,
                                 "markers": markers, "source_sentences": sentences, "prompt": prompt,
                                 "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": target, "foil": foil,
                                 "target_ids": ti, "foil_ids": fi, "target_first_id": ti[0], "foil_first_id": fi[0]})
    audit = {"rows": len(rows), "expected": 8 * 3 * 2 * 120, "families": len(FAMILIES), "permutations": len(PERMS),
             "unique_prompts": len(set(r["prompt"] for r in rows)), "first_token_distinct": all(r["target_first_id"] != r["foil_first_id"] for r in rows),
             "token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)],
             "splits": dict(Counter(r["partition"] for r in rows))}
    return rows, audit


def swap_perm(p: tuple[int, ...], i: int) -> tuple[int, ...]:
    p = list(p); p[i], p[i + 1] = p[i + 1], p[i]; return tuple(p)


def reduced_word(target: tuple[int, ...]) -> list[int]:
    current, word = list(range(5)), []
    for position, wanted in enumerate(target):
        j = current.index(wanted)
        while j > position: current[j - 1], current[j] = current[j], current[j - 1]; word.append(j - 1); j -= 1
    return word


def build_field(rows: list[dict], states: np.ndarray, q: int):
    groups = {}
    for i, r in enumerate(rows): groups.setdefault((r["family"], r["unit"], r["language"]), {})[r["permutation_index"]] = i
    keys = sorted(groups, key=str); index = np.asarray([[groups[k][pi] for pi in range(120)] for k in keys])
    return keys, index, np.asarray(states[index, q], dtype=np.float32)


def splits(keys: list[tuple]) -> dict[str, np.ndarray]:
    return {name: np.asarray([i for i, k in enumerate(keys) if k[1] == unit]) for unit, name in enumerate(("train", "confirmation", "lockbox"))}


def fit(field: np.ndarray, groups: np.ndarray):
    translations, slopes, intercepts = [], [], []
    for generator in range(4):
        target_pi = np.asarray([PERMS.index(swap_perm(p, generator)) for p in PERMS]); x = field[groups].reshape(-1, field.shape[-1]); y = field[groups][:, target_pi].reshape(-1, field.shape[-1])
        xm, ym = x.mean(0), y.mean(0); slope = ((x - xm) * (y - ym)).mean(0) / (np.square(x - xm).mean(0) + 1e-5)
        translations.append((y - x).mean(0)); slopes.append(slope); intercepts.append(ym - slope * xm)
    direct = (field[groups] - field[groups, 0][:, None]).mean(0)
    return np.stack(translations), np.stack(slopes), np.stack(intercepts), direct


def predict(field: np.ndarray, groups: np.ndarray, params, method: str):
    translations, slopes, intercepts, direct = params; actual = field[groups] - field[groups, 0][:, None]; predicted = np.zeros_like(actual)
    coord_perm = np.random.default_rng(2376).permutation(field.shape[-1])
    for pi, target in enumerate(PERMS):
        if pi == 0: continue
        if method == "direct_template": predicted[:, pi] = direct[pi]; continue
        value = field[groups, 0].copy()
        for generator in reduced_word(target):
            if method == "translation": value += translations[generator]
            elif method == "diagonal_affine": value = value * slopes[generator] + intercepts[generator]
            elif method == "coordinate_permuted_affine": value = value * slopes[generator, coord_perm] + intercepts[generator, coord_perm]
            elif method == "identity": pass
        predicted[:, pi] = value - field[groups, 0]
    return actual[:, 1:], predicted[:, 1:]


def response_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = actual.astype(np.float64), predicted.astype(np.float64); sse = np.square(a - p).sum(); sst = np.square(a - a.mean((0, 1), keepdims=True)).sum()
    return float("nan") if sst < 1e-8 else 1 - float(sse / sst)


def analyze(rows: list[dict]) -> dict:
    states = np.load(STATES, mmap_mode="r"); keys, index, _ = build_field(rows, states, 0); split = splits(keys); layers = []
    methods = ("identity", "translation", "diagonal_affine", "direct_template", "coordinate_permuted_affine")
    for q in range(38):
        _, _, field = build_field(rows, states, q); params = fit(field, split["train"]); entry = {"qpoint": q, "methods": {}}
        for method in methods:
            ac, pc = predict(field, split["confirmation"], params, method); al, pl = predict(field, split["lockbox"], params, method)
            entry["methods"][method] = {"confirmation_response_r2": response_r2(ac, pc), "lockbox_response_r2": response_r2(al, pl)}
        layers.append(entry); print(f"[phase2376 analysis] qpoint {q}/37", flush=True)
    choices = [(v["confirmation_response_r2"], layer["qpoint"], method, v) for layer in layers for method, v in layer["methods"].items()
               if method != "coordinate_permuted_affine" and np.isfinite(v["confirmation_response_r2"])]
    _, q, method, selected = max(choices); same = layers[q]["methods"]
    save(OUT / "analysis/layers.json", layers)
    return {"splits": {k: len(v) for k, v in split.items()}, "selected_qpoint": q, "selected_method": method,
            "confirmation_response_r2": selected["confirmation_response_r2"], "lockbox_response_r2": selected["lockbox_response_r2"],
            "same_layer_methods": same, "s5_operator_candidate_passed": method in ("translation", "diagonal_affine")
            and selected["lockbox_response_r2"] > 0 and selected["lockbox_response_r2"] > same["direct_template"]["lockbox_response_r2"]
            and selected["lockbox_response_r2"] > same["coordinate_permuted_affine"]["lockbox_response_r2"]}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 自动续研——五句$S_5$全排列与不同句长算子锁箱（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 因Qwen4B的$S_4$逐坐标仿射锁箱为正、但行为合格的Qwen14B/GLM4未复现，本Phase不停止也不升级结论，而是更换句数。覆盖8内容族、3个全新unit、中英、5个长句和$S_5$全部120排列，共{result['material']['rows']}条；unit0/$\sigma=e$训练，unit1/新$\sigma$确认，unit2/另一新$\sigma$锁箱。每个相邻交换$\tau_1\ldots\tau_4$学习全2560坐标逐坐标仿射，再组合预测119个非恒等排列。

$$h_{{\tau_i\pi,j}}=a_{{i,j}}h_{{\pi,j}}+b_{{i,j}}+\epsilon,qquad \pi\in S_5.$$

**结果汇总。** 材料 `{json.dumps(result['material'], ensure_ascii=False)}`；行为 `{json.dumps(result['behavior'], ensure_ascii=False)}`；算子锁箱 `{json.dumps(result['analysis'], ensure_ascii=False)}`；与$S_4$/跨模型比较 `{json.dumps(result['comparison'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2376_c14961_c15280_s5_operator_autocontinuation.py`；材料、Qwen4B全场和逐层结果位于 `tests/glm5/result/phase2376_c14961_c15280_s5_operator_autocontinuation`。

**理论进展、问题硬伤与结论。** 这是对“相邻交换响应律”的不同$n$外推，不是对模型内部显式群表示的证明。目标顺序直接出现在指令中，早层可利用词序；三个unit仍不足以覆盖自然长文本全部变化；120排列共享同一五句，不能当作120个独立语义样本。只有正锁箱、胜过直接模板和坐标置乱才保留为Qwen4B规律拼图，且跨模型普遍性仍由Phase2374否决。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows, material = compile_material(tokenizer); write_rows(MATERIAL, rows)
        collection = collect.collect_boundary(model, rows, STATES, DECISIONS, PROGRESS, "phase2376 s5", 8)
    finally:
        model_utils.release_model(model); del model; gc.collect(); torch.cuda.empty_cache()
    behavior = collect.summarize_behavior(rows, DECISIONS, ["language", "partition"]); analysis = analyze(rows)
    comparison = {"s4_qwen4_lockbox_r2": 0.21303969253213417, "s4_qwen14_lockbox_r2": -0.17647322895023265,
                  "s4_glm4_lockbox_r2": -0.06210191372237417, "s4_deepseek_lockbox_r2": 0.07606992553972192,
                  "universal_claim": False}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material": material, "collection": collection,
              "behavior": behavior, "analysis": analysis, "comparison": comparison}
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
