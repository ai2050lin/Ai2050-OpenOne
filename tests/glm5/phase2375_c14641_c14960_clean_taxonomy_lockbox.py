#!/usr/bin/env python3
"""Leak-free natural taxonomy-chain lockbox with distinct first answer tokens."""
from __future__ import annotations

import gc
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
OUT = RESULT / "phase2375_c14641_c14960_clean_taxonomy_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/clean_taxonomy_factorial.jsonl"
STATES = OUT / "raw/qwen4b_clean_taxonomy_boundary.float16.npy"
DECISIONS = OUT / "raw/qwen4b_clean_taxonomy_decisions.float32.npy"
PROGRESS = OUT / "raw/progress.json"
PHASE = 2375
CAMPAIGN = "C14641-C14960"
BASE_EN = (("sparrow", "bird", "animal", "entity"), ("salmon", "fish", "animal", "entity"),
           ("violin", "instrument", "artifact", "object"), ("cedar", "tree", "plant", "organism"),
           ("ruby", "gem", "mineral", "material"), ("canoe", "boat", "vehicle", "artifact"),
           ("sonnet", "poem", "text", "work"), ("copper", "metal", "material", "substance"),
           ("falcon", "bird", "animal", "organism"), ("tulip", "flower", "plant", "organism"),
           ("cello", "instrument", "artifact", "object"), ("basalt", "rock", "mineral", "material"),
           ("schooner", "boat", "vehicle", "artifact"), ("haiku", "poem", "text", "work"),
           ("oak", "tree", "plant", "organism"), ("quartz", "crystal", "mineral", "material"))
BASE_ZH = (("麻雀", "鸟", "动物", "实体"), ("鲑鱼", "鱼", "动物", "实体"), ("小提琴", "乐器", "制品", "物体"),
           ("雪松", "树", "植物", "生物"), ("红宝石", "宝石", "矿物", "材料"), ("独木舟", "船", "交通工具", "制品"),
           ("十四行诗", "诗", "文本", "作品"), ("铜", "金属", "材料", "物质"), ("猎鹰", "鸟", "动物", "生物"),
           ("郁金香", "花", "植物", "生物"), ("大提琴", "乐器", "制品", "物体"), ("玄武岩", "岩石", "矿物", "材料"),
           ("纵帆船", "船", "交通工具", "制品"), ("俳句", "诗", "文本", "作品"), ("橡树", "树", "植物", "生物"),
           ("石英", "晶体", "矿物", "材料"))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2369_c12721_c13040_qwen_longrange_full_field as collect  # noqa: E402
import phase2372_c13681_c14000_dual_flagship_atlas as atlas  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    for unit in range(16):
        for language in ("en", "zh"):
            for surface in ("facts", "independent_prose"):
                for cell in range(32):
                    bits = [(cell >> i) & 1 for i in range(5)]; base = BASE_EN if language == "en" else BASE_ZH
                    raw = base[unit]
                    qualifier = (("river", "河谷"), ("mountain", "山地"))[bits[0]][0 if language == "en" else 1]
                    nodes = [f"{qualifier} {raw[0]}" if language == "en" else f"{qualifier}{raw[0]}",
                             f"{qualifier} {raw[1]}" if language == "en" else f"{qualifier}{raw[1]}", raw[2], raw[3]]
                    rel = (("is a", "是一种"), ("belongs to", "属于"))[bits[1]][0 if language == "en" else 1]
                    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
                    if bits[2]: edges.pop(1)
                    if bits[3]: edges.append((nodes[0], f"recorded specimen {unit + 1}" if language == "en" else f"登记标本{unit + 1}"))
                    depth = 3 if bits[4] else 1; unknown = "unknown" if language == "en" else "未知"
                    target_value = nodes[depth] if not (bits[2] and depth == 3) else unknown
                    foil_value = unknown if target_value != unknown else nodes[3]
                    record = f"Natural catalog record {unit + 1}. " if language == "en" else f"自然目录记录{unit + 1}。"
                    if language == "en":
                        facts = "; ".join(f"{a} {rel} {b}" for a, b in edges) + "."
                        prompt = record + (facts if surface == "facts" else f"An independent curator wrote: {facts}")
                        prompt += f"\nUsing only record {unit + 1}, give the depth-{depth} category of {nodes[0]}. Answer:"
                    else:
                        facts = "；".join(f"{a}{rel}{b}" for a, b in edges) + "。"
                        prompt = record + (facts if surface == "facts" else f"一名独立馆员写道：{facts}")
                        prompt += f"\n只根据记录{unit + 1}，给出{nodes[0]}的第{depth}层类别。答案："
                    # No leading standalone whitespace: target and foil must differ at token 0 on every model-forward row.
                    target, foil = target_value, foil_value
                    ti, fi = tokenizer.encode(target, add_special_tokens=False), tokenizer.encode(foil, add_special_tokens=False)
                    rows.append({"case_id": f"c14641-clean-chain-u{unit}-{language}-{surface}-c{cell:02d}", "design_index": len(rows),
                                 "system": "taxonomy_chain", "unit": unit, "partition": "train" if unit < 8 else "confirmation" if unit < 12 else "lockbox",
                                 "language": language, "surface": surface, "cell": cell, "bits": bits, "query": f"depth_{depth}",
                                 "prompt": prompt, "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": target, "foil": foil,
                                 "target_ids": ti, "foil_ids": fi, "target_first_id": ti[0], "foil_first_id": fi[0]})
    counts = Counter(r["prompt"] for r in rows)
    audit = {"rows": len(rows), "expected": 2048, "unique_prompts": len(counts), "duplicate_prompts": sum(v - 1 for v in counts.values()),
             "unique_case_ids": len(set(r["case_id"] for r in rows)) == len(rows),
             "first_token_distinct": all(r["target_first_id"] != r["foil_first_id"] for r in rows),
             "splits": dict(Counter(r["partition"] for r in rows))}
    return rows, audit


def splits(keys: list[tuple]) -> dict[str, np.ndarray]:
    return {"train": np.asarray([i for i, k in enumerate(keys) if k[1] < 8]),
            "confirmation": np.asarray([i for i, k in enumerate(keys) if 8 <= k[1] < 12]),
            "lockbox": np.asarray([i for i, k in enumerate(keys) if k[1] >= 12])}


def analyze(rows: list[dict]) -> dict:
    states = np.load(STATES, mmap_mode="r"); keys, index = atlas.build_group_index(rows); split = splits(keys)
    candidates, orders, layers = ("global", "language", "surface", "language_surface"), (1, 2, 3, 5), []
    for q in range(38):
        field = np.asarray(states[index, q], dtype=np.float32); entries = []
        for candidate in candidates:
            for order in orders:
                # other_train is unused unless candidate=other_system_global.
                rc, _ = atlas.evaluate(field, keys, split, split["train"], candidate, order, "confirmation")
                rl, _ = atlas.evaluate(field, keys, split, split["train"], candidate, order, "lockbox")
                entries.append({"candidate": candidate, "order": order, "confirmation_response_r2": rc, "lockbox_response_r2": rl})
        layers.append({"qpoint": q, "entries": entries}); print(f"[phase2375 analysis] qpoint {q}/37", flush=True)
    choices = [(e["confirmation_response_r2"], layer["qpoint"], e) for layer in layers for e in layer["entries"] if np.isfinite(e["confirmation_response_r2"])]
    _, q, selected = max(choices, key=lambda x: x[0]); save(OUT / "analysis/layers.json", layers)
    return {"splits": {k: len(v) for k, v in split.items()}, "selected_qpoint": q, "selected_candidate": selected["candidate"],
            "selected_order": selected["order"], "confirmation_response_r2": selected["confirmation_response_r2"],
            "lockbox_response_r2": selected["lockbox_response_r2"], "passed_clean_lockbox": selected["lockbox_response_r2"] > 0}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 无重复prompt知识链与首答案token严格纠错（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** Phase2373知识链的实体轮换造成240个跨unit重复prompt，且中文前导空格会把target/foil合并为同一首token。本Phase重新冻结16个自然概念域、两个自然限定实体、中英、事实/独立叙述、32-cell全立方，共{result['material']['rows']}条；每条显式含独立记录号，禁止任何重复prompt，答案不加独立前导空格并逐条验证首token不同。unit0–7训练、8–11确认、12–15锁箱。

$$\widehat R_{{\le k}}(x)=\sum_{{1\le|S|\le k}}[\chi_S(x)-\chi_S(0)]\overline H_S.$$

**结果汇总。** 材料审计 `{json.dumps(result['material'], ensure_ascii=False)}`；行为资格 `{json.dumps(result['behavior'], ensure_ascii=False)}`；锁箱竞赛 `{json.dumps(result['analysis'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2375_c14641_c14960_clean_taxonomy_lockbox.py`；材料、Qwen4B全场及逐层结果位于 `tests/glm5/result/phase2375_c14641_c14960_clean_taxonomy_lockbox`。

**理论进展、问题硬伤与结论。** 本Phase才是知识链跨词汇复用的有效锁箱。即便正R²也只是同一人工五因子坐标下的响应模板，不证明模型内部存储“苹果—水果—食物—物体”图或Walsh齿轮；关系删除导致`unknown`、记录号和限定词仍可能形成表面捷径。下一Phase用五句$S_5$检验Qwen4B逐坐标相邻交换是否跨句数复用。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        rows, material = compile_material(tokenizer); write_rows(MATERIAL, rows)
        collection = collect.collect_boundary(model, rows, STATES, DECISIONS, PROGRESS, "phase2375 clean-chain", 16)
    finally:
        model_utils.release_model(model); del model; gc.collect(); torch.cuda.empty_cache()
    behavior = collect.summarize_behavior(rows, DECISIONS, ["language", "surface", "partition"]); analysis = analyze(rows)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "material": material, "collection": collection,
              "behavior": behavior, "analysis": analysis, "supersedes": "Phase2373 taxonomy-chain fresh lockbox only; attitude result unaffected"}
    save(final, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
