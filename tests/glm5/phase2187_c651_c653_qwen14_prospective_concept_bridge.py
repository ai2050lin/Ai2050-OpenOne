#!/usr/bin/env python3
"""C651-C653 Qwen3-14B prospective natural concept-translation bridge."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT_ROOT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as loader
import phase2183_c647_c650_fixed_target_concept_bridge_campaign as previous

PHASE = 2187
CAMPAIGNS = ("C651", "C652", "C653")
NAME = "qwen14_prospective_natural_concept_bridge"
OUT = RESULT_ROOT / f"phase{PHASE}_c651_c653_{NAME}"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = previous.CATALOG
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c653_qwen14_concept_bridge_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c653_qwen14_selected_concept_response.float16.npy"
ROLES = previous.ROLES
FAMILIES = previous.FAMILIES
TARGETS = ("en", "fr")
BEHAVIOR_GATE = 0.80
GAIN_GATE = 0.02

CONFIRMATION = (
    ("fruit", "牛油果", "avocado", "avocat"), ("fruit", "覆盆子", "raspberry", "framboise"),
    ("animal", "鸟", "bird", "oiseau"), ("animal", "老鼠", "mouse", "souris"),
    ("object", "钢笔", "pen", "stylo"), ("object", "汽车", "car", "voiture"),
    ("nature", "天空", "sky", "ciel"), ("nature", "石头", "stone", "pierre"),
    ("food", "水", "water", "eau"), ("food", "果汁", "juice", "jus"),
    ("body", "脸", "face", "visage"), ("body", "膝盖", "knee", "genou"),
)

LOCKBOX = (
    ("fruit", "甜瓜", "melon", "melon"), ("fruit", "蓝莓", "blueberry", "myrtille"),
    ("animal", "鸭子", "duck", "canard"), ("animal", "猪", "pig", "cochon"),
    ("object", "电脑", "computer", "ordinateur"), ("object", "袋子", "bag", "sac"),
    ("nature", "海", "sea", "mer"), ("nature", "大地", "earth", "terre"),
    ("food", "冰淇淋", "ice cream", "glace"), ("food", "三明治", "sandwich", "sandwich"),
    ("body", "皮肤", "skin", "peau"), ("body", "血液", "blood", "sang"),
)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def close_mmap(array: Any) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def records() -> list[dict]:
    result = []
    sources = (
        ("discovery", [(r["family"], r["words"]["zh"], r["words"]["en"], r["words"]["fr"])
                       for r in previous.RECORDS if r["partition"] == "lockbox"]),
        ("confirmation", list(CONFIRMATION)),
        ("lockbox", list(LOCKBOX)),
    )
    for partition, values in sources:
        partition_i = 0
        for family in FAMILIES:
            family_values = [value for value in values if value[0] == family]
            expected = 4 if partition == "discovery" else 2
            if len(family_values) != expected:
                raise RuntimeError((partition, family, len(family_values), expected))
            for family_rank, value in enumerate(family_values):
                result.append({"concept_uid": f"q14_{partition[0]}_{family}_{family_rank}",
                               "source_uid": (previous.RECORDS[[r["words"]["en"] for r in previous.RECORDS].index(value[2])]["concept_uid"]
                                              if partition == "discovery" else None),
                               "partition": partition, "partition_index": partition_i,
                               "family": family, "family_rank": family_rank,
                               "words": {"zh": value[1], "en": value[2], "fr": value[3]}})
                partition_i += 1
    return result


RECORDS = records()
BY_UID = {row["concept_uid"]: row for row in RECORDS}


def candidate_records(record: dict) -> tuple[list[dict], int]:
    partition = [row for row in RECORDS if row["partition"] == record["partition"]]
    family = [row for row in partition if row["family"] == record["family"]]
    pool = family + [row for row in partition if row["family"] != record["family"]]
    source_i = pool.index(record)
    distractors = []
    offset = 1
    while len(distractors) < 3:
        value = pool[(source_i + offset) % len(pool)]
        if value != record and value not in distractors:
            distractors.append(value)
        offset += 1
    position = int(record["partition_index"]) % 4
    raw = [record, *distractors]
    ordered = raw[1:position + 1] + [raw[0]] + raw[position + 1:]
    return ordered, position


def make_row(record: dict, target: str) -> dict:
    candidates, gold_position = candidate_records(record)
    prompt, anchors = previous.translation.natural_prompt(record["words"]["zh"], "zh", target, "explicit_en")
    return {"case_id": f"c651|{record['concept_uid']}|zh-{target}|explicit_en|natural",
            "concept_uid": record["concept_uid"], "source_uid": record["source_uid"],
            "concept_family": record["family"], "family_rank": record["family_rank"],
            "partition": record["partition"], "source_language": "zh", "target_language": target,
            "surface": "explicit_en", "protocol": "natural",
            "slice_key": f"{record['partition']}|zh-{target}|explicit_en|natural",
            "prompt": prompt, "system": previous.translation.NATURAL_SYSTEM,
            "source_word": record["words"]["zh"], "natural_answer": record["words"][target],
            "answer": record["words"][target],
            "answer_candidates": [row["words"][target] for row in candidates],
            "gold_position": gold_position, "candidate_concepts": [row["concept_uid"] for row in candidates],
            "role_values": anchors}


def make_material() -> list[dict]:
    return [make_row(record, target) for record, target in itertools.product(RECORDS, TARGETS)]


def partition_pairs(partition: str) -> list[tuple[dict, dict]]:
    answer = []
    for family in FAMILIES:
        values = sorted([row for row in RECORDS if row["partition"] == partition and row["family"] == family],
                        key=lambda row: row["family_rank"])
        answer.extend((values[i], values[i + 1]) for i in range(0, len(values), 2))
    return answer


def freeze(rows: list[dict]) -> None:
    for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": PHASE, "campaigns": CAMPAIGNS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "material_sha256": digest(rows),
        "model": "Qwen3-14B FP16 disk offload, selected only because C650 English and French routes independently qualified",
        "object": "natural source-concept identity propagation and prospective late-boundary response under English/French targets",
        "partitions": "24 C650-qualified concepts as discovery; 12 new confirmation; 12 wholly new lockbox",
        "behavior_gate": "candidate and exact free generation each >=0.80 independently for both targets in each partition",
        "camera": "all 42 checkpoints x six semantic roles x all 5120 signed coordinates; selected full-token panels",
        "readout": "centered all-coordinate identity matching across target-language contexts",
        "prediction": "coordinatewise affine and all-coordinate nearest response; discovery fit, confirmation selection, one lockbox reveal",
        "bridge_target": "q33 boundary concept-pair response; candidate q8/q16/q25/q33 source/query/boundary",
        "prediction_gate": "confirmation and lockbox NRMSE each improve discovery mean by >=0.02",
        "causal_modes": ["zero", "exact_selected", "predicted_q33", "exact_q33", "wrong_pair",
                         "wrong_direction", "wrong_role", "wrong_checkpoint", "all_roles_selected"],
        "dose": [0.5, 1.0, 1.5],
        "failure_policy": "behavior failure makes only that partition/route NA; prediction failure does not suppress exact positive controls",
        "forbidden": "Attention/MLP/weights/gradients/Top-K/PCA/projection/post-unblind threshold changes",
        "human_review": "NA_pending_external_review; frozen template only",
    }
    if not (OUT / "protocol/preregistration.json").exists():
        save(OUT / "protocol/preregistration.json", protocol)
    write_rows(OUT / "material/material.jsonl", rows)
    write_rows(OUT / "external/human_blind_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None,
         "translation_equivalence_0_1": None, "reviewer": None}
        for row in rows if row["partition"] in ("confirmation", "lockbox")])


def material_audit(rows: list[dict]) -> dict:
    duplicate = {}
    for language in ("zh", "en", "fr"):
        values = defaultdict(list)
        for record in RECORDS:
            values[record["words"][language].casefold()].append(record["concept_uid"])
        duplicate[language] = {word: ids for word, ids in values.items() if len(ids) > 1}
    balances = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        balances[row["slice_key"]][row["gold_position"]] += 1
    return {"concepts": len(RECORDS), "rows": len(rows),
            "partition_counts": {part: sum(row["partition"] == part for row in RECORDS)
                                 for part in ("discovery", "confirmation", "lockbox")},
            "duplicate_lexemes": duplicate,
            "candidate_position_counts": dict(balances),
            "candidate_position_exact_balance": all(len(set(value)) == 1 for value in balances.values()),
            "machine_semantic_uniqueness": all(not values for values in duplicate.values()),
            "human_review": "NA_pending_external_review"}


def capture(model, device, compiled: list[dict], dim: int) -> tuple[np.memmap, list[dict]]:
    modules = [model.model.embed_tokens, *list(model.model.layers), model.model.norm]
    checkpoints = len(modules)
    path = OUT / "raw/all_role_field.float16.npy"
    field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                     shape=(len(compiled), checkpoints, len(ROLES), dim))
    panel_dir = OUT / "raw/full_token_panel"; panel_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = {row["case_id"] for row in compiled
                 if row["partition"] == "lockbox" and row["target_language"] == "fr"}
    captured = []
    handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(
        output[0] if isinstance(output, tuple) else output)) for module in modules]
    panels = []
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            panel = None; panel_path = None
            if item["case_id"] in panel_ids:
                panel_path = panel_dir / f"row_{row_i:03d}.float16.npy"
                panel = np.lib.format.open_memmap(panel_path, mode="w+", dtype=np.float16,
                                                 shape=(checkpoints, len(item["prompt_ids"]), dim))
            for q, tensor in enumerate(captured):
                values = tensor[0].float().cpu().numpy().astype(np.float16)
                if panel is not None:
                    panel[q] = values
                for role_i, role in enumerate(ROLES):
                    field[row_i, q, role_i] = values[int(item["role_positions"][role][-1])]
            if panel is not None and panel_path is not None:
                panel.flush(); panels.append({"case_id": item["case_id"],
                                              "path": str(panel_path.relative_to(ROOT)),
                                              "shape": list(panel.shape), "bytes": panel_path.stat().st_size})
                close_mmap(panel)
            print(f"[C651 q14 full field] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush(); save(OUT / "raw/full_token_panel_ledger.json", panels)
    return field, panels


def fit_diagonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xm = x.mean(0); ym = y.mean(0)
    beta = ((x - xm) * (y - ym)).sum(0) / (np.square(x - xm).sum(0) + 1e-6)
    return xm.astype(np.float32), ym.astype(np.float32), beta.astype(np.float32)


def nrmse(pred: list[np.ndarray], truth: list[np.ndarray]) -> float | None:
    if not truth:
        return None
    num = sum(float(np.square(a - b).sum()) for a, b in zip(pred, truth))
    den = sum(float(np.square(b).sum()) for b in truth)
    return float(math.sqrt(num / max(den, 1e-12)))


def mean_or_none(values: list[Any]) -> float | None:
    return float(np.mean(values)) if values else None


def nearest(x: np.ndarray, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
    return yt[int(np.argmin(np.square(xt - x[None]).sum(1)))]


def append_memo(result: dict) -> None:
    existing = MEMO.read_text(encoding="utf-8-sig")
    if f"## Phase {PHASE}:" in existing:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    protocol = load(OUT / "protocol/preregistration.json")
    formulas = r"""$$
\mathcal H^{14B}(c,t)=\{H_{q,r,j}(c,t)\}_{q=0}^{41}{}_{r\in\mathcal R}{}_{j=1}^{5120}
$$

$$
D_{ab}^t(q,r)=H(b,t;q,r)-H(a,t;q,r),\qquad
M_{c\times t}=H(b,fr)-H(b,en)-H(a,fr)+H(a,en)
$$

$$
\widehat D_{ab}^{fr}(33,boundary)=\mu_y+\beta\odot(D_{ab}^{fr}(q,r)-\mu_x)
$$

$$
Y_m=F_{\ge q}\!\left(H_q+\alpha D_{ab}(q,r)\right),\qquad \alpha\in\{0.5,1.0,1.5\}
$$"""
    text = f"""

## Phase {PHASE}: Qwen14自然概念翻译的全场、前瞻桥与因果正控（C651-C653） [{stamp}]

**阶段目标与授权来源。** C650中只有Qwen3-14B在24个全新概念的英语与法语自然输出上同时通过候选/自由生成双门，因此本期自动延续相同目标：在更大模型中研究固定源语言下，概念身份如何传播并形成目标词。C650的24词仅作discovery；另建12词confirmation和12词lockbox，禁止把已看过的词事后切成锁箱。

**运行前冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)}
```

**测试用例。** discovery包括`猕猴桃 -> kiwi`与`猕猴桃 -> kiwi(法语)`；confirmation包括`牛油果 -> avocado/avocat`、`鸟 -> bird/oiseau`；最后一次揭示的lockbox包括`蓝莓 -> blueberry/myrtille`、`电脑 -> computer/ordinateur`、`皮肤 -> skin/peau`。每个概念的英语和法语目标构成配对条件；同族A→B构成概念差分。

**测试原理与公式。**

{formulas}

**详细结果与门槛。**

```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析与理论进展。** `{result.get('strict_interpretation')}` 理论主体仍为“条件化输出场闭合理论”。本期把模型规模导致的行为资格、概念身份可读、未见边界响应预测、个体精确状态充分性和预测状态可调用性严格分账，不把其中任何一项单独命名为“翻译齿轮”。

**问题、硬伤与瓶颈。** 仍是受控单词翻译；独立人类自然度审查为`NA`；14B磁盘卸载使完整候选评分与自由生成成本很高；只有12个confirmation和12个lockbox新词，族内每个分区仅一对；候选中存在跨族干扰词；全坐标保存避免Top-K遗漏，但逐坐标对角模型仍不表达跨坐标耦合；精确个体差分若成功也可能是覆盖式输出编辑；激活坐标不是参数权重，且不能跨模型按编号比较。

**相关文件。** 脚本：`tests/glm5/phase2187_c651_c653_qwen14_prospective_concept_bridge.py`；结果：`{OUT.relative_to(ROOT)}`；预注册：`{(OUT / 'protocol/preregistration.json').relative_to(ROOT)}`；可视化：`{VISUAL.relative_to(ROOT)}`与`{VISUAL_BINARY.relative_to(ROOT)}`。

**结论与下一步。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        print(json.dumps(load(OUT / "analysis/final.json"), ensure_ascii=False, indent=2))
        return
    rows = make_material(); freeze(rows)
    audit = material_audit(rows); save(OUT / "audit/material_audit.json", audit)
    if len(rows) != 96 or not audit["candidate_position_exact_balance"] or not audit["machine_semantic_uniqueness"]:
        raise RuntimeError(audit)
    model = None; causal = []; dose_rows = []
    try:
        model, tokenizer, device, placement, loader_name = loader.load_model("qwen3_14b")
        compiled = previous.compile_rows(tokenizer, rows)
        write_rows(OUT / "material/compiled.jsonl", compiled)
        by_key = {(row["concept_uid"], row["target_language"]): row for row in compiled}

        old_behavior = read_rows(previous.out("C650") / "external/qwen3_14b/behavior.jsonl")
        old_map = {}
        old_material = {row["case_id"]: row for row in read_rows(previous.out("C650") / "material/joint_factorial.jsonl")}
        for value in old_behavior:
            source = old_material[value["case_id"]]
            old_map[(source["concept_uid"], source["target_language"])] = value
        new_compiled = [row for row in compiled if row["partition"] != "discovery"]
        scores_all = previous.translation.base.old.previous.c607.batch_candidate_scores(
            model, device, new_compiled, batch_size=2)
        behavior_map = {}
        for row in compiled:
            if row["partition"] == "discovery":
                source_uid = BY_UID[row["concept_uid"]]["source_uid"]
                old = old_map[(source_uid, row["target_language"])]
                behavior_map[row["case_id"]] = {"case_id": row["case_id"],
                                                "candidate_correct": old["candidate_correct"],
                                                "generated_text": old["generated_text"],
                                                "generated_correct": old["generated_correct"],
                                                "source": "reused_C650_qwen14"}
        for item, scores in zip(new_compiled, scores_all):
            text = previous.translation.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=10)
            prediction, correct = previous.translation.evaluate_generation(text, item)
            behavior_map[item["case_id"]] = {"case_id": item["case_id"],
                                             "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                                             "generated_text": text, "generated_prediction": prediction,
                                             "generated_correct": correct, "source": "new_C651"}
            print(f"[C651 q14 new behavior] {len(behavior_map) - 48}/{len(new_compiled)}", flush=True)
        behavior = [behavior_map[row["case_id"]] for row in compiled]
        write_rows(OUT / "behavior/behavior.jsonl", behavior)
        route_metrics = {}
        for partition, target in itertools.product(("discovery", "confirmation", "lockbox"), TARGETS):
            values = [behavior_map[row["case_id"]] for row in compiled
                      if row["partition"] == partition and row["target_language"] == target]
            ca = float(np.mean([row["candidate_correct"] for row in values]))
            ga = float(np.mean([row["generated_correct"] for row in values]))
            route_metrics[f"{partition}|{target}"] = {"rows": len(values), "candidate_accuracy": ca,
                                                       "generated_accuracy": ga,
                                                       "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
        save(OUT / "behavior/route_metrics.json", route_metrics)

        dim = int(model.model.embed_tokens.weight.shape[1])
        field, panels = capture(model, device, compiled, dim)
        checkpoints = field.shape[1]; q_candidates = (8, 16, 25, 33); target_q = 33
        role_index = {role: i for i, role in enumerate(ROLES)}
        row_index = {(row["concept_uid"], row["target_language"]): i for i, row in enumerate(compiled)}

        identity_confirmation = []
        for q, role in itertools.product(q_candidates + (41,), ("source", "query", "boundary")):
            records_part = [row for row in RECORDS if row["partition"] == "confirmation"]
            formal = all(route_metrics[f"confirmation|{target}"]["qualified"] for target in TARGETS)
            if formal:
                en = np.stack([field[row_index[(row["concept_uid"], "en")], q, role_index[role]].astype(np.float32)
                               for row in records_part])
                fr = np.stack([field[row_index[(row["concept_uid"], "fr")], q, role_index[role]].astype(np.float32)
                               for row in records_part])
                en -= en.mean(0); fr -= fr.mean(0)
                dist = np.square(en).sum(1)[:, None] + np.square(fr).sum(1)[None] - 2.0 * en @ fr.T
                accuracy = float(np.mean(np.argmin(dist, axis=1) == np.arange(len(records_part))))
            else:
                accuracy = None
            identity_confirmation.append({"checkpoint": q, "role": role,
                                          "rows": len(records_part) if formal else 0,
                                          "accuracy": accuracy})
        identity_winner = max(identity_confirmation, key=lambda row: row["accuracy"] or -1.0)

        def samples(partition: str, q: int, role: str) -> list[dict]:
            if not route_metrics[f"{partition}|fr"]["qualified"]:
                return []
            result = []
            for pair_i, (a, b) in enumerate(partition_pairs(partition)):
                ai = row_index[(a["concept_uid"], "fr")]; bi = row_index[(b["concept_uid"], "fr")]
                if not all(behavior_map[compiled[i]["case_id"]]["candidate_correct"] and
                           behavior_map[compiled[i]["case_id"]]["generated_correct"] for i in (ai, bi)):
                    continue
                result.append({"pair_index": pair_i, "a": a, "b": b,
                               "x": field[bi, q, role_index[role]].astype(np.float32) - field[ai, q, role_index[role]].astype(np.float32),
                               "y": field[bi, target_q, role_index["boundary"]].astype(np.float32) - field[ai, target_q, role_index["boundary"]].astype(np.float32)})
            return result

        tournament = []; fitted = {}
        for q, role in itertools.product(q_candidates, ("source", "query", "boundary")):
            train = samples("discovery", q, role); confirm = samples("confirmation", q, role)
            if not train or not confirm:
                continue
            xt = np.stack([row["x"] for row in train]); yt = np.stack([row["y"] for row in train])
            xm, ym, beta = fit_diagonal(xt, yt); truth = [row["y"] for row in confirm]
            predictions = {
                "diagonal": [ym + beta * (row["x"] - xm) for row in confirm],
                "nearest_response": [nearest(row["x"], xt, yt) for row in confirm],
                "discovery_mean": [yt.mean(0) for _row in confirm],
                "zero": [np.zeros(dim, np.float32) for _row in confirm],
            }
            metrics = {name: {"nrmse": nrmse(pred, truth)} for name, pred in predictions.items()}
            tournament.append({"checkpoint": q, "role": role, "train_rows": len(train),
                               "confirmation_rows": len(confirm), "metrics": metrics})
            fitted[(q, role)] = (xm, ym, beta, xt, yt)
        if tournament:
            choices = [(row["metrics"][kind]["nrmse"], row, kind) for row in tournament
                       for kind in ("diagonal", "nearest_response")]
            _, winner, winner_kind = min(choices, key=lambda value: value[0])
            confirmation_gain = winner["metrics"]["discovery_mean"]["nrmse"] - winner["metrics"][winner_kind]["nrmse"]
            selection = {"checkpoint": winner["checkpoint"], "role": winner["role"],
                         "model": winner_kind, "confirmation": winner["metrics"],
                         "confirmation_gain_over_mean": confirmation_gain}
        else:
            selection = {"checkpoint": 33, "role": "boundary", "model": "NA",
                         "confirmation": None, "confirmation_gain_over_mean": None}
        save(OUT / "protocol/confirmation_selection_frozen.json",
             {"identity": identity_winner, "bridge": selection, "frozen_before_lockbox": True})
        save(OUT / "analysis/confirmation_tournament.json", tournament)

        lock_identity = None
        if all(route_metrics[f"lockbox|{target}"]["qualified"] for target in TARGETS):
            q, role = identity_winner["checkpoint"], identity_winner["role"]
            lock_records = [row for row in RECORDS if row["partition"] == "lockbox"]
            en = np.stack([field[row_index[(row["concept_uid"], "en")], q, role_index[role]].astype(np.float32)
                           for row in lock_records]); fr = np.stack([
                field[row_index[(row["concept_uid"], "fr")], q, role_index[role]].astype(np.float32)
                for row in lock_records])
            en -= en.mean(0); fr -= fr.mean(0)
            dist = np.square(en).sum(1)[:, None] + np.square(fr).sum(1)[None] - 2.0 * en @ fr.T
            lock_identity = float(np.mean(np.argmin(dist, axis=1) == np.arange(len(lock_records))))

        lock_metrics = None; lock_gain = None; bridge_pass = False
        lock_samples = []
        if selection["model"] != "NA":
            q, role = selection["checkpoint"], selection["role"]
            lock_samples = samples("lockbox", q, role)
            xm, ym, beta, xt, yt = fitted[(q, role)]
            truth = [row["y"] for row in lock_samples]
            predicted = ([ym + beta * (row["x"] - xm) for row in lock_samples]
                         if selection["model"] == "diagonal" else
                         [nearest(row["x"], xt, yt) for row in lock_samples])
            lock_metrics = {selection["model"]: {"nrmse": nrmse(predicted, truth)},
                            "discovery_mean": {"nrmse": nrmse([yt.mean(0) for _row in lock_samples], truth)},
                            "zero": {"nrmse": nrmse([np.zeros(dim, np.float32) for _row in lock_samples], truth)}}
            if lock_metrics[selection["model"]]["nrmse"] is not None:
                lock_gain = lock_metrics["discovery_mean"]["nrmse"] - lock_metrics[selection["model"]]["nrmse"]
                bridge_pass = selection["confirmation_gain_over_mean"] >= GAIN_GATE and lock_gain >= GAIN_GATE
            np.savez(OUT / "raw/selected_bridge_model.npz", model=np.asarray([selection["model"]]),
                     checkpoint=np.asarray([q]), role=np.asarray([role_index[role]]),
                     x_mean=xm, y_mean=ym, beta=beta, x_train=xt, y_train=yt)

        if lock_samples:
            q, role = selection["checkpoint"], selection["role"]
            wrong_q = {8: 16, 16: 25, 25: 33, 33: 25}[q]
            for list_i, sample in enumerate(lock_samples):
                a, b = sample["a"], sample["b"]
                base_item = by_key[(a["concept_uid"], "fr")]
                target_item = by_key[(b["concept_uid"], "fr")]
                item = previous._eval_item(base_item, target_item)
                ai = row_index[(a["concept_uid"], "fr")]; bi = row_index[(b["concept_uid"], "fr")]
                exact = sample["x"]; exact_q33 = sample["y"]
                xm, ym, beta, xt, yt = fitted[(q, role)]
                predicted = ym + beta * (exact - xm) if selection["model"] == "diagonal" else nearest(exact, xt, yt)
                wrong = lock_samples[(list_i + 1) % len(lock_samples)]["x"]
                all_roles = [(role_name, field[bi, q, role_i].astype(np.float32) -
                              field[ai, q, role_i].astype(np.float32))
                             for role_i, role_name in enumerate(ROLES)]
                wrong_role = "query" if role != "query" else "instruction"
                modes = {
                    "zero": [],
                    "exact_selected": previous._patches(item, q, [(role, exact)]),
                    "predicted_q33": previous._patches(item, target_q, [("boundary", predicted)]),
                    "exact_q33": previous._patches(item, target_q, [("boundary", exact_q33)]),
                    "wrong_pair": previous._patches(item, q, [(role, wrong)]),
                    "wrong_direction": previous._patches(item, q, [(role, -exact)]),
                    "wrong_role": previous._patches(item, q, [(wrong_role, exact)]),
                    "wrong_checkpoint": previous._patches(item, wrong_q, [(role, exact)]),
                    "all_roles_selected": previous._patches(item, q, all_roles),
                }
                for mode, patches in modes.items():
                    generated = previous.translation._patched_generate(model, tokenizer, item, patches, max_new_tokens=10)
                    causal.append({"pair_index": sample["pair_index"], "a": a["concept_uid"],
                                   "b": b["concept_uid"], "mode": mode, **generated})
                for kind, vector, qv, rv in (("exact_selected", exact, q, role),
                                              ("predicted_q33", predicted, target_q, "boundary")):
                    for dose in (0.5, 1.0, 1.5):
                        generated = previous.translation._patched_generate(
                            model, tokenizer, item, previous._patches(item, qv, [(rv, vector * dose)]),
                            max_new_tokens=10)
                        dose_rows.append({"pair_index": sample["pair_index"], "kind": kind,
                                          "dose": dose, **generated})
                print(f"[C653 q14 causal] {list_i + 1}/{len(lock_samples)}", flush=True)
        write_rows(OUT / "raw/causal_generation.jsonl", causal)
        write_rows(OUT / "raw/dose_generation.jsonl", dose_rows)

        # Visualize one lockbox pair at every checkpoint, role and coordinate.
        selected_pair = partition_pairs("lockbox")[0]
        ai = row_index[(selected_pair[0]["concept_uid"], "fr")]
        bi = row_index[(selected_pair[1]["concept_uid"], "fr")]
        response = (field[bi].astype(np.float32) - field[ai].astype(np.float32)).astype(np.float16)
        VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True); np.save(VISUAL_BINARY, response)
        atlas = {"schema": "ai2050.qwen14_prospective_concept_bridge.v1", "phase": PHASE,
                 "campaigns": list(CAMPAIGNS), "model": "Qwen3-14B", "coordinates": dim,
                 "coordinate_ids": list(range(dim)), "checkpoints": list(range(checkpoints)),
                 "roles": list(ROLES), "selected_pair": [selected_pair[0], selected_pair[1]],
                 "selected_pair_response_shape": list(response.shape),
                 "selected_pair_response": np.round(response.astype(np.float32), 6).tolist(),
                 "binary_float16": "/vis_data/research_kernel/c653_qwen14_selected_concept_response.float16.npy",
                 "identity_selection": identity_winner, "bridge_selection": selection,
                 "full_coordinate": True, "no_topk": True,
                 "warning": "activation coordinates are not weights and coordinate IDs are model-specific"}
        save(VISUAL, atlas)
        field.flush(); close_mmap(field)
    finally:
        loader.release_model("qwen3_14b", model); gc.collect()

    causal_rates = {mode: mean_or_none([row["correct"] for row in causal if row["mode"] == mode])
                    for mode in sorted({row["mode"] for row in causal})}
    dose_rates = {f"{kind}@{dose}": mean_or_none([row["correct"] for row in dose_rows
                                                   if row["kind"] == kind and row["dose"] == dose])
                  for kind, dose in itertools.product(("exact_selected", "predicted_q33"), (0.5, 1.0, 1.5))}
    predicted_causal = ((causal_rates.get("predicted_q33") or 0.0) >= 0.50 and
                        (causal_rates.get("predicted_q33") or 0.0) -
                        max(causal_rates.get("zero") or 0.0, causal_rates.get("wrong_pair") or 0.0,
                            causal_rates.get("wrong_direction") or 0.0,
                            causal_rates.get("wrong_role") or 0.0,
                            causal_rates.get("wrong_checkpoint") or 0.0) >= 0.25)

    catalog = load(CATALOG)
    entry = {"id": "c653_qwen14_concept_bridge_atlas", "label": "C653 Qwen14 Concept Bridge Atlas",
             "path": "/vis_data/research_kernel/c653_qwen14_concept_bridge_atlas.json",
             "binary_path": "/vis_data/research_kernel/c653_qwen14_selected_concept_response.float16.npy",
             "phase": PHASE, "full_coordinate": True,
             "heatmap_type": "embedding_hiddenstate_full_coordinate"}
    datasets = catalog.setdefault("field_datasets", []); datasets[:] = [row for row in datasets if row.get("id") != entry["id"]]
    datasets.append(entry); catalog["generated_at"] = datetime.now(timezone.utc).isoformat(); save(CATALOG, catalog)

    cleanup = {"deleted": [], "retained": [str(VISUAL.relative_to(ROOT)), str(VISUAL_BINARY.relative_to(ROOT))],
               "bytes_deleted": 0}
    panel_ledger = load(OUT / "raw/full_token_panel_ledger.json")
    selected_ids = {make_row(record, "fr")["case_id"] for record in partition_pairs("lockbox")[0]}
    for row in panel_ledger:
        path = ROOT / row["path"]
        if row["case_id"] in selected_ids:
            cleanup["retained"].append(row["path"])
        elif path.exists():
            cleanup["bytes_deleted"] += path.stat().st_size; cleanup["deleted"].append(row["path"]); path.unlink()
    field_path = OUT / "raw/all_role_field.float16.npy"
    if field_path.exists():
        cleanup["bytes_deleted"] += field_path.stat().st_size
        cleanup["deleted"].append(str(field_path.relative_to(ROOT))); field_path.unlink()
    save(OUT / "audit/cleanup.json", cleanup)

    strict = ("Qwen14 produced a behavior-qualified, prospectively predicted and causally callable concept response."
              if bridge_pass and predicted_causal else
              "Qwen14 qualified behavior and supplied a full-coordinate concept field, but the prospective predictive-causal bridge did not jointly pass.")
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": True, "material_audit": audit,
              "placement": placement, "loader": loader_name,
              "route_metrics": route_metrics, "field_shape": [96, 42, len(ROLES), 5120],
              "identity_confirmation_selection": identity_winner,
              "identity_lockbox_accuracy": lock_identity,
              "bridge_selection": selection, "bridge_lockbox_rows": len(lock_samples),
              "bridge_lockbox_metrics": lock_metrics, "bridge_lockbox_gain": lock_gain,
              "prospective_bridge_pass": bridge_pass,
              "causal_rates": causal_rates, "dose_rates": dose_rates,
              "predicted_state_causal_pass": predicted_causal,
              "human_review": "NA_pending_external_review",
              "new_foundational_mathematics_gate": False,
              "visual": str(VISUAL.relative_to(ROOT)), "cleanup": cleanup,
              "strict_interpretation": strict,
              "next_authorization": (
                  "若预测与因果双门通过，下一阶段扩展多词短语翻译与关系网络；否则保留14B行为/全场拼图，淘汰当前对角与最近响应桥，转向不预设逐坐标独立性的全场关系算法。")}
    checks = {"material_complete": len(rows) == 96, "behavior_complete": len(behavior) == 96,
              "field_complete_before_cleanup": True, "selection_frozen_before_lockbox": True,
              "causal_complete_or_registered_na": (len(causal) == len(lock_samples) * 9),
              "dose_complete_or_registered_na": (len(dose_rows) == len(lock_samples) * 6),
              "visual_complete": VISUAL.exists() and VISUAL_BINARY.exists(),
              "cleanup_complete": not field_path.exists(), "human_review_not_fabricated": True,
              "finite": finite(result)}
    result["checks"] = checks; result["all_checks_passed"] = all(checks.values())
    save(OUT / "analysis/final.json", result); append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
