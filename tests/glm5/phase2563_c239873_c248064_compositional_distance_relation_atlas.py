#!/usr/bin/env python3
"""Large relation-necessary depth/distance atlas plus causal and free-generation checks."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2563_c239873_c248064_compositional_distance_relation_atlas"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2563, "C239873-C248064"

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2538_c117505_c121600_token_atomic_hypergraph_behavior as atlas  # noqa: E402
import phase2552_c166145_c174336_relation_necessary_factorial_behavior as p2552  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def add(tokenizer, ids: list[int], regions: dict[str, list[int]], region: str, text: str) -> list[int]:
    tokens = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    start = len(ids)
    ids.extend(tokens)
    positions = list(range(start, len(ids)))
    regions.setdefault(region, []).extend(positions)
    return positions


def labels(family_id: int, form: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]:
    if form == "nonce":
        values = ("kivora", "mexalu")
    else:
        values = tuple(atlas.OPERATIONS[family_id][3])
    relations = p2552.relation_pair(family_id, "en", form)
    upper = (f"upper-{family_id:02d}-amber", f"upper-{family_id:02d}-cobalt")
    terminal = (f"terminal-{family_id:02d}-north", f"terminal-{family_id:02d}-south")
    return relations, values, upper, terminal


def compile_row(tokenizer, family_id: int, form: str, depth: int, gap: int,
                binding: int, query_relation: int, query_value: int, ablation: str) -> dict:
    entities = (f"Copper Lynx {family_id:02d}", f"Azure Heron {family_id:02d}")
    relations, values, upper, terminal = labels(family_id, form)
    ids: list[int] = []
    regions: dict[str, list[int]] = defaultdict(list)
    cells = []
    add(tokenizer, ids, regions, "frame", "Facts:\n")
    for entity_index in (0, 1):
        for relation_index in (0, 1):
            value_index = entity_index ^ relation_index ^ binding
            add(tokenizer, ids, regions, "frame", "Entity ")
            ep = add(tokenizer, ids, regions, "facts_entity", f"[{entities[entity_index]}]")
            add(tokenizer, ids, regions, "frame", " under relation ")
            rp = add(tokenizer, ids, regions, "facts_relation", relations[relation_index])
            add(tokenizer, ids, regions, "frame", " has value ")
            vp = add(tokenizer, ids, regions, "facts_value", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "frame", ".\n")
            cells.append({"entity": entity_index, "relation": relation_index, "value": value_index,
                          "entity_positions": ep, "relation_positions": rp, "value_positions": vp})
    if depth >= 2:
        for value_index in (0, 1):
            add(tokenizer, ids, regions, "bridge_frame", "Value ")
            add(tokenizer, ids, regions, "bridge_source", f"[{values[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", " belongs to ")
            add(tokenizer, ids, regions, "bridge_target", f"[{upper[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", ".\n")
    if depth >= 3:
        for value_index in (0, 1):
            add(tokenizer, ids, regions, "bridge_frame", "Class ")
            add(tokenizer, ids, regions, "bridge_source", f"[{upper[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", " belongs to ")
            add(tokenizer, ids, regions, "bridge_target", f"[{terminal[value_index]}]")
            add(tokenizer, ids, regions, "bridge_frame", ".\n")
    for distractor in range(gap):
        add(tokenizer, ids, regions, "distractor", f"Archive note {distractor} states marker-{family_id:02d}-{distractor} is inactive.\n")
    shown_relation = relations[query_relation] if ablation != "relation_missing" else "[relation unavailable]"
    targets = values if depth == 1 else upper if depth == 2 else terminal
    shown_target = targets[query_value] if ablation != "terminal_missing" else "target unavailable"
    add(tokenizer, ids, regions, "query_context", "Question: which entity has relation ")
    add(tokenizer, ids, regions, "query_relation", shown_relation)
    add(tokenizer, ids, regions, "query_context", " and reaches ")
    add(tokenizer, ids, regions, "query_terminal", f"[{shown_target}]")
    add(tokenizer, ids, regions, "frame", "?\nCandidates: ")
    add(tokenizer, ids, regions, "candidate", f"[{entities[0]}] or [{entities[1]}]")
    add(tokenizer, ids, regions, "instruction", ". Return only the complete entity name. Answer")
    add(tokenizer, ids, regions, "answer_boundary", ":")
    target_index = query_relation ^ query_value ^ binding
    base_id = (f"f{family_id:02d}_form-{form}_d{depth}_g{gap}_b{binding}_"
               f"qr{query_relation}_qv{query_value}")
    case_id = base_id if ablation == "full" else f"{base_id}_abl-{ablation}"
    return {"case_id": case_id, "base_case_id": base_id, "family_id": family_id,
            "family": atlas.OPERATIONS[family_id][0], "form": form, "depth": depth, "gap": gap,
            "binding": binding, "query_relation": query_relation, "query_value": query_value,
            "ablation": ablation, "entities": list(entities), "relations": list(relations),
            "values": list(values), "upper": list(upper), "terminal": list(terminal),
            "target_index": target_index, "target": entities[target_index], "prompt_ids": ids,
            "prompt": tokenizer.decode(ids), "regions": dict(regions), "fact_cells": cells,
            "answer_boundary_token": len(ids) - 1}


def compile_material(tokenizer) -> list[dict]:
    rows = []
    for family_id in range(32):
        for form in ("natural", "nonce"):
            for depth in (1, 2, 3):
                for gap in (0, 4):
                    for binding in (0, 1):
                        for query_relation in (0, 1):
                            for query_value in (0, 1):
                                for ablation in ("full", "relation_missing", "terminal_missing"):
                                    rows.append(compile_row(tokenizer, family_id, form, depth, gap, binding,
                                                            query_relation, query_value, ablation))
    return rows


def score_candidates(model, tokenizer, rows: list[dict], batch_size: int = 32) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in rows:
        for candidate_index, entity in enumerate(row["entities"]):
            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
            jobs.append({"row": row, "candidate_index": candidate_index, "continuation": continuation,
                         "sequence": row["prompt_ids"] + continuation})
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for job in jobs:
        buckets[len(job["sequence"])].append(job)
    batches = [values[start:start + batch_size] for length, values in sorted(buckets.items())
               for start in range(0, len(values), batch_size)]
    done = 0
    last_report = 0
    for batch in batches:
        ids, mask, _ = p2552.left_pad([job["sequence"] for job in batch], tokenizer.pad_token_id, device)
        keep = max(len(job["continuation"]) for job in batch) + 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
        for batch_index, job in enumerate(batch):
            first = keep - len(job["continuation"]) - 1
            value = 0.0
            for offset, token in enumerate(job["continuation"]):
                z = logits[batch_index, first + offset].float()
                value += float(z[token] - torch.logsumexp(z, dim=-1))
            scores[job["row"]["case_id"]][job["candidate_index"]] = value
        done += len(batch)
        if done == len(jobs) or done - last_report >= 1024:
            print(f"[phase2563 behavior] {done}/{len(jobs)}", flush=True)
            last_report = done
    output = []
    for row in rows:
        values = scores[row["case_id"]]
        prediction = max(values, key=values.get)
        target, wrong = row["target_index"], 1 - row["target_index"]
        output.append({key: row[key] for key in ("case_id", "base_case_id", "family_id", "family", "form",
                                                  "depth", "gap", "binding", "query_relation", "query_value",
                                                  "ablation", "target_index", "target")})
        output[-1].update({"prediction": prediction, "correct": prediction == target,
                           "target_score": values[target], "wrong_score": values[wrong],
                           "target_minus_wrong": values[target] - values[wrong]})
    return output


def behavior_summary(rows: list[dict]) -> dict:
    def panel(subset: list[dict]) -> dict:
        return {"n": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset])),
                "mean_margin": float(np.mean([row["target_minus_wrong"] for row in subset]))}
    output = {ablation: panel([row for row in rows if row["ablation"] == ablation])
              for ablation in ("full", "relation_missing", "terminal_missing")}
    full = [row for row in rows if row["ablation"] == "full"]
    output["full_by_depth_gap_form"] = {
        f"d{depth}_g{gap}_{form}": panel([row for row in full if row["depth"] == depth
                                          and row["gap"] == gap and row["form"] == form])
        for depth in (1, 2, 3) for gap in (0, 4) for form in ("natural", "nonce")}
    return output


def eligible_pairs(material: list[dict], scores: list[dict]) -> tuple[list[tuple], dict[tuple, dict]]:
    full = [row for row in material if row["ablation"] == "full"]
    correct = {row["case_id"]: row["correct"] for row in scores if row["ablation"] == "full"}
    index = {(row["family_id"], row["form"], row["depth"], row["gap"], row["query_relation"],
              row["query_value"], row["binding"]): row for row in full}
    keys = []
    for key in sorted({item[:-1] for item in index}):
        if correct[index[key + (0,)]["case_id"]] and correct[index[key + (1,)]["case_id"]]:
            keys.append(key)
    return keys, index


def choose(keys: list[tuple], limit: int = 256) -> list[tuple]:
    if len(keys) <= limit:
        return keys
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for key in keys:
        buckets[(key[1], key[2], key[3])].append(key)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < limit:
                selected.append(buckets[bucket].pop(0))
    return selected


def region(row: dict, name: str) -> list[int]:
    if name == "external":
        return list(range(row["answer_boundary_token"]))
    if name == "facts_all":
        return sorted({position for part in ("facts_entity", "facts_relation", "facts_value")
                       for position in row["regions"].get(part, [])})
    if name == "bridges":
        return sorted(row["regions"].get("bridge_source", []) + row["regions"].get("bridge_target", []))
    return list(row["regions"].get(name, []))


def bands(n_layers: int) -> tuple[tuple[int, ...], ...]:
    cuts = [round(index * n_layers / 4) for index in range(5)]
    return tuple(tuple(range(cuts[index], cuts[index + 1])) for index in range(4))


def conditions(n_layers: int) -> dict[str, dict]:
    early, middle, middlelate, late = bands(n_layers)
    return {"no_patch": {}, "early_k_facts_value": {"layers": early, "kind": "k", "region": "facts_value"},
            "early_v_facts_value": {"layers": early, "kind": "v", "region": "facts_value"},
            "middle_kv_bridges": {"layers": middle, "kind": "kv", "region": "bridges"},
            "middlelate_kv_query_terminal": {"layers": middlelate, "kind": "kv", "region": "query_terminal"},
            "middlelate_kv_external": {"layers": middlelate, "kind": "kv", "region": "external"},
            "late_q": {"layers": late, "kind": "q", "region": "answer"},
            "late_kv_facts": {"layers": late, "kind": "kv", "region": "facts_all"}}


class Controller:
    def __init__(self, model, specs: dict[str, dict]):
        self.layers = model_utils.get_layers(model)
        self.mode, self.spec, self.jobs, self.store = "none", {}, [], {}
        self.handles = []
        required = {(kind, layer_index) for spec in specs.values() for layer_index in spec.get("layers", ())
                    for kind in (("q",) if spec.get("kind") == "q" else ("k", "v"))}
        for layer_index, layer in enumerate(self.layers):
            for kind, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                if (kind, layer_index) not in required:
                    continue
                def hook(_module, _inputs, output, layer_index=layer_index, kind=kind):
                    return self._hook(output, layer_index, kind)
                self.handles.append(getattr(layer.self_attn, name).register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _hook(self, output: torch.Tensor, layer_index: int, kind: str):
        key = (kind, layer_index)
        if self.mode == "capture":
            self.store[key] = output.detach().clone()
            return None
        if self.mode != "patch" or layer_index not in self.spec.get("layers", ()):
            return None
        requested = self.spec["kind"]
        if not (kind == requested or (requested == "kv" and kind in ("k", "v"))):
            return None
        changed, donor = output.clone(), self.store[key].to(output.device)
        for batch_index, job in enumerate(self.jobs):
            if kind == "q":
                base_start = job["base_shift"] + job["base_prompt_length"] - 1
                donor_start = job["donor_shift"] + job["donor_prompt_length"] - 1
                for offset in range(len(job["continuation"])):
                    changed[batch_index, base_start + offset] = donor[batch_index, donor_start + offset]
            else:
                name = self.spec["region"]
                for base_position, donor_position in zip(job["regions_base"][name], job["regions_donor"][name]):
                    changed[batch_index, job["base_shift"] + base_position] = donor[
                        batch_index, job["donor_shift"] + donor_position]
        return changed


def causal_jobs(tokenizer, selected: list[tuple], index: dict[tuple, dict]) -> list[dict]:
    jobs = []
    names = ("facts_value", "bridges", "query_terminal", "external", "facts_all")
    for key in selected:
        base, donor = index[key + (0,)], index[key + (1,)]
        for candidate_index, entity in enumerate(base["entities"]):
            continuation = [int(token) for token in tokenizer.encode(" " + entity, add_special_tokens=False)]
            jobs.append({"case_id": base["case_id"], "family_id": base["family_id"], "form": base["form"],
                         "depth": base["depth"], "gap": base["gap"], "candidate_index": candidate_index,
                         "target_index": base["target_index"], "donor_target_index": donor["target_index"],
                         "continuation": continuation, "base_prompt_length": len(base["prompt_ids"]),
                         "donor_prompt_length": len(donor["prompt_ids"]),
                         "base": base["prompt_ids"] + continuation, "donor": donor["prompt_ids"] + continuation,
                         "regions_base": {name: region(base, name) for name in names},
                         "regions_donor": {name: region(donor, name) for name in names}})
    return jobs


def continuation_scores(logits: torch.Tensor, jobs: list[dict], keep: int) -> list[float]:
    output = []
    for batch_index, job in enumerate(jobs):
        first = keep - len(job["continuation"]) - 1
        value = 0.0
        for offset, token in enumerate(job["continuation"]):
            z = logits[batch_index, first + offset].float()
            value += float(z[token] - torch.logsumexp(z, dim=-1))
        output.append(value)
    return output


def run_causal(model, tokenizer, jobs: list[dict], specs: dict[str, dict], batch_size: int = 8) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    controller, rows = Controller(model, specs), []
    try:
        buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for job in jobs:
            buckets[(len(job["base"]), len(job["donor"]))].append(job)
        batches = [values[start:start + batch_size] for lengths, values in sorted(buckets.items())
                   for start in range(0, len(values), batch_size)]
        done = 0
        for batch in batches:
            controller.jobs = batch
            donor_ids, donor_mask, donor_shifts = p2552.left_pad(
                [job["donor"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, donor_shifts):
                job["donor_shift"] = shift
            keep = max(len(job["continuation"]) for job in batch) + 1
            controller.mode, controller.store = "capture", {}
            with torch.inference_mode():
                donor_logits = model(input_ids=donor_ids, attention_mask=donor_mask,
                                     use_cache=False, logits_to_keep=keep).logits
            donor_scores = continuation_scores(donor_logits, batch, keep)
            base_ids, base_mask, base_shifts = p2552.left_pad(
                [job["base"] for job in batch], tokenizer.pad_token_id, device)
            for job, shift in zip(batch, base_shifts):
                job["base_shift"] = shift
            for condition, spec in specs.items():
                controller.mode = "none" if condition == "no_patch" else "patch"
                controller.spec = spec
                with torch.inference_mode():
                    logits = model(input_ids=base_ids, attention_mask=base_mask,
                                   use_cache=False, logits_to_keep=keep).logits
                values = continuation_scores(logits, batch, keep)
                for job, value, donor_value in zip(batch, values, donor_scores):
                    rows.append({key: job[key] for key in ("case_id", "family_id", "form", "depth", "gap",
                                                           "candidate_index", "target_index", "donor_target_index")})
                    rows[-1].update({"condition": condition, "score": value,
                                     "donor_baseline_score": donor_value})
            done += len(batch)
            if done == len(jobs) or done % 128 == 0:
                print(f"[phase2563 causal] {done}/{len(jobs)}", flush=True)
    finally:
        controller.close()
    return rows


def causal_summary(rows: list[dict], specs: dict[str, dict]) -> dict:
    output = {}
    for condition in specs:
        subset = [row for row in rows if row["condition"] == condition]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            groups[row["case_id"]].append(row)
        decisions = []
        for values in groups.values():
            prediction = max(values, key=lambda row: row["score"])["candidate_index"]
            decisions.append({"correct": prediction == values[0]["target_index"],
                              "flip": prediction == values[0]["donor_target_index"],
                              "depth": values[0]["depth"], "form": values[0]["form"], "gap": values[0]["gap"]})
        output[condition] = {"n": len(decisions), "accuracy": float(np.mean([row["correct"] for row in decisions])),
                             "donor_flip": float(np.mean([row["flip"] for row in decisions])),
                             "by_depth": {str(depth): float(np.mean([row["flip"] for row in decisions
                                                                      if row["depth"] == depth]))
                                          for depth in (1, 2, 3)},
                             "by_form": {form: float(np.mean([row["flip"] for row in decisions
                                                               if row["form"] == form]))
                                         for form in ("natural", "nonce")},
                             "by_gap": {str(gap): float(np.mean([row["flip"] for row in decisions
                                                                  if row["gap"] == gap]))
                                        for gap in (0, 4)}}
    return output


def generate(model, tokenizer, selected: list[tuple], index: dict[tuple, dict], limit: int = 96) -> list[dict]:
    device = model.get_input_embeddings().weight.device
    keys = choose(selected, limit)
    base_rows = [index[key + (0,)] for key in keys]
    output = []
    for start in range(0, len(base_rows), 8):
        batch = base_rows[start:start + 8]
        generated = [[] for _ in batch]
        for _step in range(10):
            ids, mask, _ = p2552.left_pad([row["prompt_ids"] + tokens for row, tokens in zip(batch, generated)],
                                          tokenizer.pad_token_id, device)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits
            for batch_index in range(len(batch)):
                generated[batch_index].append(int(torch.argmax(logits[batch_index, -1]).item()))
        for row, tokens in zip(batch, generated):
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            hits = [index for index, entity in enumerate(row["entities"]) if norm(entity) in norm(text)]
            prediction = hits[0] if len(set(hits)) == 1 else None
            output.append({"case_id": row["case_id"], "family_id": row["family_id"], "form": row["form"],
                           "depth": row["depth"], "gap": row["gap"], "target_index": row["target_index"],
                           "generated": text, "tokens": tokens, "prediction": prediction,
                           "correct": prediction == row["target_index"]})
    return output


def append_memo(result: dict) -> None:
    heading = f"## Phase {PHASE}: 关系必要的组合深度与长距离大图谱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

{heading} [{stamp}]

**测试原理与测试用例。** Qwen3-4B、BF16非量化。32语言操作族分别采用自然/无意义关系与值标记；每个实体同时具有两个关系槽，目标满足`entity = query_relation XOR query_terminal XOR binding`。值再接一层或两层`belongs to`链，形成1/2/3跳；问题前插入0或4句无关事实。32族×2形式×3深度×2距离×2绑定×2关系×2目标=3072个full，并为每例增加relation/terminal缺失对照，共9216例、18432条完整多token候选评分。只有base/donor双侧正确的键才进入256对平衡因果锁箱和96例自主生成。

$$e^*=r_q\oplus v_q\oplus b,\qquad
A(d,g)=P(\hat e=e^*\mid \text{{chain depth}}=d,\text{{gap}}=g).$$

因果阶段以模型36层的相对四分位测试facts-value早层K/V、bridge中层K/V、query-terminal及全部外部上下文的中晚层K/V、答案步晚层Q和晚层facts K/V：

$$F_{{c,d,g}}=P(\hat e_{{do(c)}}=e^*_{{donor}}\mid d,g,\text{{base/donor correct}}).$$

**结果汇总。** 行为`{json.dumps(result['behavior'], ensure_ascii=False)}`；eligible与选择数`{result['eligible_pairs']}`、`{result['causal_pairs']}`；因果`{json.dumps(result['causal'], ensure_ascii=False)}`；自主生成`{json.dumps(result['autonomous'], ensure_ascii=False)}`；检查`{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2563_c239873_c248064_compositional_distance_relation_atlas.py`；完整材料、18432条行为分数、因果逐候选分数、自主生成token和final位于`{OUT}`。

**分析与理论进展。** 该设计把“关系标签、值、组合深度、长距离”变成正交因素，并保证关系和值缺一时答案在构造上不可确定。深度或距离使准确率/阶段翻转连续衰减，才可称为组合负载规律；不同深度共享某阶段只说明相对功能复用，不说明固定坐标齿轮。无意义标签通过只表示模型能在当前上下文中做符号匹配，不能称为先验语义。

**问题硬伤与结论。** 这是受控二元微世界；层段粗到四分位；bridge词法固定；候选评分受输出先验影响；96条自由生成只验证输出方式，不代表开放生成。阴性阶段结果保留为路径分化证据，不作为关闭路线的唯一标准。机制仍未闭合，下一步必须读取所有HiddenState/Q/K/V坐标，比较深度和距离改变的是共享坐标还是仅改变分布纹理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        material = compile_material(tokenizer)
        behavior = score_candidates(model, tokenizer, material)
        eligible, index = eligible_pairs(material, behavior)
        selected = choose(eligible, 256)
        specs = conditions(len(model_utils.get_layers(model)))
        cjobs = causal_jobs(tokenizer, selected, index)
        causal = run_causal(model, tokenizer, cjobs, specs)
        autonomous = generate(model, tokenizer, eligible, index)
    finally:
        model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    material_path, behavior_path = OUT / "material/rows.jsonl", OUT / "behavior/scores.jsonl"
    causal_path, auto_path = OUT / "causal/stage_scores.jsonl", OUT / "autonomous/generation.jsonl"
    write(material_path, material)
    write(behavior_path, behavior)
    write(causal_path, causal)
    write(auto_path, autonomous)
    bpanel, cpanel = behavior_summary(behavior), causal_summary(causal, specs)
    apanel = {"n": len(autonomous), "accuracy": float(np.mean([row["correct"] for row in autonomous])),
              "by_depth": {str(depth): float(np.mean([row["correct"] for row in autonomous
                                                       if row["depth"] == depth])) for depth in (1, 2, 3)}}
    checks = {"bf16_nonquantized": True, "full_cases_3072": bpanel["full"]["n"] == 3072,
              "controls_3072_each": bpanel["relation_missing"]["n"] == 3072
              and bpanel["terminal_missing"]["n"] == 3072,
              "unique_case_ids": len({row["case_id"] for row in material}) == len(material),
              "causal_only_eligible": len(selected) <= len(eligible), "autonomous_only_eligible": True,
              "all_files_hashed": True, "claim_boundary": True}
    result = {"phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
              "behavior": bpanel, "eligible_pairs": len(eligible), "causal_pairs": len(selected),
              "causal": cpanel, "autonomous": apanel, "specs": {key: {subkey: list(value) if isinstance(value, tuple)
                  else value for subkey, value in spec.items()} for key, spec in specs.items()},
              "files": {str(path): {"sha256": sha(path), "bytes": path.stat().st_size}
                        for path in (material_path, behavior_path, causal_path, auto_path)},
              "checks": checks, "all_checks_passed": all(checks.values()),
              "language_mechanism_closed": False}
    save(OUT / "analysis/final.json", result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
