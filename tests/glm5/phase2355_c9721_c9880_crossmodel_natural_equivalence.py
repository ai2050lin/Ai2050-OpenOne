#!/usr/bin/env python3
"""Sequential cross-model functional replication of the natural multi-future prompt atlas."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
P2352 = RESULT / "phase2352_c9241_c9400_natural_multifuture_transient_field"
OUT = RESULT / "phase2355_c9721_c9880_crossmodel_natural_equivalence"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"; VIS = ROOT / "frontend/public/vis_data/research_kernel"
PHASE = 2355; CAMPAIGN = "C9721-C9880"; MODEL_ORDER = ("qwen14b", "glm4", "deepseek7b")

sys.path.insert(0, str(TESTS))
import phase2315_c5041_c5100_active_response_contract as io  # noqa: E402
import phase2316_c5101_c5160_qwen4b_active_baseline as baseline  # noqa: E402
import phase2319_c5321_c5400_active_response_atlas_cleanup as atlas  # noqa: E402
import phase2339_c7401_c7600_crossmodel_fixed_ab_replication as crossmodel  # noqa: E402
import phase2353_c9401_c9560_conditional_equivalence_route_competition as route  # noqa: E402
import phase2352_c9241_c9400_natural_multifuture_transient_field as source  # noqa: E402
import model_utils  # noqa: E402

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_memmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def paths(key: str) -> dict[str, Path]:
    base = OUT / key
    return {"base": base, "rows": base / "material/rows.jsonl", "states": base / "raw/all_checkpoint.float16.npy",
            "decisions": base / "raw/decisions.float32.npy", "progress": base / "raw/progress.json", "final": base / "analysis/final.json"}


def compile_rows(tokenizer) -> tuple[list[dict], dict]:
    source_rows = io.read_rows(P2352 / "material/natural_multifuture_graphs.jsonl"); rows = []
    for source in source_rows:
        if source["surface"] != "natural" or source["state"] != 0 or source["query"] not in ("source", "terminal"): continue
        prompt_ids = [int(x) for x in tokenizer.encode(source["prompt"], add_special_tokens=False)]
        target_ids = [int(x) for x in tokenizer.encode(f" <answer>{source['target']}</answer>", add_special_tokens=False)]
        wrong_ids = [int(x) for x in tokenizer.encode(f" <answer>{source['foil']}</answer>", add_special_tokens=False)]
        rows.append({**source, "model_index": len(rows), "prompt_ids": prompt_ids, "target_ids": target_ids, "wrong_ids": wrong_ids})
    return rows, {"rows": len(rows), "families": 12, "languages": 2, "units": 16, "queries": ["source", "terminal"],
                  "token_length_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)]}


def candidate_score(model, device, batch: list[dict], key: str, captures: dict[int, torch.Tensor], pad: int) -> np.ndarray:
    combined = [r["prompt_ids"] + r[key] for r in batch]; ids, mask, positions = baseline.pad_right(combined, device, pad)
    captures.clear(); output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    scores = []
    for local, row in enumerate(batch):
        answer = row[key]; start = len(row["prompt_ids"]); pos = torch.arange(start - 1, start + len(answer) - 1, device=device)
        logits = model.lm_head(output.last_hidden_state[local, pos]).float(); tokens = torch.tensor(answer, dtype=torch.long, device=device)
        scores.append(float(F.log_softmax(logits, dim=-1)[torch.arange(len(answer), device=device), tokens].mean()))
    return np.asarray(scores, dtype=np.float32)


def collect(key: str, model, device, rows: list[dict]) -> dict:
    p = paths(key); modules = crossmodel.model_modules(model); dim = int(model.get_input_embeddings().weight.shape[1]); shape = (len(rows), len(modules), dim)
    if p["states"].exists() and p["decisions"].exists() and p["progress"].exists():
        completed = int(json.loads(p["progress"].read_text(encoding="utf-8"))["completed"])
        states = np.lib.format.open_memmap(p["states"], mode="r+"); decisions = np.lib.format.open_memmap(p["decisions"], mode="r+")
    else:
        completed = 0; p["states"].parent.mkdir(parents=True, exist_ok=True)
        states = np.lib.format.open_memmap(p["states"], mode="w+", dtype=np.float16, shape=shape)
        decisions = np.lib.format.open_memmap(p["decisions"], mode="w+", dtype=np.float32, shape=(len(rows), 4))
    captures = {}; handles = []
    for qpoint, module in enumerate(modules):
        def hook(_m, _i, value, qpoint=qpoint): captures[qpoint] = value[0] if isinstance(value, tuple) else value
        handles.append(module.register_forward_hook(hook))
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0); batch_size = int(crossmodel.MODEL_SPECS[key]["batch"])
    try:
        with torch.inference_mode():
            for start in range(completed, len(rows), batch_size):
                batch = rows[start:start + batch_size]; good = candidate_score(model, device, batch, "target_ids", captures, pad)
                for qpoint in range(len(modules)):
                    states[start:start + len(batch), qpoint] = torch.stack([captures[qpoint][i, len(r["prompt_ids"]) - 1]
                                                                          for i, r in enumerate(batch)]).float().cpu().numpy().astype(np.float16)
                bad = candidate_score(model, device, batch, "wrong_ids", captures, pad); margin = good - bad
                decisions[start:start + len(batch)] = np.stack([good, bad, margin, margin > 0], axis=1)
                states.flush(); decisions.flush(); save(p["progress"], {"completed": start + len(batch), "shape": list(shape)})
                if (start + len(batch)) % 96 == 0 or start + len(batch) == len(rows): print(f"[phase2355 {key}] {start + len(batch)}/{len(rows)}", flush=True)
    finally:
        for handle in handles: handle.remove()
        states.flush(); decisions.flush(); close_memmap(states); close_memmap(decisions)
    return {"shape": list(shape), "dimension": dim, "layers": len(modules) - 2, "batch_size": batch_size,
            "model_label": crossmodel.MODEL_SPECS[key]["label"], "quantization": crossmodel.MODEL_SPECS[key]["quant"]}


def behavior(key: str, rows: list[dict]) -> dict:
    decisions = np.load(paths(key)["decisions"], mmap_mode="r"); families = {}; qualified = []
    for family in sorted({r["family"] for r in rows}):
        cells = {}
        for language in ("en", "zh"):
            for partition in source.PARTITIONS:
                idx = [i for i, r in enumerate(rows) if r["family"] == family and r["language"] == language and r["partition"] == partition]
                cells[f"{language}:{partition}"] = float(np.mean(decisions[idx, 3]))
        families[family] = {"minimum": min(cells.values()), "cells": cells}
        if min(cells.values()) >= 0.75: qualified.append(family)
    result = {"overall": float(np.mean(decisions[:, 3])), "qualified": qualified, "families": families}; close_memmap(decisions); return result


def grouped(field: np.ndarray, rows: list[dict], labels: list[str], factor: str, a: Any, b: Any, partition: str) -> dict:
    prototypes = np.stack([field[[i for i, r in enumerate(rows) if r["family"] == label and r["partition"] in ("discovery", "confirmation") and r[factor] == a]].mean(axis=0, dtype=np.float64) for label in labels])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["family"] in labels and r["partition"] == partition and r[factor] == b: groups[(r["family"], r["unit"])].append(i)
    keys = sorted(groups); actual = np.stack([field[groups[k]].mean(axis=0, dtype=np.float64) for k in keys])
    return route.classify(prototypes, actual, np.asarray([labels.index(k[0]) for k in keys]))


def evaluate(field: np.ndarray, rows: list[dict], labels: list[str], partition: str) -> dict:
    axes = (("language", "en", "zh"), ("language", "zh", "en"), ("query", "source", "terminal"), ("query", "terminal", "source"))
    values = {f"{f}:{a}->{b}": grouped(field, rows, labels, f, a, b, partition) for f, a, b in axes}
    return {"transfers": values, "minimum_accuracy": min(v["accuracy"] for v in values.values()),
            "mean_accuracy": float(np.mean([v["accuracy"] for v in values.values()])),
            "maximum_distance_ratio": max(v["median_distance_ratio"] for v in values.values())}


def analyze(key: str, rows: list[dict], behavior_result: dict) -> dict:
    states = np.load(paths(key)["states"], mmap_mode="r"); labels = behavior_result["qualified"] or sorted({r["family"] for r in rows})
    train = np.asarray([r["partition"] in ("discovery", "confirmation") for r in rows]); trajectory = []; candidates = []
    for qpoint in range(states.shape[1]):
        signed = states[:, qpoint].astype(np.float32); residual = route.fit_residual(signed, rows, train, ("language", "query", "depth"))
        score = evaluate(residual, rows, labels, "fresh_confirmation"); sorted_score = evaluate(np.sort(residual, axis=1), rows, labels, "fresh_confirmation")
        trajectory.append({"qpoint": qpoint, "relative_depth": None if qpoint in (0, states.shape[1]-1) else (qpoint-1)/max(states.shape[1]-3,1),
                           "residual": {k:v for k,v in score.items() if k != "transfers"},
                           "sorted": {k:v for k,v in sorted_score.items() if k != "transfers"}})
        candidates.append((score["minimum_accuracy"], score["mean_accuracy"], -score["maximum_distance_ratio"], qpoint))
    qpoint = max(candidates)[3]; signed = states[:, qpoint].astype(np.float32); residual = route.fit_residual(signed, rows, train, ("language", "query", "depth"))
    lock = evaluate(residual, rows, labels, "fresh_lockbox"); sorted_lock = evaluate(np.sort(residual, axis=1), rows, labels, "fresh_lockbox")
    gate = {"qualified_count": len(behavior_result["qualified"]), "selected_qpoint": qpoint,
            "relative_depth": None if qpoint in (0, states.shape[1]-1) else (qpoint-1)/max(states.shape[1]-3,1),
            "lockbox_minimum": lock["minimum_accuracy"], "coordinate_advantage": lock["minimum_accuracy"]-sorted_lock["minimum_accuracy"],
            "distance_ratio": lock["maximum_distance_ratio"], "behavior_pass": len(behavior_result["qualified"]) >= 8,
            "descriptive_pass": lock["minimum_accuracy"] >= 0.30 and lock["minimum_accuracy"] >= sorted_lock["minimum_accuracy"] + 0.10 and lock["maximum_distance_ratio"] < 1.0}
    close_memmap(states); return {"labels": labels, "trajectory": trajectory, "lockbox": lock, "sorted_lockbox": sorted_lock, "gate": gate}


def publish(key: str, rows: list[dict], collection: dict, analysis: dict) -> dict:
    states = np.load(paths(key)["states"], mmap_mode="r"); qpoints = list(dict.fromkeys([0, analysis["gate"]["selected_qpoint"], states.shape[1]-1]))
    dataset_id = f"c9721_{key}_natural_multifuture_key_hiddenstate"; binary = VIS / f"{dataset_id}.float16.npy"
    out = atlas.create_binary(binary.name, len(rows)*len(qpoints), states.shape[-1], np.float16); metadata=[]; cursor=0
    for qpoint in qpoints:
        out[cursor:cursor+len(rows)] = states[:,qpoint]
        metadata.extend({"case_id":r["case_id"],"family":r["family"],"language":r["language"],"query":r["query"],
                         "depth":r["depth"],"partition":r["partition"],"unit":r["unit"],"qpoint":qpoint,
                         "relative_depth":None if qpoint in (0,states.shape[1]-1) else (qpoint-1)/max(states.shape[1]-3,1)} for r in rows); cursor += len(rows)
    out.flush(); close_memmap(out); close_memmap(states)
    return atlas.write_metadata(dataset_id, f"{collection['model_label']} natural multi-future key HiddenState", binary, metadata,
        collection["model_label"], "crossmodel_natural_multifuture_key_hiddenstate_v1", "model-local functional comparison; no coordinate-number alignment",
        "12 families x bilingual x source/terminal x 16 units, natural surface, state0",
        f"all {collection['dimension']} coordinates at embedding, model-selected checkpoint and final norm",
        {"phase":PHASE,"campaign":CAMPAIGN,"coordinate_count":collection["dimension"],"qpoints":qpoints,"quantization":collection["quantization"]})


def run_model(key: str) -> dict:
    p=paths(key)
    if p["final"].exists(): return json.loads(p["final"].read_text(encoding="utf-8"))
    progress = json.loads(p["progress"].read_text(encoding="utf-8")) if p["progress"].exists() else {}
    if p["rows"].exists() and p["states"].exists() and p["decisions"].exists() and int(progress.get("completed", 0)) == 768:
        rows=io.read_rows(p["rows"]); state=np.load(p["states"],mmap_mode="r"); shape=list(state.shape); close_memmap(state)
        collection={"shape":shape,"dimension":shape[-1],"layers":shape[1]-2,"batch_size":int(crossmodel.MODEL_SPECS[key]["batch"]),
                    "model_label":crossmodel.MODEL_SPECS[key]["label"],"quantization":crossmodel.MODEL_SPECS[key]["quant"]}
        audit={"rows":len(rows),"families":12,"languages":2,"units":16,"queries":["source","terminal"],"resumed_complete":True}
    else:
        model=None
        try:
            model,tokenizer,device=crossmodel.load_model(key); rows,audit=compile_rows(tokenizer); io.write_rows(p["rows"],rows); collection=collect(key,model,device,rows)
        finally:
            if model is not None: model_utils.release_model(model)
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    behavior_result=behavior(key,rows); analysis=analyze(key,rows,behavior_result); dataset=publish(key,rows,collection,analysis); verification=atlas.verify(dataset)
    if not all(v for k,v in verification.items() if k!="id"): raise RuntimeError((key,verification))
    raw_size=p["states"].stat().st_size; p["states"].unlink()
    result={"key":key,"material":audit,"collection":collection,"behavior":behavior_result,"analysis":analysis,
            "dataset":json.loads(json.dumps(dataset,default=str)),"verification":verification,
            "cleanup":{"bytes_reclaimed":raw_size,"deleted_ok":not p["states"].exists()}}
    save(p["final"],result); return result


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text=rf"""

## Phase {PHASE}: 三异构模型自然多未来条件等价功能复验（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 依次加载Qwen3-14B NF4、GLM4-9B INT8和DeepSeek-7B INT8，绝不同时驻留。每模型重新分词同一768条材料（12族×中英×source/terminal×16 units，natural、state0），比较完整目标与foil序列，并采集模型本地所有层、所有本地坐标。对有符号HiddenState减去训练分区语言/查询/图深主效应，在fresh_confirmation选模型本地层、fresh_lockbox裁决；只比较行为、迁移率、具体坐标相对排序优势和相对层深，不比较坐标编号。

$$
\Phi_m=(B_m,A_{{lang}},A_{{query}},\Delta A_{{coord-sort}},\rho_m^*),\qquad
\rho_m^*=\frac{{q_m^*-1}}{{L_m-1}}.
$$

**结果汇总。** 三模型 `{json.dumps(result['models'], ensure_ascii=False)}`；跨模型裁决 `{json.dumps(result['summary'], ensure_ascii=False)}`；可视化/核验 `{json.dumps(result['datasets'], ensure_ascii=False)}`；清理 `{json.dumps(result['cleanup'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2355_c9721_c9880_crossmodel_natural_equivalence.py`；结果 `tests/glm5/result/phase2355_c9721_c9880_crossmodel_natural_equivalence`；客户端三份`c9721_*`。

**理论进展、问题硬伤与结论。** 跨模型复验最多支持“不同模型在自己的坐标系和层深中出现相似功能图谱”；NF4/INT8低值场不可与Qwen4B FP16等精度，不能称共同物理坐标、同构流形或普遍齿轮。前两Phase已证明自主生成正确率显著低于teacher forcing、step>0瞬态不复现且保范数因果门失败，因此即使本Phase描述门通过也不能机制闭合。
"""
    with MEMO.open("a",encoding="utf-8",newline="\n") as h:h.write(text)


def main() -> None:
    final=OUT/"analysis/final.json"
    if final.exists():
        result=json.loads(final.read_text(encoding="utf-8"));append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2));return
    models={};datasets=[];cleanup={}
    for key in MODEL_ORDER:
        value=run_model(key);models[key]=value
        dataset=dict(value["dataset"]);dataset["metadata"]=Path(dataset["metadata"]);dataset["binary"]=Path(dataset["binary"])
        datasets.append(dataset);cleanup[key]=value["cleanup"]
    catalog=atlas.update_catalog(datasets);build=atlas.frontend_build();summary={key:{"behavior":v["behavior"]["overall"],
        "qualified":len(v["behavior"]["qualified"]),"qpoint":v["analysis"]["gate"]["selected_qpoint"],
        "relative_depth":v["analysis"]["gate"]["relative_depth"],"lockbox_minimum":v["analysis"]["gate"]["lockbox_minimum"],
        "coordinate_advantage":v["analysis"]["gate"]["coordinate_advantage"],"descriptive_pass":v["analysis"]["gate"]["descriptive_pass"]} for key,v in models.items()}
    checks={"models":len(models)==3,"assets":all(all(x for k,x in v["verification"].items() if k!="id") for v in models.values()),
            "sequential_cleanup":all(v["deleted_ok"] for v in cleanup.values()),"frontend_build":build["passed"]}
    result={"phase":PHASE,"campaign":CAMPAIGN,"model_order":list(MODEL_ORDER),"models":models,"summary":summary,
            "datasets":json.loads(json.dumps(datasets,default=str)),
            "catalog":json.loads(json.dumps(catalog,default=str)),"frontend_build":build,"cleanup":cleanup,"checks":checks,"all_checks_passed":all(checks.values())}
    save(final,result)
    if not result["all_checks_passed"]:raise RuntimeError(("phase2355_failed",checks))
    append_memo(result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
