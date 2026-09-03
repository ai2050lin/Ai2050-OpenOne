"""C1065-C1080 exact fresh semantics, sequential model workers and role-depth topology."""
from __future__ import annotations

import argparse
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
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CONTRACT_OUT = RESULT / "phase2247_c1001_c1016_natural_flagship_contract"
QWEN4_OUT = RESULT / "phase2248_c1017_c1030_qwen_natural_full_field"
OUT = RESULT / "phase2251_c1065_c1080_cross_model_fresh_panel"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as model_worker
import phase2247_c1001_c1016_natural_flagship_contract as contract


PHASE = 2251
CAMPAIGNS = tuple(f"C{i}" for i in range(1065, 1081))
MODELS = ("qwen3_14b", "glm4", "deepseek7b")


def save(path: Path, value: Any) -> None:
    contract.save(path, value)


def load(path: Path) -> Any:
    return contract.load(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    contract.write_rows(path, rows)


def read_rows(path: Path) -> list[dict]:
    return contract.read_rows(path)


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_code(text: str, row: dict) -> str | None:
    clean = text.strip().lower()
    hits = []
    for code in (row["true_code"], row["false_code"]):
        match = re.search(rf"\b{re.escape(code.lower())}\b", clean)
        if match:
            hits.append((match.start(), code))
    return min(hits)[1] if hits else None


def generation(model, tokenizer, device, rows: list[dict], model_name: str) -> list[dict]:
    batch_size = 1 if model_name == "qwen3_14b" else 8
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]
            ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, width - len(seq):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
            )
        for i, row in enumerate(batch):
            text = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_code(text, row)
            output.append({"case_id": row["case_id"], "text": text, "parsed": parsed,
                           "correct_answer": row["correct_answer"],
                           "correct": parsed == row["correct_answer"]})
        if start % 32 == 0:
            print(f"[{model_name}-generation] {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return output


def behavior_ledger(rows: list[dict], candidates: list[dict], generated: list[dict]) -> dict:
    c = {row["case_id"]: row for row in candidates}; g = {row["case_id"]: row for row in generated}
    families = {}
    for family in contract.FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        ca = float(np.mean([c[row["case_id"]]["correct"] for row in subset]))
        ga = float(np.mean([g[row["case_id"]]["correct"] for row in subset]))
        families[family] = {"rows": len(subset), "candidate_accuracy": ca,
                            "generation_accuracy": ga, "dual_qualified": min(ca, ga) >= 0.75}
    ca = float(np.mean([row["correct"] for row in candidates])); ga = float(np.mean([row["correct"] for row in generated]))
    return {"rows": len(rows), "candidate_accuracy": ca, "generation_accuracy": ga,
            "aggregate_dual_qualified": min(ca, ga) >= 0.75, "families": families,
            "parsed_generation_fraction": float(np.mean([row["parsed"] is not None for row in generated]))}


def capture_field(model, device, rows: list[dict], worker_out: Path, model_name: str) -> dict:
    base = model.model
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    dim = int(base.embed_tokens.weight.shape[1])
    path = worker_out / "raw/fresh_role_field.float16.npy"
    progress_path = worker_out / "raw/capture_progress.json"
    shape = (len(rows), len(modules), len(contract.ROLES), dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if path.exists() and progress_path.exists():
        progress = load(progress_path)
        if tuple(progress["shape"]) != shape:
            raise RuntimeError(("resume_shape", progress["shape"], shape))
        completed = int(progress["completed_rows"])
        field = np.lib.format.open_memmap(path, mode="r+")
    else:
        field = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        save(progress_path, {"shape": list(shape), "completed_rows": 0})
    captured = []

    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        row_i = completed
        while row_i < len(rows):
            # Disk-offloaded 14B weights dominate runtime. Pair only adjacent prompts
            # with exactly equal lengths so batching cannot introduce padding shifts.
            batch_size = 1
            if (model_name == "qwen3_14b" and row_i + 1 < len(rows)
                    and len(rows[row_i]["prompt_ids"]) == len(rows[row_i + 1]["prompt_ids"])):
                batch_size = 2
            batch = rows[row_i:row_i + batch_size]
            ids = torch.tensor([row["prompt_ids"] for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = mask.long().cumsum(-1) - 1
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError(("checkpoint_count", len(captured), len(modules)))
            for q, hidden in enumerate(captured):
                values = hidden.float().cpu().numpy().astype(np.float16)
                for batch_i, row in enumerate(batch):
                    for role_i, role in enumerate(contract.ROLES):
                        field[row_i + batch_i, q, role_i] = values[
                            batch_i, row["role_positions"][role][-1]]
            row_i += batch_size
            if row_i % 4 <= batch_size - 1:
                field.flush(); save(progress_path, {"shape": list(shape), "completed_rows": row_i})
                print(f"[cross-field] {row_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        field.flush(); close_mmap(field)
    save(progress_path, {"shape": list(shape), "completed_rows": len(rows)})
    write_rows(worker_out / "raw/field_index.jsonl", [{
        "hidden_index": i, "case_id": row["case_id"], "family": row["family"],
        "language": row["language"], "surface": row["surface"], "unit": row["unit"],
        "partition": row["partition"], "truth": row["truth"],
        "role_positions": row["role_positions"],
    } for i, row in enumerate(rows)])
    return {"ran": True, "path": str(path.relative_to(ROOT)), "shape": list(shape),
            "checkpoints": len(modules), "coordinates": dim}


def run_worker(model_name: str) -> dict:
    worker_out = OUT / model_name
    final_path = worker_out / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    raw = read_rows(CONTRACT_OUT / "material/fresh_broad_cases.jsonl")
    model = None
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        compiled = contract.compile_rows(tokenizer, raw)
        candidate_path = worker_out / "behavior/candidate.jsonl"
        generation_path = worker_out / "behavior/generation.jsonl"
        if candidate_path.exists() and generation_path.exists():
            candidate = read_rows(candidate_path)
            generated = read_rows(generation_path)
            expected_ids = [row["case_id"] for row in compiled]
            if ([row["case_id"] for row in candidate] != expected_ids
                    or [row["case_id"] for row in generated] != expected_ids):
                raise RuntimeError("saved behavior rows do not match the frozen compiled denominator")
            print(f"[{model_name}] reusing {len(compiled)} frozen behavior rows", flush=True)
        else:
            candidate = contract.prior.behavior_base.batch_behavior(
                model, device, compiled, batch_size=2 if model_name == "qwen3_14b" else 8)
            generated = generation(model, tokenizer, device, compiled, model_name)
            write_rows(candidate_path, candidate)
            write_rows(generation_path, generated)
        behavior = behavior_ledger(compiled, candidate, generated)
        save(worker_out / "behavior/ledger.json", behavior)
        field = capture_field(model, device, compiled, worker_out, model_name) if behavior["aggregate_dual_qualified"] else {
            "ran": False, "reason": "aggregate_dual_behavior_below_0.75"}
    finally:
        model_worker.release_model(model_name, model)
        gc.collect()
    checks = {"rows_complete": len(raw) == 384, "behavior_complete": behavior["rows"] == len(raw),
              "field_iff_qualified": field["ran"] == behavior["aggregate_dual_qualified"],
              "own_tokenizer_compilation": True}
    result = {"phase": PHASE, "model": model_name, "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "loader": loader,
              "placement": placement, "behavior": behavior, "field": field, "checks": checks,
              "all_checks_passed": all(checks.values()),
              "hashes": {"field": file_hash(ROOT / field["path"]) if field["ran"] else None},
              "strict_conclusion": "Internal field exists only after the exact fresh denominator passes both behavior interfaces."}
    save(final_path, result)
    print(json.dumps({"model": model_name, "behavior": behavior, "field": field, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


def pair_rows(index: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in index:
        key = (row["family"], row["language"], row["unit"], row["surface"])
        groups[key][bool(row["truth"])] = row
    return [{"family": key[0], "language": key[1], "unit": key[2], "surface": key[3],
             "false_index": values[False]["hidden_index"], "true_index": values[True]["hidden_index"]}
            for key, values in sorted(groups.items()) if set(values) == {False, True}]


def profiles(field, index: list[dict]) -> dict:
    rows = pair_rows(index)
    output = {}
    for family in contract.FAMILIES:
        for unit in sorted({row["unit"] for row in rows}):
            subset = [row for row in rows if row["family"] == family and row["unit"] == unit]
            responses = np.stack([
                np.asarray(field[row["true_index"]], np.float32) - np.asarray(field[row["false_index"]], np.float32)
                for row in subset
            ])
            rms = np.sqrt(np.mean(responses * responses, axis=(0, 3)))
            rms /= np.sqrt(np.sum(rms * rms, axis=1, keepdims=True)) + 1e-12
            output[(family, unit)] = rms.astype(np.float32)
    return output


def resample(profile: np.ndarray, steps: int = 64) -> np.ndarray:
    source = np.linspace(0.0, 1.0, profile.shape[0]); target = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(target, source, profile[:, role_i]) for role_i in range(profile.shape[1])], axis=1)


def retrieval(source: dict, target: dict, centered: bool) -> dict:
    units = sorted({unit for _family, unit in source})
    rows = []
    for family, unit in sorted(source):
        query = resample(source[(family, unit)])
        if centered:
            query -= np.mean([resample(source[(other, unit)]) for other in contract.FAMILIES], axis=0)
        distances = {}
        for candidate in contract.FAMILIES:
            prototypes = []
            for other_unit in units:
                if other_unit == unit or (candidate, other_unit) not in target:
                    continue
                value = resample(target[(candidate, other_unit)])
                if centered:
                    value -= np.mean([resample(target[(other, other_unit)]) for other in contract.FAMILIES], axis=0)
                prototypes.append(value)
            prototype = np.mean(prototypes, axis=0)
            distances[candidate] = float(np.mean(np.abs(query - prototype)))
        predicted = min(contract.FAMILIES, key=lambda candidate: (distances[candidate], contract.FAMILIES.index(candidate)))
        wrong = min(value for key, value in distances.items() if key != family)
        rows.append({"family": family, "unit": unit, "predicted": predicted, "correct": predicted == family,
                     "same_family_distance": distances[family], "nearest_wrong_distance": wrong,
                     "margin": wrong - distances[family]})
    return {"queries": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])),
            "median_margin": float(np.median([row["margin"] for row in rows])), "errors": [row for row in rows if not row["correct"]]}


def qwen4_source() -> tuple[Any, list[dict]]:
    final = load(QWEN4_OUT / "analysis/final.json")
    field = np.load(ROOT / final["field"]["path"], mmap_mode="r")
    all_index = read_rows(QWEN4_OUT / "raw/field_index.jsonl")
    selected = [row for row in all_index if row["panel"] == "natural_broad" and row["fresh"]]
    indices = [row["hidden_index"] for row in selected]
    view = np.asarray(field[indices])
    close_mmap(field)
    remapped = [{**row, "hidden_index": i} for i, row in enumerate(selected)]
    return view, remapped


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    if marker in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    behavior = {name: {"candidate": row["behavior"]["candidate_accuracy"],
                       "generation": row["behavior"]["generation_accuracy"],
                       "qualified": row["behavior"]["aggregate_dual_qualified"],
                       "field": row["field"]}
                for name, row in result["models"].items()}
    text = f"""

## Phase {PHASE}: 同义fresh分母的顺序跨模型角色深度图谱（C1065-C1080） [{stamp}]

**测试原理与用例。** Qwen3-14B、GLM4和DeepSeek-7B逐个加载，每个模型用自己的tokenizer重新编译同一384条fresh宽族材料；候选与自由生成均不低于0.75才采集六角色全部坐标。跨模型不比较坐标编号，而是在各模型内部先计算真假响应的全坐标RMS角色曲线，再重采样到64个相对深度点，做双向留一单元族检索。

**公式。** 模型内角色响应强度为：

$$
E_{{m,f,u,q,r}}=\\sqrt{{\\frac{{1}}{{Nd_m}}\\sum_{{i,j}}(\\Delta H^m_{{i,q,r,j}})^2}},\\qquad
\\widetilde E=E/(\\lVert E_{{q,:}}\\rVert_2+\\epsilon).
$$

模型间只比较 `\\widetilde E` 的角色-相对深度拓扑，不声明物理坐标同构。

**结果汇总。** 各模型行为与字段账为 `{json.dumps(behavior, ensure_ascii=False)}`。合格模型对的双向raw与单元中心化检索为 `{json.dumps(result['topology'], ensure_ascii=False)}`。

**分析与理论进展。** 高双向检索表示不同尺度模型可能形成相似的族条件角色时序；它不是相同坐标、相同参数或相同因果电路。未通过双行为的模型内部场严格记NA，不用输出接口失败推断“模型没有语义”。理论主体与RDC不变。

**问题、硬伤与结论。** RMS仍把坐标压成角色强度，只用于跨维度模型的功能比较；完整坐标原始场仍保存在模型内分析中。样本只有8个fresh单元和受控双语模板；Qwen14使用磁盘卸载，数值类型与Qwen4不同；人类自然度盲评仍为NA。工程检查 `{result['all_checks_passed']}`。

**相关文件。** 脚本 `tests/glm5/phase2251_c1065_c1080_cross_model_fresh_panel.py`；结果 `tests/glm5/result/phase2251_c1065_c1080_cross_model_fresh_panel`。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def synthesize() -> dict:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        return load(final_path)
    workers = {name: load(OUT / name / "analysis/final.json") for name in MODELS}
    q4_final = load(QWEN4_OUT / "analysis/final.json")
    models = {"qwen3_4b": {"behavior": q4_final["behavior"], "field": q4_final["field"]}, **workers}
    q4_field, q4_index = qwen4_source(); q4_profiles = profiles(q4_field, q4_index)
    topology = {}
    try:
        for name, worker in workers.items():
            if not worker["field"]["ran"]:
                topology[name] = {"status": "NA_behavior_unqualified"}
                continue
            field = np.load(ROOT / worker["field"]["path"], mmap_mode="r")
            index = read_rows(OUT / name / "raw/field_index.jsonl")
            target = profiles(field, index)
            topology[name] = {
                "qwen4_to_model_raw": retrieval(q4_profiles, target, False),
                "model_to_qwen4_raw": retrieval(target, q4_profiles, False),
                "qwen4_to_model_centered": retrieval(q4_profiles, target, True),
                "model_to_qwen4_centered": retrieval(target, q4_profiles, True),
            }
            close_mmap(field)
    finally:
        del q4_field
    checks = {"all_workers_complete": all(row["all_checks_passed"] for row in workers.values()),
              "exact_denominator": all(row["behavior"]["rows"] == 384 for row in workers.values()),
              "qualified_fields_only": all(row["field"]["ran"] == row["behavior"]["aggregate_dual_qualified"] for row in workers.values()),
              "topology_for_each_worker": set(topology) == set(MODELS)}
    result = {"phase": PHASE, "campaigns": list(CAMPAIGNS), "status": "closed",
              "timestamp": datetime.now().astimezone().isoformat(), "models": models,
              "topology": topology, "checks": checks, "all_checks_passed": all(checks.values()),
              "strict_conclusion": "Cross-model results are relative-depth role topologies. Coordinate identity and causal-circuit isomorphism are not tested.",
              "next_authorization": "Export important Qwen4/Qwen14 full-coordinate atlases and audit cleanup; no more tuning on this denominator."}
    save(final_path, result); append_memo(result)
    print(json.dumps({"models": {k: v["behavior"] for k, v in models.items()},
                      "topology": topology, "checks": checks}, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODELS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
    else:
        synthesize()


if __name__ == "__main__":
    main()
